import argparse
import copy
import datetime
import hashlib
import models
import numpy as np
import os
import shutil
import time
import round_log
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed_rolex import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from round_log import TransparencyLog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}
cfg['local_train_size'] = 10
cfg['noise_scale'] = None
cfg['distribute_init_val'] = 0.25
cfg['file_output'] = "New_Tables/MNIST_Rolex_TEST"
# -------------------------------------------------------------------------
# Prototype-2 defense configuration
# -------------------------------------------------------------------------
cfg.setdefault('validation_response', 'zero_change')
cfg.setdefault('round_log_dir', os.path.join('output', 'round_log', 'prototype2'))
cfg.setdefault('verifier_val_size', 2)
cfg.setdefault('verifier_max_loss_increase', 0.35)
cfg.setdefault('verifier_max_acc_drop', 0.20)
cfg.setdefault('verifier_min_relative_change', 1.0e-6)
full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Max_PSNR Max_Recovered\n")

def compare_local_parameters(received, expected, atol=1e-6, rtol=1e-4):
    for k in expected:
        if k not in received:
            return False, f'missing key: {k}'

        recv = received[k]
        exp = expected[k]

        if recv.shape != exp.shape:
            return False, f'shape mismatch for {k}: got={tuple(recv.shape)} expected={tuple(exp.shape)}'

        recv = recv.to(device=exp.device, dtype=exp.dtype)

        if not torch.allclose(recv, exp, atol=atol, rtol=rtol):
            max_diff = torch.max(torch.abs(recv - exp)).item()
            return False, f'value mismatch for {k}: max_diff={max_diff:.6e}'

    return True, 'ok'


def extract_expected_local_parameters(federation, user_idx, param_idx):
    expected_local_parameters, _ = federation.extract_honest_local_parameters(
        user_idx, param_idx=param_idx
    )
    return expected_local_parameters


def evaluate_local_parameters(local_parameters, model_rate, data_loader, label_split, max_steps=None):
    metric = Metric()
    model = eval('models.{}(model_rate=model_rate).to(cfg["device"])'.format(cfg['model_name']))
    model.load_state_dict(local_parameters)
    model.train(False)

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    with torch.no_grad():
        for i, input in enumerate(data_loader):
            if max_steps is not None and i >= max_steps:
                break

            input = collate(input)
            input_size = input['img'].size(0)
            input['label_split'] = torch.tensor(label_split)
            input = to_device(input, cfg['device'])

            output = model(input)
            evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)

            total_loss += float(evaluation['Local-Loss']) * input_size
            total_acc += float(evaluation['Local-Accuracy']) * input_size
            total_seen += input_size

    if total_seen == 0:
        return {'Local-Loss': float('inf'), 'Local-Accuracy': 0.0}

    return {
        'Local-Loss': total_loss / total_seen,
        'Local-Accuracy': total_acc / total_seen,
    }


def relative_model_change(prev_local_parameters, cand_local_parameters):
    prev_vec = []
    cand_vec = []

    for k in prev_local_parameters:
        prev_vec.append(prev_local_parameters[k].detach().float().reshape(-1).cpu())
        cand_vec.append(cand_local_parameters[k].detach().float().reshape(-1).cpu())

    prev_vec = torch.cat(prev_vec)
    cand_vec = torch.cat(cand_vec)

    denom = torch.norm(prev_vec, p=2).item() + 1.0e-12
    return float(torch.norm(cand_vec - prev_vec, p=2).item() / denom)

def clone_state_dict(state_dict):
    cloned = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned


def hash_local_parameter_dict(local_parameters):
    hasher = hashlib.sha256()
    for key in sorted(local_parameters.keys()):
        tensor = local_parameters[key].detach().cpu().contiguous()
        hasher.update(key.encode('utf-8'))
        hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
        hasher.update(str(tensor.dtype).encode('utf-8'))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def state_dict_relative_l2(a_state, b_state):
    a_vec = []
    b_vec = []
    for k in a_state:
        if torch.is_tensor(a_state[k]) and torch.is_tensor(b_state[k]):
            a_vec.append(a_state[k].detach().float().reshape(-1).cpu())
            b_vec.append(b_state[k].detach().float().reshape(-1).cpu())
    if not a_vec:
        return 0.0
    a_vec = torch.cat(a_vec)
    b_vec = torch.cat(b_vec)
    denom = torch.norm(a_vec, p=2).item() + 1.0e-12
    return float(torch.norm(b_vec - a_vec, p=2).item() / denom)


def add_uniform_noise_to_state_dict(state_dict, noise_scale):
    noisy = clone_state_dict(state_dict)

    if noise_scale is None or float(noise_scale) <= 0.0:
        return noisy, {
            'noise_enabled': False,
            'noise_scale': float(noise_scale or 0.0),
            'noise_rel_l2': 0.0,
            'noise_max_abs': 0.0,
            'noise_mean_abs': 0.0,
        }

    total_noise_sq = 0.0
    total_base_sq = 0.0
    max_abs = 0.0
    total_abs = 0.0
    total_count = 0

    for k, v in noisy.items():
        if not torch.is_tensor(v) or not torch.is_floating_point(v):
            continue

        mean_abs = torch.mean(torch.abs(v)).item()
        if not np.isfinite(mean_abs) or mean_abs == 0.0:
            continue

        delta = float(noise_scale) * mean_abs
        noise = torch.empty_like(v).uniform_(-delta, delta)
        noisy[k] = v + noise

        total_noise_sq += float(torch.sum(noise.float() ** 2).item())
        total_base_sq += float(torch.sum(v.float() ** 2).item())

        abs_noise = torch.abs(noise).detach().float()
        max_abs = max(max_abs, float(torch.max(abs_noise).item()))
        total_abs += float(torch.sum(abs_noise).item())
        total_count += int(abs_noise.numel())

    rel_l2 = float(np.sqrt(total_noise_sq) / (np.sqrt(total_base_sq) + 1.0e-12))
    mean_abs = float(total_abs / max(total_count, 1))

    return noisy, {
        'noise_enabled': True,
        'noise_scale': float(noise_scale),
        'noise_rel_l2': rel_l2,
        'noise_max_abs': max_abs,
        'noise_mean_abs': mean_abs,
    }


def build_candidate_parent_for_commitment(epoch, federation, honest_aggregated_state, logger):
    """
    Build the parent model the server will actually commit to the committee.

    Honest path:
      candidate = honest aggregated parent

    Malicious replay path on attack_source_round:
      candidate = replay of the parent used in this round (federation.initial_parent_state),
      optionally with additive noise
    """
    attack_enabled = bool(cfg.get('attack_commit_enabled', True))
    attack_source_round = int(cfg.get('attack_source_round', 3))
    attack_replay_round = int(cfg.get('attack_replay_round', 4))
    commit_noise_scale = float(cfg.get('attack_commit_noise_scale', 0.0))

    previous_parent_state = clone_state_dict(federation.initial_parent_state)
    honest_parent_state = clone_state_dict(honest_aggregated_state)

    previous_hash = round_log.hash_state_dict(previous_parent_state)
    honest_hash = round_log.hash_state_dict(honest_parent_state)

    if attack_enabled and int(epoch) == attack_source_round:
        candidate_state = clone_state_dict(previous_parent_state)
        candidate_source = 'replay_previous_parent'
        noise_report = {
            'noise_enabled': False,
            'noise_scale': 0.0,
            'noise_rel_l2': 0.0,
            'noise_max_abs': 0.0,
            'noise_mean_abs': 0.0,
        }

        if commit_noise_scale > 0.0:
            candidate_state, noise_report = add_uniform_noise_to_state_dict(candidate_state, commit_noise_scale)
            candidate_source = 'replay_previous_parent_plus_noise'
    else:
        candidate_state = clone_state_dict(honest_parent_state)
        candidate_source = 'honest_aggregate'
        noise_report = {
            'noise_enabled': False,
            'noise_scale': 0.0,
            'noise_rel_l2': 0.0,
            'noise_max_abs': 0.0,
            'noise_mean_abs': 0.0,
        }

    candidate_hash = round_log.hash_state_dict(candidate_state)

    candidate_meta = {
        'candidate_source': candidate_source,
        'round_produced': int(epoch),
        'target_parent_round': int(epoch) + 1,
        'attack_source_round': attack_source_round,
        'attack_replay_round': attack_replay_round,
        'previous_parent_hash': previous_hash,
        'honest_aggregated_hash': honest_hash,
        'candidate_hash': candidate_hash,
        'relative_change_vs_previous_parent': state_dict_relative_l2(previous_parent_state, candidate_state),
        'relative_change_vs_honest_aggregated': state_dict_relative_l2(honest_parent_state, candidate_state),
        **noise_report,
    }

    if cfg.get('debug_attack_replay', False):
        logger.append({
            'info': [
                f'[REPLAY] round_produced={epoch}',
                f'[REPLAY] target_parent_round={epoch + 1}',
                f'[REPLAY] source={candidate_meta["candidate_source"]}',
                f'[REPLAY] previous_parent_hash={candidate_meta["previous_parent_hash"]}',
                f'[REPLAY] honest_aggregated_hash={candidate_meta["honest_aggregated_hash"]}',
                f'[REPLAY] candidate_hash={candidate_meta["candidate_hash"]}',
                f'[REPLAY] rel_change_vs_prev={candidate_meta["relative_change_vs_previous_parent"]:.6e}',
                f'[REPLAY] rel_change_vs_honest={candidate_meta["relative_change_vs_honest_aggregated"]:.6e}',
                f'[REPLAY] noise_enabled={candidate_meta["noise_enabled"]}',
                f'[REPLAY] noise_scale={candidate_meta["noise_scale"]}',
                f'[REPLAY] noise_rel_l2={candidate_meta["noise_rel_l2"]:.6e}',
                f'[REPLAY] noise_max_abs={candidate_meta["noise_max_abs"]:.6e}',
                f'[REPLAY] noise_mean_abs={candidate_meta["noise_mean_abs"]:.6e}',
            ]
        }, 'train', mean=False)

    return candidate_state, candidate_meta

def select_verifiers(local, user_idx, federation):
    """
    Pick at least one verifier per cohort from the active clients of this round.
    """
    selected = OrderedDict()
    for m in range(len(user_idx)):
        rate = float(federation.model_rate[user_idx[m]])
        if rate not in selected:
            selected[rate] = (m, int(user_idx[m]), local[m])
    return list(selected.values())


def verify_and_commit_candidate_parent(
    round_log,
    epoch,
    candidate_state_dict,
    local,
    user_idx,
    federation,
    label_split,
    logger,
    candidate_meta=None,
):
    """
    Prototype-2 verifier rule:
    - one verifier per cohort
    - compare candidate parent vs previous approved parent
    - use small private validation and relative model change
    - write approval / rejection into the non-server-controlled log
    """
    previous_parent_record = round_log.get_latest_approved_parent()
    previous_parent_state = round_log.load_parent_state_dict(previous_parent_record)

    committee = select_verifiers(local, user_idx, federation)
    verifier_reports = []

    if cfg.get('debug_commitment', False):
        committee_desc = [
            f'(slot={slot}, user={verifier_user_id}, rate={float(verifier_local.model_rate)})'
            for slot, verifier_user_id, verifier_local in committee
        ]
        logger.append({
            'info': [
                f'[COMMIT] target_parent_round={epoch + 1}',
                f'[COMMIT] candidate_source={candidate_meta.get("candidate_source", "unknown") if candidate_meta else "unknown"}',
                f'[COMMIT] committee={committee_desc}',
            ]
        }, 'train', mean=False)

    for _, verifier_user_id, verifier_local in committee:
        prev_federation = Federation(
            epoch + 1,
            copy.deepcopy(previous_parent_state),
            cfg['model_rate'],
            label_split
        )

        cand_federation = Federation(
            epoch + 1,
            copy.deepcopy(candidate_state_dict),
            cfg['model_rate'],
            label_split
        )

        prev_local_parameters, _ = prev_federation.extract_honest_local_parameters([verifier_user_id])
        cand_local_parameters, _ = cand_federation.extract_honest_local_parameters([verifier_user_id])

        prev_local_parameters = prev_local_parameters[0]
        cand_local_parameters = cand_local_parameters[0]

        prev_eval = evaluate_local_parameters(
            prev_local_parameters,
            prev_federation.model_rate[verifier_user_id],
            verifier_local.data_loader,
            verifier_local.label_split,
            max_steps=cfg['verifier_val_size'],
        )

        cand_eval = evaluate_local_parameters(
            cand_local_parameters,
            cand_federation.model_rate[verifier_user_id],
            verifier_local.data_loader,
            verifier_local.label_split,
            max_steps=cfg['verifier_val_size'],
        )

        rel_change = relative_model_change(prev_local_parameters, cand_local_parameters)
        cohort_rate = float(cand_federation.model_rate[verifier_user_id])

        approved = True
        reason = 'ok'

        if cand_eval['Local-Loss'] > prev_eval['Local-Loss'] + cfg['verifier_max_loss_increase']:
            approved = False
            reason = 'validation loss increased too much'
        elif cand_eval['Local-Accuracy'] + cfg['verifier_max_acc_drop'] < prev_eval['Local-Accuracy']:
            approved = False
            reason = 'validation accuracy dropped too much'
        elif rel_change < cfg['verifier_min_relative_change']:
            approved = False
            reason = 'candidate too similar to previous approved parent'

        report = {
            'user_id': int(verifier_user_id),
            'cohort_rate': cohort_rate,
            'approved': approved,
            'reason': reason,
            'prev_eval': prev_eval,
            'cand_eval': cand_eval,
            'relative_change': rel_change,
        }
        verifier_reports.append(report)

        if cfg.get('debug_commitment', False):
            logger.append({
                'info': [
                    f'[COMMIT][Verifier {verifier_user_id}] cohort_rate={cohort_rate}',
                    f'[COMMIT][Verifier {verifier_user_id}] approved={approved}',
                    f'[COMMIT][Verifier {verifier_user_id}] reason={reason}',
                    f'[COMMIT][Verifier {verifier_user_id}] prev_loss={prev_eval["Local-Loss"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] cand_loss={cand_eval["Local-Loss"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] prev_acc={prev_eval["Local-Accuracy"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] cand_acc={cand_eval["Local-Accuracy"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] rel_change={rel_change:.6e}',
                ]
            }, 'train', mean=False)

    active_user_model_rates = {
        int(uid): float(federation.model_rate[uid]) for uid in user_idx
    }

    event = round_log.record_candidate_parent(
        epoch=epoch,
        state_dict=copy.deepcopy(candidate_state_dict),
        active_users=user_idx,
        active_user_model_rates=active_user_model_rates,
        verifier_reports=verifier_reports,
        candidate_metadata=candidate_meta,
    )

    logger.append({
        'info': [
            f'Prototype-2 commitment check for round {epoch + 1}',
            f'candidate source: {candidate_meta.get("candidate_source", "unknown") if candidate_meta else "unknown"}',
            f'commitment approved: {event["approved"]}',
            f'approved cohorts: {event["approved_cohort_rates"]}',
            f'required cohorts: {event["required_cohort_rates"]}',
            f'candidate hash: {event["model_hash"]}',
        ]
    }, 'train', mean=False)

    return event

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        print('---------------------------------------')
        runExperiment()
    return

def move_state_dict_to_device(state_dict, device):
    moved = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved

def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    # print("Run Experiment: seed = ", seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()

    round_log = TransparencyLog(cfg['round_log_dir'])
    round_log.bootstrap_initial_parent(global_parameters, parent_for_round=1)

    model_history_block2 = {}
    model_history_block2['blocks.2.weight'] = []
    model_history_block2['blocks.2.bias'] = []
    model_history_block2['blocks.7.weight'] = []
    model_history_block2['blocks.7.bias'] = []
    model_history_block2['blocks.12.weight'] = []
    model_history_block2['blocks.12.bias'] = []
    model_history_block2['blocks.17.weight'] = []
    model_history_block2['blocks.17.bias'] = []
    model_history_block2['blocks.0.bias'] = []

    model_history_fcnn = {}
    model_history_fcnn['layers.0.weight'] = []
    model_history_fcnn['layers.0.bias'] = []


    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)

        approved_parent_record = round_log.get_latest_approved_parent()
        approved_parent_state = round_log.load_parent_state_dict(approved_parent_record)
        approved_parent_state = move_state_dict_to_device(approved_parent_state, cfg['device'])
        global_parameters = copy.deepcopy(approved_parent_state)

        federation = Federation(epoch, global_parameters, cfg['model_rate'], label_split)
        train(
            model_history_block2,
            model_history_fcnn,
            dataset['train'],
            data_split['train'],
            label_split,
            federation,
            model,
            optimizer,
            logger,
            epoch,
            round_log
        )
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(model_history_block2, model_history_fcnn, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, round_log):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, round_log, logger)
    distributed_local_parameters = copy.deepcopy(local_parameters)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']

    start_time = time.time()

    img_list = None
    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]] # This line modifies the learning rate based on user. 
        (local_parameters[m], img_data) = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if img_data:
            img_list = copy.deepcopy(img_data)
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])   
    
    federation.combine(local_parameters, param_idx, user_idx)

    honest_aggregated_state = clone_state_dict(federation.global_parameters)

    candidate_parent_state, candidate_meta = build_candidate_parent_for_commitment(
        epoch,
        federation,
        honest_aggregated_state,
        logger,
    )

    commitment_event = verify_and_commit_candidate_parent(
        round_log,
        epoch,
        copy.deepcopy(candidate_parent_state),
        local,
        user_idx,
        federation,
        label_split,
        logger,
        candidate_meta=candidate_meta,
    )

    final_parent_state = None

    if commitment_event['approved']:
        final_parent_state = clone_state_dict(candidate_parent_state)

    else:
        rejection_response = cfg.get('commit_rejection_response', 'rollback_previous')

        logger.append({
            'info': [
                f'[COMMIT] primary candidate rejected for round {epoch + 1}',
                f'[COMMIT] rejection_response={rejection_response}',
                f'[COMMIT] rejected_commitment_id={commitment_event["commitment_id"]}',
            ]
        }, 'train', mean=False)

        if rejection_response == 'abort':
            raise RuntimeError(
                f'Candidate parent rejected for round {epoch + 1}: {commitment_event["commitment_id"]}'
            )

        elif rejection_response == 'fallback_honest':
            fallback_meta = {
                'candidate_source': 'honest_fallback_after_reject',
                'round_produced': int(epoch),
                'target_parent_round': int(epoch) + 1,
                'previous_parent_hash': round_log.hash_state_dict(federation.initial_parent_state),
                'honest_aggregated_hash': round_log.hash_state_dict(honest_aggregated_state),
                'candidate_hash': round_log.hash_state_dict(honest_aggregated_state),
                'relative_change_vs_previous_parent': state_dict_relative_l2(
                    federation.initial_parent_state, honest_aggregated_state
                ),
                'relative_change_vs_honest_aggregated': 0.0,
                'noise_enabled': False,
                'noise_scale': 0.0,
                'noise_rel_l2': 0.0,
                'noise_max_abs': 0.0,
                'noise_mean_abs': 0.0,
                'fallback_triggered_by_commitment_id': commitment_event['commitment_id'],
            }

            fallback_event = verify_and_commit_candidate_parent(
                round_log,
                epoch,
                copy.deepcopy(honest_aggregated_state),
                local,
                user_idx,
                federation,
                label_split,
                logger,
                candidate_meta=fallback_meta,
            )

            if fallback_event['approved']:
                final_parent_state = clone_state_dict(honest_aggregated_state)
            else:
                approved_parent_record = round_log.get_latest_approved_parent()
                rollback_state = round_log.load_parent_state_dict(approved_parent_record)
                rollback_state = move_state_dict_to_device(rollback_state, cfg['device'])
                final_parent_state = clone_state_dict(rollback_state)

        else:
            approved_parent_record = round_log.get_latest_approved_parent()
            rollback_state = round_log.load_parent_state_dict(approved_parent_record)
            rollback_state = move_state_dict_to_device(rollback_state, cfg['device'])
            final_parent_state = clone_state_dict(rollback_state)

    federation.global_parameters = clone_state_dict(final_parent_state)
    global_model.load_state_dict(final_parent_state)
    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())

    # Append to the model history. 
    if cfg['model_name'] == 'fcnn':
        targetWeights = ['layers.0.weight']
        targetBiases = ['layers.0.bias']

        attack_source_round = int(cfg.get('attack_source_round', 3))
        attack_replay_round = int(cfg.get('attack_replay_round', 4))

        for m in range(num_active_users):
            weight_grad = None
            bias_grad = None

            max_pearson_overall = 0.0
            max_psnr_overall = 0.0
            max_pearson_list = []
            max_psnr_list = []
            num_recovered_list = []

            # Only target the 0.25 cohort
            if federation.model_rate[user_idx[m]] != 0.25:
                continue

            for k, v in global_model_state_dict_copy.items():
                if k in targetWeights or k in targetBiases:
                    model_history_fcnn[k].append(global_model_state_dict_copy[k])

                    # Run leakage only on the replay round, not hardcoded round 2
                    if epoch == attack_replay_round and len(model_history_fcnn[k]) >= 2:
                        distributed_model = distributed_local_parameters[m][k]

                        if cfg.get('debug_attack_replay', False):
                            print(
                                f"[REPLAY][LEAKAGE] epoch={epoch} user={user_idx[m]} key={k} "
                                f"rate={federation.model_rate[user_idx[m]]} "
                                f"dist_shape={tuple(distributed_model.shape)} "
                                f"hist_len={len(model_history_fcnn[k])}",
                                flush=True
                            )
                        if k in targetWeights:
                            weight_grad = fcnn_leakage(
                                k,
                                user_idx[m],
                                num_active_users,
                                federation.model_rate[user_idx[m]],
                                local_parameters[m][k],
                                model_history_fcnn[k],
                                distributed_model
                            )
                        else:
                            bias_grad = fcnn_leakage(
                                k,
                                user_idx[m],
                                num_active_users,
                                federation.model_rate[user_idx[m]],
                                local_parameters[m][k],
                                model_history_fcnn[k],
                                distributed_model
                            )

            if weight_grad is not None and bias_grad is not None:
                bias_grad_sum = torch.abs(torch.sum(bias_grad)).item()

                if bias_grad_sum != 0.0:
                    if img_list is not None and len(img_list) > 0:
                        (max_pearson, max_psnr, num_recovered) = reconstruct_image(
                            weight_grad, bias_grad, img_list
                        )
                        max_pearson_overall = max(max_pearson_overall, max_pearson)
                        max_psnr_overall = max(max_psnr_overall, max_psnr)
                        max_pearson_list.append(max_pearson)
                        max_psnr_list.append(max_psnr)
                        num_recovered_list.append(num_recovered)
                    else:
                        print("RECONSTRUCT: skipped because no local image batch was recorded this round")

            if len(max_pearson_list) > 0:
                N_Table = cfg['local_train_size']
                Max_Pearson_Table = max(max_pearson_list)
                Max_PSNR_Table = max(max_psnr_list)
                Max_Recovered_Table = max(num_recovered_list)
                fp.write("%s %s %s %s\n" % (
                    N_Table, Max_Pearson_Table, Max_PSNR_Table, Max_Recovered_Table
                ))


    if cfg['model_name'] == 'conv':
        targetWeights = ['blocks.2.weight']
        targetBiases = ['blocks.0.bias', 'blocks.2.bias']
        for m in range(num_active_users):
            for k, v in global_model_state_dict_copy.items():
                if federation.model_rate[user_idx[m]] == 0.25:
                    if v.dim() <= 1:
                        if (k in targetWeights or k in targetBiases):
                            model_history_block2[k].append(global_model_state_dict_copy[k])
                            if (epoch == 2): 
                                gradient_leakage(k, user_idx[m], num_active_users, federation.model_rate[user_idx[m]], local_parameters[m][k], model_history_block2[k])

    return

def reconstruct_image(weight_grad, bias_grad, img_list):
    count = 0
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0

    # Define rows and columns for plot. 
    rows = 2
    columns = cfg['local_train_size']
    # fig.add_subplot(rows, columns, 1)
    
    # (fig, axs) = plt.subplots(rows, columns)
    # for ax in axs.reshape(-1):
    #     ax.grid(False)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     # ax.set_zticks([])
    # plt.axis('off')
    # plt.grid(False)

    for elem in img_list:
        elem_extracted = elem[0].to(cfg["device"])
        max_pearson = 0.0
        max_ssim = 0.0
        max_psnr = 0.0

        pearson = PearsonCorrCoef().to(cfg["device"])
        count = count + 1
        best_partial_recon = None

        for i in range(weight_grad.size()[0]):
            partial_recon = torch.divide(weight_grad[i], bias_grad[i])

            img_reshaped = elem_extracted.reshape(partial_recon.size())

            # Compute Pearson Similarity: 
            pearson_coef = pearson(img_reshaped, partial_recon).item()

            partial_recon_as_img = partial_recon.reshape(elem_extracted.size())
            elem_extracted_numpy = np.squeeze(elem_extracted.cpu().numpy())
            partial_recon_numpy = np.squeeze(partial_recon_as_img.cpu().numpy())

            # Data range calculation based on normalization in data.py. 
            curr_ssim = ssim(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)
            # PSNR metric: 
            curr_psnr = psnr(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)

            max_ssim = max(max_ssim, curr_ssim)
            max_psnr = max(max_psnr, curr_psnr)
            max_pearson = max(max_pearson, pearson_coef)
            if max_pearson == pearson_coef:
                best_partial_recon = partial_recon

        if max_pearson >= 0.98:
            num_recovered = num_recovered + 1

        avg_ssim.append(max_ssim)
        avg_psnr.append(max_psnr)
        avg_pearson.append(max_pearson)

        partial_recon_as_img = best_partial_recon.reshape(elem_extracted.size())

        # axs[0, count].imshow(elem_extracted.cpu().numpy()[0], cmap='gray')
        # axs[1, count].imshow(partial_recon_as_img.cpu().numpy()[0], cmap='gray')  

        count = count + 1

        # plt.imshow(partial_recon_as_img.cpu().numpy()[0], cmap='gray')
        # fig_save_str = "New_Images/reconstructedConvRate_%s"%(count)
        # plt.savefig(fig_save_str)

    print("RECONSTRUCT: best_ssim across images = %s"%(max(avg_ssim)))
    print("RECONSTRUCT: best_psnr across images = %s"%(max(avg_psnr)))
    print("RECONSTRUCT: best_pearson across images = %s"%(max(avg_pearson)))
    print("RECONSTRUCT: num_recovered = %s"%(num_recovered))

    print("RECONSTRUCT: avg_ssim across images = %s"%(sum(avg_ssim) / len(avg_ssim)))
    print("RECONSTRUCT: avg_psnr across images = %s"%(sum(avg_psnr) / len(avg_psnr)))
    print("RECONSTRUCT: avg_pearson across images = %s"%(sum(avg_pearson) / len(avg_pearson)))

    # plt.show() 
    # plt.savefig('orig_recon_rolex_n%s_v2_noise.png'%(cfg['local_train_size']))

    return (max(avg_pearson), max(avg_psnr), num_recovered)



def fcnn_leakage(k, user, num_active_users, model_rate, local_params_after_train, model_history, distributed_model):
    """
    Reconstruct the target user's gradient using:
      - model_history[-2] = aggregated parent after the source round
      - model_history[-1] = aggregated parent after the replay round
      - distributed_model  = the actual local slice sent to the target at replay round
    """
    if len(model_history) < 2:
        return None

    hidden_layer_size = model_history[-2].size()[0]
    client_cap = int(model_rate * hidden_layer_size)

    lower = client_cap
    upper = lower + client_cap

    # Round t source aggregate contribution over the shifted block
    agg_val_prev = torch.multiply(model_history[-2][lower:upper], num_active_users - 1)

    # Round t+1 replay aggregate contribution over the shifted block
    agg_val_curr = torch.multiply(model_history[-1][lower:upper], num_active_users)

    agg_diff = torch.subtract(agg_val_curr, agg_val_prev)

    malicious_model_sent = distributed_model.to(cfg["device"])
    user_grad = torch.subtract(malicious_model_sent, agg_diff)
    return user_grad

def gradient_leakage(k, user, num_active_users, model_rate, local_params, global_params_list):

    # Take the gradient difference between rounds 1 and 2. 
    hidden_layer_size = global_params_list[0].size()[0]
    avg_val_rd_0 = global_params_list[0][int(hidden_layer_size * model_rate)]
    avg_val_rd_1 = global_params_list[1][int(hidden_layer_size * model_rate)]
    agg_val_rd_0 = avg_val_rd_0 * (num_active_users - 1) # A + B
    agg_val_rd_1 = avg_val_rd_1 * num_active_users # A + B + C
    agg_val_diff = torch.subtract(agg_val_rd_1, agg_val_rd_0)
    server_error = torch.abs(torch.subtract(agg_val_diff, local_params[-1])).item()
    # print("In gradient_leakage: server_error = %s"%(server_error))
    fp.write("%s %s\n"%(k, server_error))

def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


def cast_local_parameters_to_reference(local_parameters, expected_local_parameters):
    fixed = [OrderedDict() for _ in range(len(local_parameters))]
    for m in range(len(local_parameters)):
        for k in local_parameters[m]:
            if k in expected_local_parameters[m]:
                fixed[m][k] = local_parameters[m][k].to(
                    device=expected_local_parameters[m][k].device,
                    dtype=expected_local_parameters[m][k].dtype
                )
            else:
                fixed[m][k] = local_parameters[m][k]
    return fixed

def make_local(dataset, data_split, label_split, federation, round_log, logger):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()

    local_parameters, param_idx = federation.distribute(user_idx)
    expected_local_parameters = extract_expected_local_parameters(federation, user_idx, param_idx)
    local_parameters = cast_local_parameters_to_reference(local_parameters, expected_local_parameters)

    local = [None for _ in range(num_active_users)]

    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({
            'train': SplitDataset(dataset, data_split[user_idx[m]])
        })['train']

        local[m] = Local(
            user_idx[m],
            model_rate_m,
            data_loader_m,
            label_split[user_idx[m]],
            expected_local_parameters[m]
        )

        local[m].trusted_round, local[m].validation_reason = compare_local_parameters(
            local_parameters[m],
            expected_local_parameters[m]
        )

        if cfg.get('debug_extraction', False):
            received_hash = hash_local_parameter_dict(local_parameters[m])
            expected_hash = hash_local_parameter_dict(expected_local_parameters[m])

            logger.append({
                'info': [
                    f'[EXTRACT][Round {federation.rd}] user={int(user_idx[m])}',
                    f'[EXTRACT][Round {federation.rd}] rate={float(model_rate_m)}',
                    f'[EXTRACT][Round {federation.rd}] received_hash={received_hash}',
                    f'[EXTRACT][Round {federation.rd}] expected_hash={expected_hash}',
                    f'[EXTRACT][Round {federation.rd}] match={local[m].trusted_round}',
                    f'[EXTRACT][Round {federation.rd}] reason={local[m].validation_reason}',
                ]
            }, 'train', mean=False)

    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, user_id, model_rate, data_loader, label_split, expected_local_parameters):
        self.user_id = user_id
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.expected_local_parameters = expected_local_parameters
        self.trusted_round = True
        self.validation_reason = 'ok' 

    def train(self, local_parameters, lr, logger):
        if not self.trusted_round:
            logger.append({
                'info': [
                    f'Client {self.user_id} rejected malicious submodel before training',
                    f'Reason: {self.validation_reason}',
                    f'Validation response: {cfg["validation_response"]}',
                ]
            }, 'train', mean=False)

            if cfg['validation_response'] == 'strict_abort':
                raise RuntimeError(
                    f'Client {self.user_id} rejected round metadata/submodel: {self.validation_reason}'
                )

            safe_local_parameters = copy.deepcopy(self.expected_local_parameters)
            for k in safe_local_parameters:
                safe_local_parameters[k] = safe_local_parameters[k].to(cfg['device'])

            return safe_local_parameters, []
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        criterion = nn.CrossEntropyLoss()
        input_list = []

        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in list(enumerate(self.data_loader))[:cfg['local_train_size']]:
                input_list.append(input['img'])
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return (local_parameters, input_list)


if __name__ == "__main__":
    main()