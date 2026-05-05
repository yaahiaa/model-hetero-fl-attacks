import argparse
import csv
import copy
import datetime
import hashlib
import json
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed_rolex import Federation
from metrics import Metric
from utils import (
    save,
    to_device,
    process_control,
    process_dataset,
    make_optimizer,
    make_scheduler,
    resume,
    collate,
    makedir_exist_ok,
)
from logger import Logger
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
import round_log as round_log_module
from round_log import TransparencyLog, hash_state_dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value')


parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--global_epochs', default=None, type=int)
parser.add_argument('--local_epochs', default=None, type=int)
parser.add_argument('--local_train_size', default=None, type=int)
parser.add_argument('--train_batch_size', default=None, type=int)
parser.add_argument('--test_batch_size', default=None, type=int)
parser.add_argument('--experiment_method', default=None, type=str)
parser.add_argument('--experiment_id', default=None, type=str)
parser.add_argument('--results_dir', default=None, type=str)
parser.add_argument('--leakage_results_csv', default=None, type=str)
parser.add_argument('--epoch_results_csv', default=None, type=str)
parser.add_argument('--overhead_results_csv', default=None, type=str)
parser.add_argument('--enable_experiment_logging', default=None, type=str_to_bool)
parser.add_argument('--attack_noise_amount', default=None, type=float)
parser.add_argument('--noise_scale', default=None, type=float)
parser.add_argument('--attack_blocked_zero_metrics', default=None, type=str_to_bool)
parser.add_argument('--recovered_pearson_threshold', default=None, type=float)
parser.add_argument('--convergence_mode', default=None, type=str_to_bool)
parser.add_argument('--disable_attack_for_convergence', default=None, type=str_to_bool)
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
cfg['local_train_size'] = 10 if args['local_train_size'] is None else int(args['local_train_size'])
cfg['noise_scale'] = args['noise_scale']
cfg['distribute_init_val'] = 0.25
cfg['file_output'] = "New_Tables/MNIST_Rolex_TEST"
# -------------------------------------------------------------------------
# Prototype-2 defense configuration
# -------------------------------------------------------------------------
cfg.setdefault('validation_response', 'zero_change')
cfg.setdefault('round_log_dir', os.path.join('output', 'round_log', 'prototype2'))
cfg.setdefault('verifier_val_size', 32)
cfg.setdefault('verifier_max_loss_increase', 0.05)
cfg.setdefault('verifier_max_acc_drop', 10.0)
# Lower bound: reject near-exact replay
cfg.setdefault('verifier_min_relative_change', 1.0e-5)
# Upper bound: reject candidates that move too far from the previous approved parent
cfg.setdefault('verifier_max_relative_change', 0.6)
# Warmup: do not apply structural drift bounds during the first few rounds
cfg.setdefault('verifier_structure_warmup_rounds', 2)
cfg.setdefault('verifier_behavior_freeze_enabled', True)
# Minimum behavior movement expected when the model parameters changed.
cfg.setdefault('verifier_min_behavior_loss_delta', 0.0003)
cfg.setdefault('verifier_min_behavior_acc_delta', 1.0e-9)
cfg.setdefault('verifier_behavior_freeze_min_relative_change', 0.02)
# Only apply the freeze detector in the low/mid drift region.
cfg.setdefault('verifier_behavior_freeze_max_relative_change', 0.20)
cfg.setdefault('attack_model_mode', 'hybrid_trap')
cfg.setdefault('debug_hybrid_trap', True)
cfg.setdefault('experiment_method', 'committee')
cfg.setdefault('experiment_id', None)
cfg.setdefault('results_dir', 'results')
cfg.setdefault('leakage_results_csv', '{results_dir}/leakage_raw.csv')
cfg.setdefault('epoch_results_csv', '{results_dir}/epoch_raw.csv')
cfg.setdefault('overhead_results_csv', '{results_dir}/overhead_raw.csv')
cfg.setdefault('enable_experiment_logging', True)
cfg.setdefault('attack_noise_amount', None)
cfg.setdefault('attack_blocked_zero_metrics', True)
cfg.setdefault('recovered_pearson_threshold', 0.98)
cfg.setdefault('convergence_mode', False)
cfg.setdefault('disable_attack_for_convergence', False)
if args['seed'] is not None:
    cfg['init_seed'] = int(args['seed'])
    cfg['num_experiments'] = 1
if args['experiment_method'] is not None:
    cfg['experiment_method'] = args['experiment_method']
if args['experiment_id'] is not None:
    cfg['experiment_id'] = args['experiment_id']
if args['results_dir'] is not None:
    cfg['results_dir'] = args['results_dir']
if args['leakage_results_csv'] is not None:
    cfg['leakage_results_csv'] = args['leakage_results_csv']
if args['epoch_results_csv'] is not None:
    cfg['epoch_results_csv'] = args['epoch_results_csv']
if args['overhead_results_csv'] is not None:
    cfg['overhead_results_csv'] = args['overhead_results_csv']
if args['enable_experiment_logging'] is not None:
    cfg['enable_experiment_logging'] = args['enable_experiment_logging']
if args['attack_noise_amount'] is not None:
    cfg['attack_noise_amount'] = float(args['attack_noise_amount'])
if args['attack_blocked_zero_metrics'] is not None:
    cfg['attack_blocked_zero_metrics'] = args['attack_blocked_zero_metrics']
if args['recovered_pearson_threshold'] is not None:
    cfg['recovered_pearson_threshold'] = float(args['recovered_pearson_threshold'])
if args['convergence_mode'] is not None:
    cfg['convergence_mode'] = args['convergence_mode']
if args['disable_attack_for_convergence'] is not None:
    cfg['disable_attack_for_convergence'] = args['disable_attack_for_convergence']


def safe_cfg_get(*keys, default=''):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def ensure_dir(path):
    if path:
        makedir_exist_ok(path)


def resolve_results_path(path_value):
    if path_value is None:
        return None
    resolved = str(path_value).format(results_dir=cfg['results_dir'])
    directory = os.path.dirname(resolved)
    if directory:
        ensure_dir(directory)
    return resolved


def now_seconds():
    return time.perf_counter()


def infer_experiment_method():
    explicit = safe_cfg_get('experiment_method', default='unknown')
    if explicit and explicit != 'unknown':
        return str(explicit).lower()
    if any(k in cfg for k in ['verifier_val_size', 'verifier_max_loss_increase', 'commit_rejection_response']):
        return 'committee'
    if any(k in cfg for k in ['ldp_enabled', 'local_dp', 'apply_ldp']):
        return 'ldp'
    if any(k in cfg for k in ['ddp_enabled', 'distributed_dp', 'apply_ddp']):
        return 'ddp'
    return 'base'


def attack_execution_enabled():
    return bool(safe_cfg_get('attack_commit_enabled', default=False)) and not (
        bool(safe_cfg_get('convergence_mode', default=False))
        and bool(safe_cfg_get('disable_attack_for_convergence', default=False))
    )


def detect_dp_mode():
    method = str(cfg.get('experiment_method', 'unknown')).lower()
    if method in {'ldp', 'ddp'}:
        return method
    return 'none'


def resolve_experiment_defaults():
    cfg['experiment_method'] = infer_experiment_method()
    if cfg.get('experiment_id') in (None, '', 'default'):
        cfg['experiment_id'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.get('attack_noise_amount') is None:
        cfg['attack_noise_amount'] = float(
            safe_cfg_get('attack_commit_noise_scale', 'noise_scale', default=0.0) or 0.0
        )
    else:
        cfg['attack_commit_noise_scale'] = float(cfg['attack_noise_amount'])
    if cfg.get('noise_scale') is None:
        cfg['noise_scale'] = safe_cfg_get('attack_commit_noise_scale', default=0.0)
    cfg['leakage_results_csv'] = resolve_results_path(cfg['leakage_results_csv'])
    cfg['epoch_results_csv'] = resolve_results_path(cfg['epoch_results_csv'])
    cfg['overhead_results_csv'] = resolve_results_path(cfg['overhead_results_csv'])


def append_csv_row(path, fieldnames, row):
    if not cfg.get('enable_experiment_logging', True):
        return
    directory = os.path.dirname(path)
    if directory:
        ensure_dir(directory)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})


def empty_reconstruction_metrics():
    return {
        'best_pearson': 0.0,
        'avg_pearson': 0.0,
        'best_psnr': 0.0,
        'avg_psnr': 0.0,
        'num_recovered': 0,
    }


def merge_reconstruction_metrics(current_metrics, candidate_metrics):
    merged = empty_reconstruction_metrics()
    for key in merged:
        merged[key] = max(current_metrics.get(key, 0.0), candidate_metrics.get(key, 0.0))
    return merged


def summarize_committee_status(event):
    if event is None:
        return {
            'committee_enabled': cfg.get('experiment_method') == 'committee',
            'committee_approved': '',
            'candidate_rejected': False,
            'attack_blocked': False,
        }
    approved = bool(event.get('approved', False))
    return {
        'committee_enabled': cfg.get('experiment_method') == 'committee',
        'committee_approved': approved,
        'candidate_rejected': not approved,
        'attack_blocked': not approved,
    }


def write_epoch_result(row):
    fieldnames = [
        'experiment_id', 'experiment_method', 'seed', 'epoch', 'dataset', 'model_name',
        'control_name', 'global_accuracy', 'global_loss', 'local_accuracy_mean',
        'local_loss_mean', 'epoch_time_sec', 'train_time_sec', 'aggregation_time_sec',
        'committee_time_sec', 'dp_time_sec', 'test_time_sec', 'attack_enabled',
        'attack_round', 'committee_enabled', 'committee_approved_this_epoch',
        'candidate_rejected_this_epoch',
    ]
    append_csv_row(cfg['epoch_results_csv'], fieldnames, row)


def write_leakage_result(row):
    fieldnames = [
        'experiment_id', 'experiment_method', 'seed', 'dataset', 'model_name',
        'control_name', 'global_epochs', 'local_epochs', 'local_train_size',
        'batch_size_train', 'attack_source_round', 'attack_replay_round',
        'attack_noise_amount', 'noise_scale', 'dp_mode', 'dp_clip_norm',
        'dp_noise_multiplier', 'committee_enabled', 'committee_approved',
        'attack_blocked', 'commit_rejection_response', 'best_pearson',
        'avg_pearson', 'best_psnr', 'avg_psnr', 'num_recovered',
        'total_runtime_sec', 'final_global_accuracy', 'final_global_loss',
    ]
    append_csv_row(cfg['leakage_results_csv'], fieldnames, row)


def write_overhead_summary(row):
    fieldnames = [
        'experiment_id', 'experiment_method', 'seed', 'dataset', 'model_name',
        'control_name', 'num_epochs', 'total_runtime_sec', 'mean_epoch_time_sec',
        'mean_train_time_sec', 'mean_aggregation_time_sec', 'mean_committee_time_sec',
        'mean_dp_time_sec', 'mean_test_time_sec', 'relative_notes',
    ]
    append_csv_row(cfg['overhead_results_csv'], fieldnames, row)


def build_common_result_fields(seed):
    return {
        'experiment_id': cfg['experiment_id'],
        'experiment_method': cfg['experiment_method'],
        'seed': seed,
        'dataset': cfg['data_name'],
        'model_name': cfg['model_name'],
        'control_name': cfg['control_name'],
    }


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def apply_runtime_overrides():
    if args['global_epochs'] is not None:
        cfg['num_epochs']['global'] = int(args['global_epochs'])
    if args['local_epochs'] is not None:
        cfg['num_epochs']['local'] = int(args['local_epochs'])
    if args['train_batch_size'] is not None:
        cfg['batch_size']['train'] = int(args['train_batch_size'])
    if args['test_batch_size'] is not None:
        cfg['batch_size']['test'] = int(args['test_batch_size'])


full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Avg_Pearson Max_PSNR Avg_PSNR Max_Recovered\n")

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

def behavior_freeze_check(prev_eval, cand_eval, rel_change):
    loss_delta = abs(float(cand_eval['Local-Loss']) - float(prev_eval['Local-Loss']))
    acc_delta = abs(float(cand_eval['Local-Accuracy']) - float(prev_eval['Local-Accuracy']))

    min_loss_delta = float(cfg.get('verifier_min_behavior_loss_delta', 0.005))
    min_acc_delta = float(cfg.get('verifier_min_behavior_acc_delta', 1.0e-9))
    max_rel_for_freeze = float(cfg.get('verifier_behavior_freeze_max_relative_change', 0.20))
    min_rel_for_freeze = float(cfg.get('verifier_behavior_freeze_min_relative_change', 0.04))

    enabled = bool(cfg.get('verifier_behavior_freeze_enabled', True))

    is_frozen = (
        enabled
        and float(rel_change) <= max_rel_for_freeze
        and float(rel_change) >= min_rel_for_freeze
        and loss_delta < min_loss_delta
        and acc_delta <= min_acc_delta
    )

    details = {
        'behavior_freeze_enabled': enabled,
        'behavior_loss_delta': loss_delta,
        'behavior_acc_delta': acc_delta,
        'verifier_min_behavior_loss_delta': min_loss_delta,
        'verifier_min_behavior_acc_delta': min_acc_delta,
        'verifier_behavior_freeze_max_relative_change': max_rel_for_freeze,
        'behavior_frozen': is_frozen,
    }

    return is_frozen, details

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

def get_fcnn_shifted_block_contributor_counts(federation, user_idx, target_rate=0.25):
    active_rates = [float(federation.model_rate[u]) for u in user_idx]
    num_source_contributors = sum(1 for r in active_rates if r > target_rate)
    num_replay_contributors = sum(1 for r in active_rates if r >= target_rate)
    return num_source_contributors, num_replay_contributors

def build_candidate_parent_for_commitment(epoch, federation, honest_aggregated_state, logger):
    """
    Build the parent model the server will actually commit to the committee.

    Honest path:
      candidate = honest aggregated parent

    Malicious replay path on attack_source_round:
      candidate = replay of the parent used in this round (federation.initial_parent_state),
      optionally with additive noise
    """
    attack_enabled = attack_execution_enabled()
    attack_source_round = int(cfg.get('attack_source_round', 3))
    attack_replay_round = int(cfg.get('attack_replay_round', 4))
    commit_noise_scale = float(cfg.get('attack_commit_noise_scale', 0.0))

    previous_parent_state = clone_state_dict(federation.initial_parent_state)
    honest_parent_state = clone_state_dict(honest_aggregated_state)

    previous_hash = round_log_module.hash_state_dict(previous_parent_state)
    honest_hash = round_log_module.hash_state_dict(honest_parent_state)

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

    candidate_hash = round_log_module.hash_state_dict(candidate_state)

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
    round_log_module,
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
    previous_parent_record = round_log_module.get_latest_approved_parent()
    previous_parent_state = round_log_module.load_parent_state_dict(previous_parent_record)

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

        # Allow larger honest movement during the first few rounds
        structure_checks_enabled = int(epoch) > int(cfg.get('verifier_structure_warmup_rounds', 2))

        behavior_frozen, behavior_report = behavior_freeze_check(
            prev_eval=prev_eval,
            cand_eval=cand_eval,
            rel_change=rel_change,
        )

        if cand_eval['Local-Loss'] > prev_eval['Local-Loss'] + cfg['verifier_max_loss_increase']:
            approved = False
            reason = 'validation loss increased too much'

        elif cand_eval['Local-Accuracy'] + cfg['verifier_max_acc_drop'] < prev_eval['Local-Accuracy']:
            approved = False
            reason = 'validation accuracy dropped too much'

        elif structure_checks_enabled and rel_change < cfg['verifier_min_relative_change']:
            approved = False
            reason = 'candidate too similar to previous approved parent'

        elif structure_checks_enabled and rel_change > cfg['verifier_max_relative_change']:
            approved = False
            reason = 'candidate too different from previous approved parent'

        elif structure_checks_enabled and behavior_frozen:
            approved = False
            reason = 'candidate has parameter drift but frozen verifier behavior'
        report = {
            'user_id': int(verifier_user_id),
            'cohort_rate': cohort_rate,
            'approved': approved,
            'reason': reason,
            'prev_eval': prev_eval,
            'cand_eval': cand_eval,
            'relative_change': rel_change,
            'structure_checks_enabled': structure_checks_enabled,
            'verifier_min_relative_change': float(cfg['verifier_min_relative_change']),
            'verifier_max_relative_change': float(cfg['verifier_max_relative_change']),
            **behavior_report,
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
                    f'[COMMIT][Verifier {verifier_user_id}] structure_checks_enabled={structure_checks_enabled}',
                    f'[COMMIT][Verifier {verifier_user_id}] min_rel_change={cfg["verifier_min_relative_change"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] max_rel_change={cfg["verifier_max_relative_change"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_loss_delta={behavior_report["behavior_loss_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_acc_delta={behavior_report["behavior_acc_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_frozen={behavior_report["behavior_frozen"]}',
                    f'[COMMIT][Verifier {verifier_user_id}] min_behavior_loss_delta={behavior_report["verifier_min_behavior_loss_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] min_behavior_acc_delta={behavior_report["verifier_min_behavior_acc_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_freeze_max_rel={behavior_report["verifier_behavior_freeze_max_relative_change"]:.6e}',
                ]
            }, 'train', mean=False) 

    active_user_model_rates = {
        int(uid): float(federation.model_rate[uid]) for uid in user_idx
    }

    event = round_log_module.record_candidate_parent(
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
    apply_runtime_overrides()
    resolve_experiment_defaults()
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
    experiment_tracker = {
        'seed': seed,
        'epoch_rows': [],
        'last_reconstruction': empty_reconstruction_metrics(),
        'committee_approved': '',
        'attack_blocked': False,
        'final_global_accuracy': 0.0,
        'final_global_loss': 0.0,
    }
    total_runtime_start = now_seconds()
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

    round_log_module = TransparencyLog(cfg['round_log_dir'])
    round_log_module.bootstrap_initial_parent(global_parameters, parent_for_round=1)

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
    fcnn_attack_cache = {
    'source_honest_parent': {},
    'replay_result_parent': {},
}
    runtime_control = {
    'skip_next_epoch': False,
    'skip_reason': None,
}


    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        epoch_start = now_seconds()
        logger.safe(True)

        approved_parent_record = round_log_module.get_latest_approved_parent()
        approved_parent_state = round_log_module.load_parent_state_dict(approved_parent_record)
        approved_parent_state = move_state_dict_to_device(approved_parent_state, cfg['device'])
        global_parameters = copy.deepcopy(approved_parent_state)

        federation = Federation(epoch, global_parameters, cfg['model_rate'], label_split)
        train_context = train(
            model_history_block2,
            model_history_fcnn,
            fcnn_attack_cache,
            dataset['train'],
            data_split['train'],
            label_split,
            federation,
            model,
            optimizer,
            logger,
            epoch,
            round_log_module,
            runtime_control,
            experiment_tracker
        )
        test_start = now_seconds()
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        test_time_sec = now_seconds() - test_start
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        epoch_time_sec = now_seconds() - epoch_start
        final_global_accuracy = float(logger.mean.get('test/Global-Accuracy', 0.0))
        final_global_loss = float(logger.mean.get('test/Global-Loss', 0.0))
        experiment_tracker['final_global_accuracy'] = final_global_accuracy
        experiment_tracker['final_global_loss'] = final_global_loss
        epoch_row = {
            **build_common_result_fields(seed),
            'epoch': int(epoch),
            'global_accuracy': final_global_accuracy,
            'global_loss': final_global_loss,
            'local_accuracy_mean': float(logger.mean.get('train/Local-Accuracy', 0.0)),
            'local_loss_mean': float(logger.mean.get('train/Local-Loss', 0.0)),
            'epoch_time_sec': epoch_time_sec,
            'train_time_sec': float(train_context.get('train_time_sec', 0.0)),
            'aggregation_time_sec': float(train_context.get('aggregation_time_sec', 0.0)),
            'committee_time_sec': float(train_context.get('committee_time_sec', 0.0)),
            'dp_time_sec': float(train_context.get('dp_time_sec', 0.0)),
            'test_time_sec': test_time_sec,
            'attack_enabled': bool(train_context.get('attack_enabled', False)),
            'attack_round': bool(train_context.get('attack_round', False)),
            'committee_enabled': bool(train_context.get('committee_enabled', False)),
            'committee_approved_this_epoch': train_context.get('committee_approved_this_epoch', ''),
            'candidate_rejected_this_epoch': bool(train_context.get('candidate_rejected_this_epoch', False)),
        }
        experiment_tracker['epoch_rows'].append(epoch_row)
        write_epoch_result(epoch_row)
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
    total_runtime_sec = now_seconds() - total_runtime_start
    epoch_rows = experiment_tracker['epoch_rows']
    train_times = [float(row['train_time_sec']) for row in epoch_rows]
    aggregation_times = [float(row['aggregation_time_sec']) for row in epoch_rows]
    committee_times = [float(row['committee_time_sec']) for row in epoch_rows]
    dp_times = [float(row['dp_time_sec']) for row in epoch_rows]
    test_times = [float(row['test_time_sec']) for row in epoch_rows]
    epoch_times = [float(row['epoch_time_sec']) for row in epoch_rows]
    reconstruction_metrics = experiment_tracker.get('last_reconstruction', empty_reconstruction_metrics())
    leakage_row = {
        **build_common_result_fields(seed),
        'global_epochs': cfg['num_epochs']['global'],
        'local_epochs': cfg['num_epochs']['local'],
        'local_train_size': cfg['local_train_size'],
        'batch_size_train': safe_cfg_get('batch_size', default={}).get('train', ''),
        'attack_source_round': safe_cfg_get('attack_source_round', default=''),
        'attack_replay_round': safe_cfg_get('attack_replay_round', default=''),
        'attack_noise_amount': cfg['attack_noise_amount'],
        'noise_scale': safe_cfg_get('noise_scale', default=''),
        'dp_mode': detect_dp_mode(),
        'dp_clip_norm': safe_cfg_get('clip_norm', 'dp_clip_norm', default=''),
        'dp_noise_multiplier': safe_cfg_get('noise_multiplier', 'dp_noise_multiplier', default=''),
        'committee_enabled': cfg['experiment_method'] == 'committee',
        'committee_approved': experiment_tracker.get('committee_approved', ''),
        'attack_blocked': experiment_tracker.get('attack_blocked', False),
        'commit_rejection_response': safe_cfg_get('commit_rejection_response', default=''),
        'best_pearson': reconstruction_metrics['best_pearson'],
        'avg_pearson': reconstruction_metrics['avg_pearson'],
        'best_psnr': reconstruction_metrics['best_psnr'],
        'avg_psnr': reconstruction_metrics['avg_psnr'],
        'num_recovered': reconstruction_metrics['num_recovered'],
        'total_runtime_sec': total_runtime_sec,
        'final_global_accuracy': experiment_tracker['final_global_accuracy'],
        'final_global_loss': experiment_tracker['final_global_loss'],
    }
    write_leakage_result(leakage_row)
    write_overhead_summary({
        **build_common_result_fields(seed),
        'num_epochs': cfg['num_epochs']['global'],
        'total_runtime_sec': total_runtime_sec,
        'mean_epoch_time_sec': mean_or_zero(epoch_times),
        'mean_train_time_sec': mean_or_zero(train_times),
        'mean_aggregation_time_sec': mean_or_zero(aggregation_times),
        'mean_committee_time_sec': mean_or_zero(committee_times),
        'mean_dp_time_sec': mean_or_zero(dp_times),
        'mean_test_time_sec': mean_or_zero(test_times),
        'relative_notes': 'convergence_mode={} disable_attack_for_convergence={}'.format(
            cfg['convergence_mode'], cfg['disable_attack_for_convergence']
        ),
    })
    logger.safe(False)
    return


def train(model_history_block2, model_history_fcnn,fcnn_attack_cache, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, transparency_log, runtime_control, experiment_tracker):
    global_model.load_state_dict(federation.global_parameters)
    attack_enabled = attack_execution_enabled()
    attack_round = attack_enabled and int(epoch) in {
        int(safe_cfg_get('attack_source_round', default=-1)),
        int(safe_cfg_get('attack_replay_round', default=-1)),
    }
    committee_enabled = cfg.get('experiment_method') == 'committee'
    committee_status = {
        'committee_enabled': committee_enabled,
        'committee_approved': '',
        'candidate_rejected': False,
        'attack_blocked': False,
    }
    train_time_sec = 0.0
    aggregation_time_sec = 0.0
    committee_time_sec = 0.0
    dp_time_sec = 0.0

    if runtime_control.get('skip_next_epoch', False):
        runtime_control['skip_next_epoch'] = False
        skip_reason = runtime_control.get('skip_reason', 'rejected_previous_round')
        runtime_control['skip_reason'] = None

        logger.append({
            'info': [
                f'[EPOCH-NOOP] epoch={epoch}',
                f'[EPOCH-NOOP] reason={skip_reason}',
                '[EPOCH-NOOP] local training, aggregation, commitment, reconstruction, and test evaluation were skipped',
            ]
        }, 'train', mean=False)

        print(
            f"[EPOCH-NOOP] epoch={epoch} reason={skip_reason} "
            f"local training/aggregation/commitment/reconstruction/test skipped",
            flush=True
        )

        return {
            'train_time_sec': train_time_sec,
            'aggregation_time_sec': aggregation_time_sec,
            'committee_time_sec': committee_time_sec,
            'dp_time_sec': dp_time_sec,
            'attack_enabled': attack_enabled,
            'attack_round': attack_round,
            'committee_enabled': committee_enabled,
            'committee_approved_this_epoch': '',
            'candidate_rejected_this_epoch': False,
        }

    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, transparency_log, logger)
    distributed_local_parameters = copy.deepcopy(local_parameters)
    num_active_users = len(local)

    start_time = time.time()
    train_timer_start = now_seconds()
    img_list_by_user = {}

    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]]
        (local_parameters[m], img_data) = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))

        if img_data:
            img_list_by_user[int(user_idx[m])] = copy.deepcopy(img_data)
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
    train_time_sec = now_seconds() - train_timer_start

    aggregation_timer_start = now_seconds()
    federation.combine(local_parameters, param_idx, user_idx)
    aggregation_time_sec = now_seconds() - aggregation_timer_start

    honest_aggregated_state = clone_state_dict(federation.global_parameters)
    if cfg['model_name'] == 'fcnn' and attack_enabled:
        attack_source_round = int(cfg.get('attack_source_round', 3))
        if epoch == attack_source_round:
            for key in ['layers.0.weight', 'layers.0.bias']:
                fcnn_attack_cache['source_honest_parent'][key] = clone_state_dict(honest_aggregated_state)[key]

    candidate_parent_state, candidate_meta = build_candidate_parent_for_commitment(
        epoch,
        federation,
        honest_aggregated_state,
        logger,
    )

    committee_timer_start = now_seconds()
    commitment_event = verify_and_commit_candidate_parent(
        transparency_log,
        epoch,
        copy.deepcopy(candidate_parent_state),
        local,
        user_idx,
        federation,
        label_split,
        logger,
        candidate_meta=candidate_meta,
    )
    committee_time_sec += now_seconds() - committee_timer_start
    committee_status = summarize_committee_status(commitment_event)
    experiment_tracker['committee_approved'] = committee_status['committee_approved']
    experiment_tracker['attack_blocked'] = bool(
        experiment_tracker.get('attack_blocked', False) or committee_status['attack_blocked']
    )

    final_parent_state = None
    round_was_skipped = False
    round_skip_reason = None

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

        if cfg.get('debug_hybrid_trap', False) and cfg.get('attack_model_mode', 'real_replay') == 'hybrid_trap':
            logger.append({
                'info': [
                    f'[HYBRID_TRAP][COMMIT_REJECTED] epoch={epoch}',
                    f'[HYBRID_TRAP][COMMIT_REJECTED] commitment_id={commitment_event["commitment_id"]}',
                    f'[HYBRID_TRAP][COMMIT_REJECTED] rejection_response={rejection_response}',
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
                'previous_parent_hash': transparency_log.hash_state_dict(federation.initial_parent_state),
                'honest_aggregated_hash': transparency_log.hash_state_dict(honest_aggregated_state),
                'candidate_hash': transparency_log.hash_state_dict(honest_aggregated_state),
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

            fallback_committee_timer_start = now_seconds()
            fallback_event = verify_and_commit_candidate_parent(
                transparency_log,
                epoch,
                copy.deepcopy(honest_aggregated_state),
                local,
                user_idx,
                federation,
                label_split,
                logger,
                candidate_meta=fallback_meta,
            )
            committee_time_sec += now_seconds() - fallback_committee_timer_start

            if fallback_event['approved']:
                final_parent_state = clone_state_dict(honest_aggregated_state)
            else:
                approved_parent_record = transparency_log.get_latest_approved_parent()
                rollback_state = transparency_log.load_parent_state_dict(approved_parent_record)
                rollback_state = move_state_dict_to_device(rollback_state, cfg['device'])
                final_parent_state = clone_state_dict(rollback_state)
                round_was_skipped = True
                round_skip_reason = 'fallback_honest_rejected'

        elif rejection_response == 'rollback_previous':
            approved_parent_record = transparency_log.get_latest_approved_parent()
            rollback_state = transparency_log.load_parent_state_dict(approved_parent_record)
            rollback_state = move_state_dict_to_device(rollback_state, cfg['device'])

            runtime_control['skip_next_epoch'] = True
            runtime_control['skip_reason'] = 'candidate_rejected_round_voided'
            final_parent_state = clone_state_dict(rollback_state)

        else:
            raise ValueError(f"Unknown commit_rejection_response: {rejection_response}")

    federation.global_parameters = clone_state_dict(final_parent_state)
    global_model.load_state_dict(final_parent_state)
    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())
    if round_was_skipped:
        next_parent_hash = hash_state_dict(final_parent_state)

        logger.append({
            'info': [
                f'[ROUND-SKIP] epoch={epoch}',
                f'[ROUND-SKIP] reason={round_skip_reason}',
                f'[ROUND-SKIP] next_parent_hash={next_parent_hash}',
                '[ROUND-SKIP] rejected round update was discarded; training will continue next epoch from last approved parent',
            ]
        }, 'train', mean=False)

        print(
            f"[ROUND-SKIP] epoch={epoch} reason={round_skip_reason} "
            f"next_parent_hash={next_parent_hash} "
            f"training continues from last approved parent",
            flush=True
        )

    if cfg['model_name'] == 'fcnn' and attack_enabled:
        attack_replay_round = int(cfg.get('attack_replay_round', 4))

        if round_was_skipped:
            fcnn_attack_cache['replay_result_parent'].clear()
        elif epoch == attack_replay_round:
            for key in ['layers.0.weight', 'layers.0.bias']:
                fcnn_attack_cache['replay_result_parent'][key] = global_model_state_dict_copy[key].detach().clone()

    if cfg['model_name'] == 'fcnn':
        targetWeights = ['layers.0.weight']
        targetBiases = ['layers.0.bias']
        round_reconstruction_metrics = empty_reconstruction_metrics()
        attack_replay_round = int(cfg.get('attack_replay_round', 4))

        for m in range(num_active_users):
            weight_grad = None
            bias_grad = None
            user_reconstruction_metrics = empty_reconstruction_metrics()

            if federation.model_rate[user_idx[m]] != 0.25:
                continue

            for k, v in global_model_state_dict_copy.items():
                if k in targetWeights or k in targetBiases:
                    if attack_enabled and epoch == attack_replay_round and not round_was_skipped and not runtime_control.get('skip_next_epoch', False):
                        distributed_model = distributed_local_parameters[m][k]

                        num_source_contributors, num_replay_contributors = get_fcnn_shifted_block_contributor_counts(
                            federation, user_idx, target_rate=0.25
                        )

                        source_parent_tensor = fcnn_attack_cache['source_honest_parent'].get(k, None)
                        replay_parent_tensor = fcnn_attack_cache['replay_result_parent'].get(k, None)

                        if cfg.get('debug_attack_replay', False):
                            print(
                                f"[REPLAY][LEAKAGE] epoch={epoch} user={user_idx[m]} key={k} "
                                f"rate={federation.model_rate[user_idx[m]]} "
                                f"dist_shape={tuple(distributed_model.shape)} "
                                f"src_count={num_source_contributors} "
                                f"replay_count={num_replay_contributors} "
                                f"have_source={source_parent_tensor is not None} "
                                f"have_replay={replay_parent_tensor is not None}",
                                flush=True
                            )

                        recovered = fcnn_leakage_from_cache(
                            key=k,
                            model_rate=federation.model_rate[user_idx[m]],
                            distributed_model=distributed_model,
                            source_honest_parent=source_parent_tensor,
                            replay_result_parent=replay_parent_tensor,
                            num_source_contributors=num_source_contributors,
                            num_replay_contributors=num_replay_contributors,
                        )

                        if k in targetWeights:
                            weight_grad = recovered
                        else:
                            bias_grad = recovered

            if weight_grad is not None and bias_grad is not None:
                if cfg.get('debug_hybrid_trap', False):
                    bias_grad_abs = torch.abs(bias_grad.detach().float())
                    bias_grad_abs_min = float(torch.min(bias_grad_abs).item()) if bias_grad_abs.numel() > 0 else 0.0
                    bias_grad_abs_max = float(torch.max(bias_grad_abs).item()) if bias_grad_abs.numel() > 0 else 0.0
                    bias_grad_nonzero = int(torch.count_nonzero(bias_grad.detach()).item())
                    weight_grad_norm = float(torch.norm(weight_grad.detach().float(), p=2).item())
                    bias_grad_norm = float(torch.norm(bias_grad.detach().float(), p=2).item())

                    logger.append({
                        'info': [
                            f'[HYBRID_TRAP][RECON] epoch={epoch}',
                            f'[HYBRID_TRAP][RECON] user_id={int(user_idx[m])}',
                            f'[HYBRID_TRAP][RECON] bias_grad_abs_min={bias_grad_abs_min:.6e}',
                            f'[HYBRID_TRAP][RECON] bias_grad_abs_max={bias_grad_abs_max:.6e}',
                            f'[HYBRID_TRAP][RECON] bias_grad_nonzero_count={bias_grad_nonzero}',
                            f'[HYBRID_TRAP][RECON] weight_grad_norm={weight_grad_norm:.6e}',
                            f'[HYBRID_TRAP][RECON] bias_grad_norm={bias_grad_norm:.6e}',
                        ]
                    }, 'train', mean=False)

                bias_grad_sum = torch.abs(torch.sum(bias_grad)).item()

                if bias_grad_sum != 0.0:
                    target_img_list = img_list_by_user.get(int(user_idx[m]), None)
                    if target_img_list is not None and len(target_img_list) > 0:
                        user_reconstruction_metrics = reconstruct_image(weight_grad, bias_grad, target_img_list)
                    else:
                        print(
                            f"RECONSTRUCT: skipped because no image batch was recorded for target user {user_idx[m]}",
                            flush=True
                        )

            round_reconstruction_metrics['best_pearson'] = max(
                round_reconstruction_metrics['best_pearson'],
                user_reconstruction_metrics['best_pearson']
            )
            round_reconstruction_metrics['avg_pearson'] = max(
                round_reconstruction_metrics['avg_pearson'],
                user_reconstruction_metrics['avg_pearson']
            )
            round_reconstruction_metrics['best_psnr'] = max(
                round_reconstruction_metrics['best_psnr'],
                user_reconstruction_metrics['best_psnr']
            )
            round_reconstruction_metrics['avg_psnr'] = max(
                round_reconstruction_metrics['avg_psnr'],
                user_reconstruction_metrics['avg_psnr']
            )
            round_reconstruction_metrics['num_recovered'] = max(
                round_reconstruction_metrics['num_recovered'],
                user_reconstruction_metrics['num_recovered']
            )

        if not attack_enabled and cfg.get('attack_blocked_zero_metrics', True):
            round_reconstruction_metrics = empty_reconstruction_metrics()

        fp.write("%s %s %s %s %s %s\n" % (
            cfg['local_train_size'],
            round_reconstruction_metrics['best_pearson'],
            round_reconstruction_metrics['avg_pearson'],
            round_reconstruction_metrics['best_psnr'],
            round_reconstruction_metrics['avg_psnr'],
            round_reconstruction_metrics['num_recovered'],
        ))
        experiment_tracker['last_reconstruction'] = merge_reconstruction_metrics(
            experiment_tracker.get('last_reconstruction', empty_reconstruction_metrics()),
            round_reconstruction_metrics,
        )

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

    return {
        'train_time_sec': train_time_sec,
        'aggregation_time_sec': aggregation_time_sec,
        'committee_time_sec': committee_time_sec,
        'dp_time_sec': dp_time_sec,
        'attack_enabled': attack_enabled,
        'attack_round': attack_round,
        'committee_enabled': committee_enabled,
        'committee_approved_this_epoch': committee_status['committee_approved'],
        'candidate_rejected_this_epoch': committee_status['candidate_rejected'],
    }

def reconstruct_image(weight_grad, bias_grad, img_list):
    count = 0
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0
    threshold = float(safe_cfg_get('recovered_pearson_threshold', default=0.98))

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
            denom = bias_grad[i]
            eps = 1.0e-8

            if torch.abs(denom).item() < eps:
                continue

            partial_recon = weight_grad[i] / denom

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

        if max_pearson >= threshold:
            num_recovered = num_recovered + 1

        avg_ssim.append(max_ssim)
        avg_psnr.append(max_psnr)
        avg_pearson.append(max_pearson)
        if best_partial_recon is None:
            continue
        partial_recon_as_img = best_partial_recon.reshape(elem_extracted.size())

        # axs[0, count].imshow(elem_extracted.cpu().numpy()[0], cmap='gray')
        # axs[1, count].imshow(partial_recon_as_img.cpu().numpy()[0], cmap='gray')  

        count = count + 1

        # plt.imshow(partial_recon_as_img.cpu().numpy()[0], cmap='gray')
        # fig_save_str = "New_Images/reconstructedConvRate_%s"%(count)
        # plt.savefig(fig_save_str)
    if len(avg_pearson) == 0:
        print("RECONSTRUCT: no valid partial reconstructions", flush=True)
        return empty_reconstruction_metrics()
    best_ssim = max(avg_ssim)
    best_psnr = max(avg_psnr)
    best_pearson = max(avg_pearson)
    avg_psnr_value = sum(avg_psnr) / len(avg_psnr)
    avg_pearson_value = sum(avg_pearson) / len(avg_pearson)
    print("RECONSTRUCT: best_ssim across images = %s"%(best_ssim))
    print("RECONSTRUCT: best_psnr across images = %s"%(best_psnr))
    print("RECONSTRUCT: best_pearson across images = %s"%(best_pearson))
    print("RECONSTRUCT: num_recovered = %s"%(num_recovered))

    print("RECONSTRUCT: avg_ssim across images = %s"%(sum(avg_ssim) / len(avg_ssim)))
    print("RECONSTRUCT: avg_psnr across images = %s"%(avg_psnr_value))
    print("RECONSTRUCT: avg_pearson across images = %s"%(avg_pearson_value))

    # plt.show() 
    # plt.savefig('orig_recon_rolex_n%s_v2_noise.png'%(cfg['local_train_size']))

    return {
        'best_pearson': best_pearson,
        'avg_pearson': avg_pearson_value,
        'best_psnr': best_psnr,
        'avg_psnr': avg_psnr_value,
        'num_recovered': num_recovered,
    }



def fcnn_leakage_from_cache(
    *,
    key,
    model_rate,
    distributed_model,
    source_honest_parent,
    replay_result_parent,
    num_source_contributors,
    num_replay_contributors,
):
    """
    Recover the target cohort's plaintext update on the shifted block.

    source_honest_parent: honest aggregate after the source round
    replay_result_parent: resulting parent after the replay round
    distributed_model: actual local slice sent to the target client at replay round
    """

    if source_honest_parent is None or replay_result_parent is None:
        return None

    hidden_layer_size = source_honest_parent.size(0)
    client_cap = int(model_rate * hidden_layer_size)

    lower = client_cap
    upper = lower + client_cap

    # Aggregate over the shifted block:
    # source round: only larger cohorts contribute
    # replay round: larger cohorts + target cohort contribute
    source_block = source_honest_parent[lower:upper]
    replay_block = replay_result_parent[lower:upper]

    agg_val_prev = source_block * float(num_source_contributors)
    agg_val_curr = replay_block * float(num_replay_contributors)

    agg_diff = agg_val_curr - agg_val_prev
    malicious_model_sent = distributed_model.to(cfg["device"])

    user_grad = malicious_model_sent - agg_diff
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

def make_local(dataset, data_split, label_split, federation, round_log_module, logger):
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
