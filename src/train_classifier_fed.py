import argparse
import csv
import copy
import datetime
import hashlib
import json
import math
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from torchmetrics.regression import PearsonCorrCoef
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation, extract_honest_static_submodel_from_parent
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
import round_log as round_log_module
from commitment_architecture import CommitmentService, JsonCommitmentLedger, LocalArtifactStore, VerificationReport
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
parser.add_argument('--attack_blocked_zero_metrics', default=None, type=str_to_bool)
parser.add_argument('--recovered_pearson_threshold', default=None, type=float)
parser.add_argument('--convergence_mode', default=None, type=str_to_bool)
parser.add_argument('--disable_attack_for_convergence', default=None, type=str_to_bool)
parser.add_argument('--commitment_backend', default=None, type=str)
parser.add_argument('--artifact_store_backend', default=None, type=str)
parser.add_argument('--ledger_backend', default=None, type=str)
parser.add_argument('--commitment_architecture_enabled', default=None, type=str_to_bool)
parser.add_argument('--commitment_artifact_dirname', default=None, type=str)
parser.add_argument('--commitment_ledger_filename', default=None, type=str)
parser.add_argument('--commitment_quorum_rule', default=None, type=str)
parser.add_argument('--cra_committee_enabled', default=None, type=str_to_bool)
parser.add_argument('--cra_distribution_consistency_enabled', default=None, type=str_to_bool)
parser.add_argument('--cra_parent_commit_enabled', default=None, type=str_to_bool)
parser.add_argument('--cra_commit_rejection_response', default=None, type=str)
parser.add_argument('--cra_distribution_atol', default=None, type=float)
parser.add_argument('--cra_distribution_rtol', default=None, type=float)
parser.add_argument('--cra_attack_blocked_zero_metrics', default=None, type=str_to_bool)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {
    'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
    'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']},
}
cfg['local_train_size'] = 10 if args['local_train_size'] is None else int(args['local_train_size'])
cfg['distribute_init_val'] = 0.25
cfg['file_output'] = "New_Tables/MNIST_ConvRate_TEST"
cfg.setdefault('validation_response', 'zero_change')
cfg.setdefault('round_log_dir', os.path.join('output', 'round_log', 'prototype2'))
cfg.setdefault('verifier_val_size', 32)
cfg.setdefault('verifier_max_loss_increase', 0.05)
cfg.setdefault('verifier_max_acc_drop', 10.0)
cfg.setdefault('verifier_min_relative_change', 1.0e-5)
cfg.setdefault('verifier_max_relative_change', 0.6)
cfg.setdefault('verifier_structure_warmup_rounds', 2)
cfg.setdefault('verifier_behavior_freeze_enabled', True)
cfg.setdefault('verifier_min_behavior_loss_delta', 0.0003)
cfg.setdefault('verifier_min_behavior_acc_delta', 1.0e-9)
cfg.setdefault('verifier_behavior_freeze_min_relative_change', 0.02)
cfg.setdefault('verifier_behavior_freeze_max_relative_change', 0.20)
cfg.setdefault('experiment_method', 'cra_base')
cfg.setdefault('experiment_id', None)
cfg.setdefault('results_dir', 'results')
cfg.setdefault('leakage_results_csv', '{results_dir}/leakage_raw.csv')
cfg.setdefault('epoch_results_csv', '{results_dir}/epoch_raw.csv')
cfg.setdefault('overhead_results_csv', '{results_dir}/overhead_raw.csv')
cfg.setdefault('enable_experiment_logging', True)
cfg.setdefault('attack_blocked_zero_metrics', True)
cfg.setdefault('recovered_pearson_threshold', 0.98)
cfg.setdefault('convergence_mode', False)
cfg.setdefault('disable_attack_for_convergence', False)
cfg.setdefault('commitment_backend', 'local')
cfg.setdefault('artifact_store_backend', 'local')
cfg.setdefault('ledger_backend', 'json')
cfg.setdefault('commitment_architecture_enabled', True)
cfg.setdefault('commitment_artifact_dirname', 'artifacts')
cfg.setdefault('commitment_ledger_filename', 'commitment_ledger.json')
cfg.setdefault('commitment_quorum_rule', 'majority')
cfg.setdefault('cra_committee_enabled', False)
cfg.setdefault('cra_distribution_consistency_enabled', True)
cfg.setdefault('cra_parent_commit_enabled', True)
cfg.setdefault('cra_commit_rejection_response', 'rollback_previous')
cfg.setdefault('cra_distribution_atol', 1.0e-8)
cfg.setdefault('cra_distribution_rtol', 1.0e-6)
cfg.setdefault('cra_attack_blocked_zero_metrics', True)
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
if args['attack_blocked_zero_metrics'] is not None:
    cfg['attack_blocked_zero_metrics'] = args['attack_blocked_zero_metrics']
if args['recovered_pearson_threshold'] is not None:
    cfg['recovered_pearson_threshold'] = float(args['recovered_pearson_threshold'])
if args['convergence_mode'] is not None:
    cfg['convergence_mode'] = args['convergence_mode']
if args['disable_attack_for_convergence'] is not None:
    cfg['disable_attack_for_convergence'] = args['disable_attack_for_convergence']
if args['commitment_backend'] is not None:
    cfg['commitment_backend'] = args['commitment_backend']
if args['artifact_store_backend'] is not None:
    cfg['artifact_store_backend'] = args['artifact_store_backend']
if args['ledger_backend'] is not None:
    cfg['ledger_backend'] = args['ledger_backend']
if args['commitment_architecture_enabled'] is not None:
    cfg['commitment_architecture_enabled'] = args['commitment_architecture_enabled']
if args['commitment_artifact_dirname'] is not None:
    cfg['commitment_artifact_dirname'] = args['commitment_artifact_dirname']
if args['commitment_ledger_filename'] is not None:
    cfg['commitment_ledger_filename'] = args['commitment_ledger_filename']
if args['commitment_quorum_rule'] is not None:
    cfg['commitment_quorum_rule'] = args['commitment_quorum_rule']
if args['cra_committee_enabled'] is not None:
    cfg['cra_committee_enabled'] = args['cra_committee_enabled']
if args['cra_distribution_consistency_enabled'] is not None:
    cfg['cra_distribution_consistency_enabled'] = args['cra_distribution_consistency_enabled']
if args['cra_parent_commit_enabled'] is not None:
    cfg['cra_parent_commit_enabled'] = args['cra_parent_commit_enabled']
if args['cra_commit_rejection_response'] is not None:
    cfg['cra_commit_rejection_response'] = args['cra_commit_rejection_response']
if args['cra_distribution_atol'] is not None:
    cfg['cra_distribution_atol'] = float(args['cra_distribution_atol'])
if args['cra_distribution_rtol'] is not None:
    cfg['cra_distribution_rtol'] = float(args['cra_distribution_rtol'])
if args['cra_attack_blocked_zero_metrics'] is not None:
    cfg['cra_attack_blocked_zero_metrics'] = args['cra_attack_blocked_zero_metrics']


def safe_cfg_get(*keys, default=''):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def committee_enabled():
    return bool(cfg.get('cra_committee_enabled', False))


def distribution_check_enabled():
    return committee_enabled() and bool(cfg.get('cra_distribution_consistency_enabled', True))


def parent_commit_enabled():
    return committee_enabled() and bool(cfg.get('cra_parent_commit_enabled', True))


def attack_execution_enabled():
    return not (
        bool(safe_cfg_get('convergence_mode', default=False))
        and bool(safe_cfg_get('disable_attack_for_convergence', default=False))
    )


def attack_round_enabled(epoch):
    return attack_execution_enabled() and int(epoch) == int(cfg['num_epochs']['global'])


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
        return str(explicit)
    return 'committee_cra' if committee_enabled() else 'cra_base'


def detect_dp_mode():
    method = str(cfg.get('experiment_method', 'unknown')).lower()
    if method in {'ldp', 'ddp'}:
        return method
    return 'none'


def resolve_experiment_defaults():
    cfg['experiment_method'] = infer_experiment_method()
    if cfg.get('experiment_id') in (None, '', 'default'):
        cfg['experiment_id'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
            'committee_enabled': committee_enabled(),
            'committee_approved': '',
            'candidate_rejected': False,
            'attack_blocked': False,
        }
    approved = bool(event.get('approved', False))
    return {
        'committee_enabled': committee_enabled(),
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
        'committee_enabled', 'committee_approved', 'attack_blocked',
        'commit_rejection_response', 'best_pearson', 'avg_pearson',
        'best_psnr', 'avg_psnr', 'num_recovered', 'total_runtime_sec',
        'final_global_accuracy', 'final_global_loss', 'cra_distribution_violation',
        'cra_distribution_violation_user', 'cra_distribution_relative_change',
        'cra_parent_commit_approved',
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


def commitment_architecture_enabled():
    return bool(cfg.get('commitment_architecture_enabled', True))


def resolve_commitment_quorum_rule():
    configured_rule = str(cfg.get('commitment_quorum_rule', 'majority'))
    if args.get('commitment_quorum_rule') is not None:
        return configured_rule
    if configured_rule != 'majority':
        return configured_rule
    if committee_enabled():
        return 'committee_unanimous'
    return configured_rule


def create_commitment_backend():
    round_log_dir = cfg.get('round_log_dir') or os.path.join('output', 'round_log', 'prototype2')

    if not commitment_architecture_enabled():
        return TransparencyLog(round_log_dir)

    artifact_backend = str(cfg.get('artifact_store_backend', 'local')).lower()
    ledger_backend = str(cfg.get('ledger_backend', 'json')).lower()
    commitment_backend = str(cfg.get('commitment_backend', 'local')).lower()

    if commitment_backend != 'local':
        raise ValueError(f'Unsupported commitment_backend for CRA: {commitment_backend}')
    if artifact_backend != 'local':
        raise ValueError(f'Unsupported artifact_store_backend for CRA: {artifact_backend}')
    if ledger_backend != 'json':
        raise ValueError(f'Unsupported ledger_backend for CRA: {ledger_backend}')

    artifact_store = LocalArtifactStore(
        round_log_dir,
        artifact_dirname=cfg.get('commitment_artifact_dirname', 'artifacts'),
    )
    ledger = JsonCommitmentLedger(
        round_log_dir,
        ledger_filename=cfg.get('commitment_ledger_filename', 'commitment_ledger.json'),
    )
    return CommitmentService(artifact_store, ledger, cfg)


def bootstrap_commitment_backend(commitment_backend, initial_state_dict):
    if isinstance(commitment_backend, CommitmentService):
        return commitment_backend.ensure_genesis_parent(
            initial_state_dict,
            metadata={
                'source': 'bootstrap',
                'parent_for_round': 1,
                'pipeline': 'cra',
            },
        )
    return commitment_backend.bootstrap_initial_parent(initial_state_dict, parent_for_round=1)


def fetch_latest_approved_parent(commitment_backend):
    if isinstance(commitment_backend, CommitmentService):
        parent_record = commitment_backend.get_latest_approved_parent()
        parent_state = commitment_backend.get_latest_approved_parent_state()
        return parent_record, parent_state

    parent_record = commitment_backend.get_latest_approved_parent()
    parent_state = commitment_backend.load_parent_state_dict(parent_record)
    return parent_record, parent_state


def get_parent_hash(parent_record):
    if parent_record is None:
        return None
    if hasattr(parent_record, 'parent_hash'):
        return str(parent_record.parent_hash)
    return str(parent_record.get('model_hash'))


def get_round_log_events_path(commitment_backend):
    round_log_dir = cfg.get('round_log_dir') or os.path.join('output', 'round_log', 'prototype2')
    if isinstance(commitment_backend, CommitmentService):
        return os.path.join(round_log_dir, 'commitment_events.jsonl')
    return os.path.join(round_log_dir, 'events.jsonl')


def append_round_log_event(commitment_backend, event):
    path = get_round_log_events_path(commitment_backend)
    ensure_dir(os.path.dirname(path))
    payload = copy.deepcopy(event)
    if isinstance(commitment_backend, CommitmentService):
        payload = {
            'event_type': event['event_type'],
            'record_id': 'cra-{:d}'.format(int(time.time() * 1000000)),
            'timestamp': time.time(),
            'payload': payload,
        }
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Avg_Pearson Max_PSNR Avg_PSNR Max_Recovered\n")


def compute_local_parameter_diff(received, expected, atol=1e-8, rtol=1e-6):
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_numel = 0
    total_diff_sq = 0.0
    total_expected_sq = 0.0
    num_mismatched_tensors = 0
    reason = 'ok'

    for k in expected:
        if k not in received:
            return False, {
                'reason': f'missing key: {k}',
                'max_abs_diff': float('inf'),
                'mean_abs_diff': float('inf'),
                'relative_diff': float('inf'),
                'num_mismatched_tensors': len(expected),
            }

        recv = received[k]
        exp = expected[k]

        if recv.shape != exp.shape:
            return False, {
                'reason': f'shape mismatch for {k}: got={tuple(recv.shape)} expected={tuple(exp.shape)}',
                'max_abs_diff': float('inf'),
                'mean_abs_diff': float('inf'),
                'relative_diff': float('inf'),
                'num_mismatched_tensors': len(expected),
            }

        recv = recv.to(device=exp.device, dtype=exp.dtype)
        diff = torch.abs(recv - exp).detach().float()
        max_abs_diff = max(max_abs_diff, float(torch.max(diff).item()) if diff.numel() > 0 else 0.0)
        total_abs_diff += float(torch.sum(diff).item())
        total_numel += int(diff.numel())
        total_diff_sq += float(torch.sum((recv.detach().float() - exp.detach().float()) ** 2).item())
        total_expected_sq += float(torch.sum(exp.detach().float() ** 2).item())

        if not torch.allclose(recv, exp, atol=atol, rtol=rtol):
            num_mismatched_tensors += 1
            if reason == 'ok':
                reason = f'value mismatch for {k}: max_diff={float(torch.max(diff).item()):.6e}'

    mean_abs_diff = float(total_abs_diff / max(total_numel, 1))
    relative_diff = float(np.sqrt(total_diff_sq) / (np.sqrt(total_expected_sq) + 1.0e-12))
    matched = num_mismatched_tensors == 0
    return matched, {
        'reason': reason,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'relative_diff': relative_diff,
        'num_mismatched_tensors': num_mismatched_tensors,
    }


def compare_local_parameters(received, expected, atol=1e-8, rtol=1e-6):
    matched, metrics = compute_local_parameter_diff(received, expected, atol=atol, rtol=rtol)
    return matched, metrics['reason']


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


def cast_local_parameters_to_reference(local_parameters, expected_local_parameters):
    if expected_local_parameters is None:
        return local_parameters
    fixed = [OrderedDict() for _ in range(len(local_parameters))]
    for m in range(len(local_parameters)):
        for k in local_parameters[m]:
            if k in expected_local_parameters[m]:
                fixed[m][k] = local_parameters[m][k].to(
                    device=expected_local_parameters[m][k].device,
                    dtype=expected_local_parameters[m][k].dtype,
                )
            else:
                fixed[m][k] = local_parameters[m][k]
    return fixed


def build_candidate_parent_for_commitment(epoch, federation, honest_aggregated_state):
    previous_parent_state = clone_state_dict(federation.initial_parent_state)
    candidate_state = clone_state_dict(honest_aggregated_state)
    previous_hash = hash_state_dict(previous_parent_state)
    candidate_hash = hash_state_dict(candidate_state)
    candidate_meta = {
        'candidate_source': 'honest_aggregate',
        'round_produced': int(epoch),
        'target_parent_round': int(epoch) + 1,
        'previous_parent_hash': previous_hash,
        'honest_aggregated_hash': candidate_hash,
        'candidate_hash': candidate_hash,
        'relative_change_vs_previous_parent': state_dict_relative_l2(previous_parent_state, candidate_state),
        'relative_change_vs_honest_aggregated': 0.0,
    }
    return candidate_state, candidate_meta


def select_verifiers(local, user_idx, federation):
    selected = OrderedDict()
    for m in range(len(user_idx)):
        rate = float(federation.model_rate[user_idx[m]])
        if rate not in selected:
            selected[rate] = (m, int(user_idx[m]), local[m])
    return list(selected.values())


def build_static_verifier_submodel(parent_state_dict, verifier_user_id, verifier_rate):
    return extract_honest_static_submodel_from_parent(
        clone_state_dict(parent_state_dict),
        verifier_rate,
        verifier_user_id,
        cfg,
    )


def verify_and_commit_candidate_parent(
    commitment_backend,
    epoch,
    candidate_state_dict,
    local,
    user_idx,
    federation,
    logger,
    candidate_meta=None,
):
    if isinstance(commitment_backend, CommitmentService):
        return verify_and_commit_candidate_parent_via_service(
            commitment_backend,
            epoch,
            candidate_state_dict,
            local,
            user_idx,
            federation,
            logger,
            candidate_meta=candidate_meta,
        )

    return verify_and_commit_candidate_parent_legacy(
        commitment_backend,
        epoch,
        candidate_state_dict,
        local,
        user_idx,
        federation,
        logger,
        candidate_meta=candidate_meta,
    )


def verify_candidate_against_committee(
    epoch,
    candidate_state_dict,
    previous_parent_state,
    local,
    user_idx,
    federation,
    logger,
):
    committee = select_verifiers(local, user_idx, federation)
    verifier_reports = []

    for _, verifier_user_id, verifier_local in committee:
        verifier_rate = float(federation.model_rate[verifier_user_id])
        prev_local_parameters = build_static_verifier_submodel(previous_parent_state, verifier_user_id, verifier_rate)
        cand_local_parameters = build_static_verifier_submodel(candidate_state_dict, verifier_user_id, verifier_rate)

        prev_eval = evaluate_local_parameters(
            prev_local_parameters,
            verifier_rate,
            verifier_local.data_loader,
            verifier_local.label_split,
            max_steps=cfg['verifier_val_size'],
        )
        cand_eval = evaluate_local_parameters(
            cand_local_parameters,
            verifier_rate,
            verifier_local.data_loader,
            verifier_local.label_split,
            max_steps=cfg['verifier_val_size'],
        )

        rel_change = relative_model_change(prev_local_parameters, cand_local_parameters)
        approved = True
        reason = 'ok'
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
            'cohort_rate': verifier_rate,
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
                    f'[COMMIT][Verifier {verifier_user_id}] cohort_rate={verifier_rate}',
                    f'[COMMIT][Verifier {verifier_user_id}] approved={approved}',
                    f'[COMMIT][Verifier {verifier_user_id}] reason={reason}',
                    f'[COMMIT][Verifier {verifier_user_id}] prev_loss={prev_eval["Local-Loss"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] cand_loss={cand_eval["Local-Loss"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] prev_acc={prev_eval["Local-Accuracy"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] cand_acc={cand_eval["Local-Accuracy"]:.6f}',
                    f'[COMMIT][Verifier {verifier_user_id}] rel_change={rel_change:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_loss_delta={behavior_report["behavior_loss_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_acc_delta={behavior_report["behavior_acc_delta"]:.6e}',
                    f'[COMMIT][Verifier {verifier_user_id}] behavior_frozen={behavior_report["behavior_frozen"]}',
                ]
            }, 'train', mean=False)

    return committee, verifier_reports


def verify_and_commit_candidate_parent_via_service(
    commitment_service,
    epoch,
    candidate_state_dict,
    local,
    user_idx,
    federation,
    logger,
    candidate_meta=None,
):
    round_id = int(epoch) + 1
    previous_parent_record, previous_parent_state = commitment_service.fetch_previous_approved_parent()
    if previous_parent_record is None or previous_parent_state is None:
        raise RuntimeError('No approved parent is available for committee verification')

    committee, verifier_reports = verify_candidate_against_committee(
        epoch,
        candidate_state_dict,
        previous_parent_state,
        local,
        user_idx,
        federation,
        logger,
    )
    committee_ids = [int(verifier_user_id) for _, verifier_user_id, _ in committee]
    active_cohorts = sorted({float(federation.model_rate[uid]) for uid in user_idx})
    active_user_model_rates = {int(uid): float(federation.model_rate[uid]) for uid in user_idx}
    candidate_metadata = dict(candidate_meta or {})
    candidate_metadata.update({
        'active_users': [int(uid) for uid in user_idx],
        'active_user_model_rates': active_user_model_rates,
        'required_cohort_rates': active_cohorts,
    })

    candidate_record, candidate_submission_id = commitment_service.submit_candidate_parent(
        round_id=round_id,
        candidate_state_dict=copy.deepcopy(candidate_state_dict),
        previous_parent_hash=previous_parent_record.parent_hash,
        schedule_id=str(candidate_metadata.get('candidate_source', f'round-{round_id}')),
        active_cohorts=active_cohorts,
        committee_ids=committee_ids,
        metadata=candidate_metadata,
    )

    report_submission_ids = []
    for report in verifier_reports:
        report_record = VerificationReport(
            round_id=round_id,
            verifier_user_id=int(report['user_id']),
            cohort_rate=float(report['cohort_rate']),
            approved=bool(report['approved']),
            reason=str(report['reason']),
            relative_change=float(report['relative_change']),
            prev_eval=dict(report['prev_eval']),
            cand_eval=dict(report['cand_eval']),
            metrics={
                'structure_checks_enabled': bool(report['structure_checks_enabled']),
                'verifier_min_relative_change': float(report['verifier_min_relative_change']),
                'verifier_max_relative_change': float(report['verifier_max_relative_change']),
                'behavior_freeze_enabled': bool(report['behavior_freeze_enabled']),
                'behavior_loss_delta': float(report['behavior_loss_delta']),
                'behavior_acc_delta': float(report['behavior_acc_delta']),
                'verifier_min_behavior_loss_delta': float(report['verifier_min_behavior_loss_delta']),
                'verifier_min_behavior_acc_delta': float(report['verifier_min_behavior_acc_delta']),
                'verifier_behavior_freeze_max_relative_change': float(report['verifier_behavior_freeze_max_relative_change']),
                'behavior_frozen': bool(report['behavior_frozen']),
            },
            timestamp=time.time(),
        )
        report_submission_ids.append(commitment_service.submit_verification_report(report_record))

    for report, report_submission_id in zip(verifier_reports, report_submission_ids):
        report['report_submission_id'] = report_submission_id

    decision = commitment_service.finalize_round(
        round_id,
        quorum_rule=resolve_commitment_quorum_rule(),
    )
    latest_approved_parent = commitment_service.get_latest_approved_parent()
    required = sorted({float(r['cohort_rate']) for r in verifier_reports})
    approved_cohorts = sorted({float(r['cohort_rate']) for r in verifier_reports if r.get('approved', False)})
    event = {
        'event_type': 'approved_parent' if decision.approved else 'candidate_parent',
        'round_produced': int(epoch),
        'parent_for_round': round_id,
        'model_hash': candidate_record.candidate_parent_hash,
        'artifact_ref': {
            'cid': candidate_record.candidate_artifact.cid,
            'uri': candidate_record.candidate_artifact.uri,
            'sha256': candidate_record.candidate_artifact.sha256,
            'size_bytes': candidate_record.candidate_artifact.size_bytes,
            'encrypted': candidate_record.candidate_artifact.encrypted,
            'encryption_alg': candidate_record.candidate_artifact.encryption_alg,
        },
        'active_users': [int(u) for u in user_idx],
        'active_user_model_rates': {str(int(k)): float(v) for k, v in active_user_model_rates.items()},
        'required_cohort_rates': required,
        'approved_cohort_rates': approved_cohorts,
        'verifier_reports': verifier_reports,
        'approved': bool(decision.approved),
        'timestamp': int(decision.timestamp),
        'candidate_metadata': candidate_metadata,
        'commitment_id': candidate_submission_id,
        'quorum_decision_reason': decision.reason,
        'quorum_rule': decision.quorum_rule,
        'decision_num_approved': int(decision.num_approved),
        'decision_num_rejected': int(decision.num_rejected),
        'latest_approved_parent_hash': latest_approved_parent.parent_hash if latest_approved_parent else None,
    }
    return event


def verify_and_commit_candidate_parent_legacy(
    round_log_backend,
    epoch,
    candidate_state_dict,
    local,
    user_idx,
    federation,
    logger,
    candidate_meta=None,
):
    previous_parent_record = round_log_backend.get_latest_approved_parent()
    previous_parent_state = round_log_backend.load_parent_state_dict(previous_parent_record)

    _, verifier_reports = verify_candidate_against_committee(
        epoch,
        candidate_state_dict,
        previous_parent_state,
        local,
        user_idx,
        federation,
        logger,
    )

    active_user_model_rates = {int(uid): float(federation.model_rate[uid]) for uid in user_idx}
    return round_log_backend.record_candidate_parent(
        epoch=epoch,
        state_dict=copy.deepcopy(candidate_state_dict),
        active_users=user_idx,
        active_user_model_rates=active_user_model_rates,
        verifier_reports=verifier_reports,
        candidate_metadata=candidate_meta,
    )


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        'cra_distribution_violation': False,
        'cra_distribution_violation_user': '',
        'cra_distribution_relative_change': 0.0,
        'cra_parent_commit_approved': '',
    }
    total_runtime_start = now_seconds()

    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(
            model, cfg['model_tag'], optimizer, scheduler
        )
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

    global_parameters = clone_state_dict(model.state_dict())
    model_dist = OrderedDict()
    commitment_backend = create_commitment_backend() if committee_enabled() else None
    if commitment_backend is not None:
        bootstrap_commitment_backend(commitment_backend, global_parameters)

    model_history = {
        'blocks.0.bias': [],
        'blocks.2.weight': [],
        'blocks.2.bias': [],
    }
    model_history_fcnn = {
        'layers.0.weight': [],
        'layers.0.bias': [],
    }
    runtime_control = {
        'skip_next_epoch': False,
        'skip_reason': None,
    }

    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        epoch_start = now_seconds()
        logger.safe(True)

        approved_parent_hash = None
        if committee_enabled():
            approved_parent_record, approved_parent_state = fetch_latest_approved_parent(commitment_backend)
            approved_parent_hash = get_parent_hash(approved_parent_record)
            global_parameters = move_state_dict_to_device(approved_parent_state, cfg['device'])
        else:
            global_parameters = move_state_dict_to_device(global_parameters, cfg['device'])

        federation = Federation(epoch, model_dist, global_parameters, cfg['model_rate'], label_split)
        train_context = train(
            model_history,
            model_history_fcnn,
            dataset['train'],
            data_split['train'],
            label_split,
            federation,
            model,
            optimizer,
            logger,
            epoch,
            commitment_backend,
            runtime_control,
            experiment_tracker,
            approved_parent_hash,
        )
        model_dist = copy.deepcopy(federation.model_to_distribute)
        global_parameters = clone_state_dict(model.state_dict())

        test_start = now_seconds()
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        test_time_sec = now_seconds() - test_start

        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean.get('train/{}'.format(cfg['pivot_metric']), 0.0))
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
            'cfg': cfg,
            'epoch': epoch + 1,
            'data_split': data_split,
            'label_split': label_split,
            'model_dict': model_state_dict,
            'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'logger': logger,
        }
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean.get('test/{}'.format(cfg['pivot_metric']), -float('inf')):
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy(
                './output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                './output/model/{}_best.pt'.format(cfg['model_tag']),
            )
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
    if experiment_tracker.get('attack_blocked', False) and cfg.get('cra_attack_blocked_zero_metrics', True):
        reconstruction_metrics = empty_reconstruction_metrics()

    leakage_row = {
        **build_common_result_fields(seed),
        'global_epochs': cfg['num_epochs']['global'],
        'local_epochs': cfg['num_epochs']['local'],
        'local_train_size': cfg['local_train_size'],
        'batch_size_train': safe_cfg_get('batch_size', default={}).get('train', ''),
        'attack_source_round': cfg['num_epochs']['global'],
        'attack_replay_round': cfg['num_epochs']['global'],
        'committee_enabled': committee_enabled(),
        'committee_approved': experiment_tracker.get('committee_approved', ''),
        'attack_blocked': experiment_tracker.get('attack_blocked', False),
        'commit_rejection_response': cfg.get('cra_commit_rejection_response', ''),
        'best_pearson': reconstruction_metrics['best_pearson'],
        'avg_pearson': reconstruction_metrics['avg_pearson'],
        'best_psnr': reconstruction_metrics['best_psnr'],
        'avg_psnr': reconstruction_metrics['avg_psnr'],
        'num_recovered': reconstruction_metrics['num_recovered'],
        'total_runtime_sec': total_runtime_sec,
        'final_global_accuracy': experiment_tracker['final_global_accuracy'],
        'final_global_loss': experiment_tracker['final_global_loss'],
        'cra_distribution_violation': experiment_tracker.get('cra_distribution_violation', False),
        'cra_distribution_violation_user': experiment_tracker.get('cra_distribution_violation_user', ''),
        'cra_distribution_relative_change': experiment_tracker.get('cra_distribution_relative_change', 0.0),
        'cra_parent_commit_approved': experiment_tracker.get('cra_parent_commit_approved', ''),
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


def train(
    model_history,
    model_history_fcnn,
    dataset,
    data_split,
    label_split,
    federation,
    global_model,
    optimizer,
    logger,
    epoch,
    commitment_backend,
    runtime_control,
    experiment_tracker,
    approved_parent_hash,
):
    global_model.load_state_dict(federation.global_parameters)
    attack_enabled = attack_execution_enabled()
    attack_round = attack_round_enabled(epoch)
    committee_status = {
        'committee_enabled': committee_enabled(),
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
                '[EPOCH-NOOP] local training, aggregation, and reconstruction were skipped',
            ]
        }, 'train', mean=False)
        return {
            'train_time_sec': train_time_sec,
            'aggregation_time_sec': aggregation_time_sec,
            'committee_time_sec': committee_time_sec,
            'dp_time_sec': dp_time_sec,
            'attack_enabled': attack_enabled,
            'attack_round': attack_round,
            'committee_enabled': committee_enabled(),
            'committee_approved_this_epoch': '',
            'candidate_rejected_this_epoch': False,
        }

    global_model.train(True)
    local, local_parameters, user_idx, param_idx, expected_local_parameters, distribution_reports = make_local(
        dataset,
        data_split,
        label_split,
        federation,
        logger,
    )
    distributed_local_parameters = copy.deepcopy(local_parameters)
    num_active_users = len(local)

    if distribution_check_enabled():
        committee_timer_start = now_seconds()
        violating_report = next((report for report in distribution_reports if not report['match']), None)
        committee_time_sec += now_seconds() - committee_timer_start
        if violating_report is not None:
            event = {
                'event_type': 'cra_distribution_violation',
                'epoch': int(epoch),
                'round_produced': int(epoch),
                'user_id': int(violating_report['user_id']),
                'model_rate': float(violating_report['model_rate']),
                'cohort': float(violating_report['model_rate']),
                'expected_parent_hash': approved_parent_hash or hash_state_dict(federation.initial_parent_state),
                'max_abs_diff': float(violating_report['max_abs_diff']),
                'mean_abs_diff': float(violating_report['mean_abs_diff']),
                'relative_diff': float(violating_report['relative_diff']),
                'num_mismatched_tensors': int(violating_report['num_mismatched_tensors']),
                'reason': str(violating_report['reason']),
            }
            append_round_log_event(commitment_backend, event)
            logger.append({
                'info': [
                    f'[CRA-COMMIT] distribution violation blocked round {epoch}',
                    f'[CRA-COMMIT] user_id={event["user_id"]}',
                    f'[CRA-COMMIT] model_rate={event["model_rate"]}',
                    f'[CRA-COMMIT] expected_parent_hash={event["expected_parent_hash"]}',
                    f'[CRA-COMMIT] relative_diff={event["relative_diff"]:.6e}',
                    f'[CRA-COMMIT] reason={event["reason"]}',
                ]
            }, 'train', mean=False)
            experiment_tracker['attack_blocked'] = True
            experiment_tracker['cra_distribution_violation'] = True
            experiment_tracker['cra_distribution_violation_user'] = int(event['user_id'])
            experiment_tracker['cra_distribution_relative_change'] = float(event['relative_diff'])
            experiment_tracker['committee_approved'] = False
            experiment_tracker['cra_parent_commit_approved'] = False
            if cfg.get('cra_attack_blocked_zero_metrics', True):
                experiment_tracker['last_reconstruction'] = empty_reconstruction_metrics()
                fp.write("%s %s %s %s %s %s\n" % (
                    cfg['local_train_size'], 0.0, 0.0, 0.0, 0.0, 0,
                ))
            federation.global_parameters = clone_state_dict(federation.initial_parent_state)
            global_model.load_state_dict(federation.initial_parent_state)
            return {
                'train_time_sec': train_time_sec,
                'aggregation_time_sec': aggregation_time_sec,
                'committee_time_sec': committee_time_sec,
                'dp_time_sec': dp_time_sec,
                'attack_enabled': attack_enabled,
                'attack_round': attack_round,
                'committee_enabled': committee_enabled(),
                'committee_approved_this_epoch': False,
                'candidate_rejected_this_epoch': False,
            }

    target_users = []
    converged_users = []
    for m in range(num_active_users):
        if federation.model_rate[user_idx[m]] == 0.5:
            target_users.append(user_idx[m])
        if federation.model_rate[user_idx[m]] == 1:
            converged_users.append(user_idx[m])
    num_updaters = len(target_users) + len(converged_users)

    start_time = time.time()
    train_timer_start = now_seconds()
    img_list_by_user = {}
    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]]
        local_parameters[m], img_data = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if img_data:
            img_list_by_user[int(user_idx[m])] = copy.deepcopy(img_data)
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users)
            )
            info = {'info': [
                'Model: {}'.format(cfg['model_tag']),
                'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                'Learning rate: {}'.format(lr),
                'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                'Epoch Finished Time: {}'.format(epoch_finished_time),
                'Experiment Finished Time: {}'.format(exp_finished_time),
            ]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])
    train_time_sec = now_seconds() - train_timer_start

    aggregation_timer_start = now_seconds()
    federation.combine(local_parameters, param_idx, user_idx)
    aggregation_time_sec = now_seconds() - aggregation_timer_start

    honest_aggregated_state = clone_state_dict(federation.global_parameters)
    final_parent_state = honest_aggregated_state

    if parent_commit_enabled():
        candidate_parent_state, candidate_meta = build_candidate_parent_for_commitment(
            epoch,
            federation,
            honest_aggregated_state,
        )
        committee_timer_start = now_seconds()
        commitment_event = verify_and_commit_candidate_parent(
            commitment_backend,
            epoch,
            copy.deepcopy(candidate_parent_state),
            local,
            user_idx,
            federation,
            logger,
            candidate_meta=candidate_meta,
        )
        committee_time_sec += now_seconds() - committee_timer_start
        committee_status = summarize_committee_status(commitment_event)
        experiment_tracker['committee_approved'] = committee_status['committee_approved']
        experiment_tracker['cra_parent_commit_approved'] = committee_status['committee_approved']
        experiment_tracker['attack_blocked'] = bool(
            experiment_tracker.get('attack_blocked', False) or committee_status['attack_blocked']
        )

        if commitment_event['approved']:
            final_parent_state = clone_state_dict(candidate_parent_state)
        else:
            rejection_response = cfg.get('cra_commit_rejection_response', 'rollback_previous')
            logger.append({
                'info': [
                    f'[CRA-COMMIT] parent candidate rejected for round {epoch + 1}',
                    f'[CRA-COMMIT] rejection_response={rejection_response}',
                    f'[CRA-COMMIT] commitment_id={commitment_event["commitment_id"]}',
                ]
            }, 'train', mean=False)

            if rejection_response == 'abort':
                raise RuntimeError(
                    f'CRA candidate parent rejected for round {epoch + 1}: {commitment_event["commitment_id"]}'
                )

            approved_parent_record, rollback_state = fetch_latest_approved_parent(commitment_backend)
            rollback_state = move_state_dict_to_device(rollback_state, cfg['device'])
            final_parent_state = clone_state_dict(rollback_state)
            if rejection_response == 'rollback_previous':
                runtime_control['skip_next_epoch'] = True
                runtime_control['skip_reason'] = 'candidate_rejected_round_voided'
            elif rejection_response == 'fallback_honest':
                runtime_control['skip_reason'] = 'fallback_honest_unavailable_for_cra'
            else:
                raise ValueError(f"Unknown cra_commit_rejection_response: {rejection_response}")

    federation.global_parameters = clone_state_dict(final_parent_state)
    global_model.load_state_dict(final_parent_state)
    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())

    round_reconstruction_metrics = empty_reconstruction_metrics()
    round_blocked = bool(committee_status.get('attack_blocked', False))
    if cfg['model_name'] == 'fcnn' and not round_blocked:
        model_history_fcnn['layers.0.weight'].append(global_model_state_dict_copy['layers.0.weight'])
        model_history_fcnn['layers.0.bias'].append(global_model_state_dict_copy['layers.0.bias'])

        for m in range(num_active_users):
            if user_idx[m] not in target_users:
                continue

            weight_grad = None
            bias_grad = None
            for k in global_model.state_dict().keys():
                if attack_round and k == 'layers.0.weight':
                    weight_grad = fcnn_leakage(
                        epoch, k, user_idx[m], num_active_users,
                        federation.model_rate[user_idx[m]],
                        local_parameters[m][k],
                        model_history_fcnn[k],
                        distributed_local_parameters[m][k],
                    )
                elif attack_round and k == 'layers.0.bias':
                    bias_grad = fcnn_leakage(
                        epoch, k, user_idx[m], num_active_users,
                        federation.model_rate[user_idx[m]],
                        local_parameters[m][k],
                        model_history_fcnn[k],
                        distributed_local_parameters[m][k],
                    )

            if weight_grad is not None and bias_grad is not None:
                bias_grad_sum = torch.abs(torch.sum(bias_grad)).item()
                if bias_grad_sum != 0.0:
                    user_metrics = reconstruct_image(
                        weight_grad,
                        bias_grad,
                        img_list_by_user.get(int(user_idx[m]), []),
                    )
                    round_reconstruction_metrics = merge_reconstruction_metrics(
                        round_reconstruction_metrics,
                        user_metrics,
                    )

    if cfg['model_name'] == 'conv':
        model_history['blocks.0.bias'].append(global_model_state_dict_copy['blocks.0.bias'])
        model_history['blocks.2.weight'].append(global_model_state_dict_copy['blocks.2.weight'])
        model_history['blocks.2.bias'].append(global_model_state_dict_copy['blocks.2.bias'])

        converged_user_gradients = {}
        target_user_gradients = {}
        for m in range(num_active_users):
            local_params_size = local_parameters[m]['blocks.0.bias'].size()[0]
            if user_idx[m] in converged_users and epoch >= 2:
                lower = local_params_size // 4
                upper = local_params_size // 2
                converged_user_gradients['blocks.0.bias'] = (
                    local_parameters[m]['blocks.0.bias'][lower:upper]
                    - model_history['blocks.0.bias'][epoch - 2][lower:upper]
                )
            elif user_idx[m] in target_users and epoch >= 2:
                lower = local_params_size // 2
                upper = local_params_size
                target_user_gradients['blocks.0.bias'] = (
                    local_parameters[m]['blocks.0.bias'][lower:upper]
                    - model_history['blocks.0.bias'][epoch - 2][lower:upper]
                )

        for m in range(num_active_users):
            if user_idx[m] in target_users:
                for k in global_model.state_dict().keys():
                    if k == 'blocks.0.bias' and epoch >= 2:
                        local_params_size = local_parameters[m]['blocks.0.bias'].size()[0]
                        lower = local_params_size // 2
                        upper = local_params_size
                        global_model_with_gradients = copy.deepcopy(model_history['blocks.0.bias'][epoch - 2])
                        global_model_with_gradients_scaled = global_model_with_gradients[lower:upper]
                        global_model_with_gradients_scaled += converged_user_gradients['blocks.0.bias']
                        global_model_with_gradients_scaled += target_user_gradients['blocks.0.bias']
                        gradient_leakage(
                            epoch,
                            k,
                            user_idx[m],
                            num_updaters,
                            local_parameters[m][k],
                            model_history,
                            global_model_with_gradients_scaled,
                            target_user_gradients['blocks.0.bias'],
                        )

    if (not attack_enabled or round_blocked) and cfg.get('cra_attack_blocked_zero_metrics', True):
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

    return {
        'train_time_sec': train_time_sec,
        'aggregation_time_sec': aggregation_time_sec,
        'committee_time_sec': committee_time_sec,
        'dp_time_sec': dp_time_sec,
        'attack_enabled': attack_enabled,
        'attack_round': attack_round,
        'committee_enabled': committee_enabled(),
        'committee_approved_this_epoch': committee_status['committee_approved'],
        'candidate_rejected_this_epoch': committee_status['candidate_rejected'],
    }


def reconstruct_image(weight_grad, bias_grad, img_list):
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0
    threshold = float(safe_cfg_get('recovered_pearson_threshold', default=0.98))

    for elem in img_list:
        elem_extracted = elem[0].to(cfg["device"])
        max_pearson = 0.0
        max_ssim = 0.0
        max_psnr = 0.0
        pearson = PearsonCorrCoef().to(cfg["device"])
        best_partial_recon = None

        for i in range(weight_grad.size()[0]):
            denom = bias_grad[i]
            eps = 1.0e-8
            if torch.abs(denom).item() < eps:
                continue

            partial_recon = weight_grad[i] / denom
            img_reshaped = elem_extracted.reshape(partial_recon.size())
            pearson_coef = pearson(img_reshaped, partial_recon).item()
            partial_recon_as_img = partial_recon.reshape(elem_extracted.size())
            elem_extracted_numpy = np.squeeze(elem_extracted.cpu().numpy())
            partial_recon_numpy = np.squeeze(partial_recon_as_img.cpu().numpy())
            curr_ssim = ssim(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)
            curr_psnr = psnr(elem_extracted_numpy, partial_recon_numpy, data_range=3.245699448)
            max_ssim = max(max_ssim, curr_ssim)
            max_psnr = max(max_psnr, curr_psnr)
            max_pearson = max(max_pearson, pearson_coef)
            if max_pearson == pearson_coef:
                best_partial_recon = partial_recon

        if max_pearson >= threshold:
            num_recovered += 1
        avg_ssim.append(max_ssim)
        avg_psnr.append(max_psnr)
        avg_pearson.append(max_pearson)
        if best_partial_recon is None:
            continue
        _ = best_partial_recon.reshape(elem_extracted.size())

    if len(avg_pearson) == 0:
        return empty_reconstruction_metrics()

    return {
        'best_pearson': max(avg_pearson),
        'avg_pearson': sum(avg_pearson) / len(avg_pearson),
        'best_psnr': max(avg_psnr),
        'avg_psnr': sum(avg_psnr) / len(avg_psnr),
        'num_recovered': num_recovered,
    }


def fcnn_leakage(epoch, k, user, num_active_users, model_rate, local_params, model_history, distributed_model):
    local_params_resized = copy.deepcopy(local_params)
    model_history_params_resized = copy.deepcopy(model_history[epoch - 2])
    model_history_params_curr_resized = copy.deepcopy(model_history[epoch - 1])

    lower = local_params.size()[0] // 2
    upper = local_params.size()[0]
    local_params_resized = local_params_resized[lower:upper]
    model_history_params_resized = model_history_params_resized[lower:upper]
    model_history_params_curr_resized = model_history_params_curr_resized[lower:upper]

    potential_B = torch.multiply(model_history_params_curr_resized, 2)
    potential_B = torch.subtract(potential_B, model_history_params_resized)
    user_grad = torch.subtract(distributed_model[lower:upper].to(cfg["device"]), potential_B)
    user_grad_concat = torch.cat((user_grad, user_grad))
    return user_grad_concat


def gradient_leakage(epoch, k, user, num_updaters, local_params, model_history, global_params, target_gradient):
    global_params_resized = copy.deepcopy(global_params)
    local_params_resized = copy.deepcopy(local_params)
    model_history_params_resized = copy.deepcopy(model_history[k][epoch - 2])
    model_history_params_curr_resized = copy.deepcopy(model_history[k][epoch - 1])

    lower = local_params.size()[0] // 2
    upper = local_params.size()[0]
    local_params_resized = local_params_resized[lower:upper]
    model_history_params_resized = model_history_params_resized[lower:upper]
    model_history_params_curr_resized = model_history_params_curr_resized[lower:upper]

    global_local_param_diff = torch.subtract(model_history_params_curr_resized, local_params_resized)
    global_local_param_diff_L1 = torch.abs(global_local_param_diff)
    count = 0
    for i in range(0, global_local_param_diff.size()[0]):
        if global_local_param_diff_L1[i] < 0.0001:
            count = count + 1
    L1_dist = torch.sum(global_local_param_diff_L1).item()
    global_local_param_diff_squared = torch.square(global_local_param_diff)
    sum_squared_diff = torch.sum(global_local_param_diff_squared).item()
    L2_dist = math.sqrt(sum_squared_diff)
    if epoch == cfg['num_epochs']['global']:
        fp.write("%s %s %s %s\n" % (k, L1_dist, L2_dist, count))
        fp.flush()


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


def make_local(dataset, data_split, label_split, federation, logger):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)

    expected_local_parameters = None
    distribution_reports = []
    if distribution_check_enabled():
        expected_local_parameters = extract_expected_local_parameters(federation, user_idx, param_idx)
        local_parameters = cast_local_parameters_to_reference(local_parameters, expected_local_parameters)

    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(user_idx[m], model_rate_m, data_loader_m, label_split[user_idx[m]])

        if expected_local_parameters is not None:
            match, metrics = compute_local_parameter_diff(
                local_parameters[m],
                expected_local_parameters[m],
                atol=float(cfg.get('cra_distribution_atol', 1.0e-8)),
                rtol=float(cfg.get('cra_distribution_rtol', 1.0e-6)),
            )
            distribution_reports.append({
                'user_id': int(user_idx[m]),
                'model_rate': float(model_rate_m),
                'match': bool(match),
                **metrics,
            })

            if cfg.get('debug_extraction', False):
                received_hash = hash_local_parameter_dict(local_parameters[m])
                expected_hash = hash_local_parameter_dict(expected_local_parameters[m])
                logger.append({
                    'info': [
                        f'[EXTRACT][Round {federation.rd}] user={int(user_idx[m])}',
                        f'[EXTRACT][Round {federation.rd}] rate={float(model_rate_m)}',
                        f'[EXTRACT][Round {federation.rd}] received_hash={received_hash}',
                        f'[EXTRACT][Round {federation.rd}] expected_hash={expected_hash}',
                        f'[EXTRACT][Round {federation.rd}] match={bool(match)}',
                        f'[EXTRACT][Round {federation.rd}] reason={metrics["reason"]}',
                    ]
                }, 'train', mean=False)

    return local, local_parameters, user_idx, param_idx, expected_local_parameters, distribution_reports


class Local:
    def __init__(self, user_id, model_rate, data_loader, label_split):
        self.user_id = user_id
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split

    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
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
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return local_parameters, input_list


if __name__ == "__main__":
    main()
