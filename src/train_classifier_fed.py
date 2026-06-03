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
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import round_log as round_log_module
from commitment_architecture import (
    CommitmentService,
    EthereumCommitmentLedger,
    IPFSArtifactStore,
    JsonCommitmentLedger,
    LocalArtifactStore,
    VerificationReport,
)
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
parser.add_argument('--metrics-dir', '--metrics_dir', dest='metrics_dir', default=None, type=str)
parser.add_argument('--experiment-tag', '--experiment_tag', dest='experiment_tag', default=None, type=str)
parser.add_argument('--save-recon-images', '--save_recon_images', dest='save_recon_images', default=None, type=str_to_bool)
parser.add_argument('--save-raw-metrics', '--save_raw_metrics', dest='save_raw_metrics', default=None, type=str_to_bool)
parser.add_argument('--defense-mode', '--defense_mode', dest='defense_mode', default=None, type=str)
parser.add_argument('--dp-mode', '--dp_mode', dest='dp_mode', default=None, type=str)
parser.add_argument('--noise-multiplier', '--noise_multiplier', dest='noise_multiplier', default=None, type=float)
parser.add_argument('--clip-norm', '--clip_norm', dest='clip_norm', default=None, type=float)
parser.add_argument('--attack-noise-amount', '--attack_noise_amount', dest='attack_noise_amount', default=None, type=float)
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
parser.add_argument('--ipfs_api_url', default=None, type=str)
parser.add_argument('--ipfs_gateway_url', default=None, type=str)
parser.add_argument('--ipfs_pin_artifacts', default=None, type=str_to_bool)
parser.add_argument('--blockchain_rpc_url', default=None, type=str)
parser.add_argument('--blockchain_chain_id', default=None, type=int)
parser.add_argument('--blockchain_private_key', default=None, type=str)
parser.add_argument('--blockchain_account_index', default=None, type=int)
parser.add_argument('--blockchain_contract_address', default=None, type=str)
parser.add_argument('--blockchain_deploy_contract', default=None, type=str_to_bool)
parser.add_argument('--blockchain_wait_for_receipt', default=None, type=str_to_bool)
parser.add_argument('--blockchain_receipt_timeout_sec', default=None, type=int)
parser.add_argument('--blockchain_gas_limit', default=None, type=int)
parser.add_argument('--blockchain_gas_price_wei', default=None, type=int)
parser.add_argument('--blockchain_store_full_report_json', default=None, type=str_to_bool)
parser.add_argument('--blockchain_report_payload_mode', default=None, type=str)
parser.add_argument('--commitment_measure_real_overhead', default=None, type=str_to_bool)
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
cfg.setdefault('metrics_dir', '{results_dir}/metrics')
cfg.setdefault('plot_ready_dir', '{results_dir}/plot_ready')
cfg.setdefault('experiment_tag', None)
cfg.setdefault('save_recon_images', False)
cfg.setdefault('save_raw_metrics', True)
cfg.setdefault('defense_mode', None)
cfg.setdefault('dp_mode', 'none')
cfg.setdefault('noise_multiplier', None)
cfg.setdefault('clip_norm', None)
cfg.setdefault('attack_noise_amount', 0.0)
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
cfg.setdefault('ipfs_api_url', 'http://127.0.0.1:5001')
cfg.setdefault('ipfs_gateway_url', 'http://127.0.0.1:8081/ipfs')
cfg.setdefault('ipfs_pin_artifacts', True)
cfg.setdefault('blockchain_rpc_url', 'http://127.0.0.1:8545')
cfg.setdefault('blockchain_chain_id', 1337)
cfg.setdefault('blockchain_private_key', None)
cfg.setdefault('blockchain_account_index', 0)
cfg.setdefault('blockchain_contract_address', None)
cfg.setdefault('blockchain_deploy_contract', True)
cfg.setdefault('blockchain_wait_for_receipt', True)
cfg.setdefault('blockchain_receipt_timeout_sec', 120)
cfg.setdefault('blockchain_gas_limit', 8000000)
cfg.setdefault('blockchain_gas_price_wei', None)
cfg.setdefault('blockchain_store_full_report_json', False)
cfg.setdefault('blockchain_report_payload_mode', 'hash_only')
cfg.setdefault('commitment_measure_real_overhead', True)
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
for metrics_key in [
    'metrics_dir',
    'experiment_tag',
    'save_recon_images',
    'save_raw_metrics',
    'defense_mode',
    'dp_mode',
    'noise_multiplier',
    'clip_norm',
    'attack_noise_amount',
]:
    if args.get(metrics_key) is not None:
        cfg[metrics_key] = args[metrics_key]
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
for optional_key in [
    'ipfs_api_url',
    'ipfs_gateway_url',
    'ipfs_pin_artifacts',
    'blockchain_rpc_url',
    'blockchain_chain_id',
    'blockchain_private_key',
    'blockchain_account_index',
    'blockchain_contract_address',
    'blockchain_deploy_contract',
    'blockchain_wait_for_receipt',
    'blockchain_receipt_timeout_sec',
    'blockchain_gas_limit',
    'blockchain_gas_price_wei',
    'blockchain_store_full_report_json',
    'blockchain_report_payload_mode',
    'commitment_measure_real_overhead',
]:
    if args.get(optional_key) is not None:
        cfg[optional_key] = args[optional_key]
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
    cfg['metrics_dir'] = resolve_results_path(cfg.get('metrics_dir', '{results_dir}/metrics'))
    cfg['plot_ready_dir'] = resolve_results_path(cfg.get('plot_ready_dir', '{results_dir}/plot_ready'))
    ensure_dir(cfg['metrics_dir'])
    ensure_dir(cfg['plot_ready_dir'])
    if cfg.get('defense_mode') in (None, ''):
        if committee_enabled():
            cfg['defense_mode'] = 'committee'
        elif str(cfg.get('dp_mode', 'none')).lower() not in {'', 'none'}:
            cfg['defense_mode'] = str(cfg.get('dp_mode')).lower()
        else:
            cfg['defense_mode'] = 'none'


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
        'median_pearson': 0.0,
        'std_pearson': 0.0,
        'best_psnr': 0.0,
        'avg_psnr': 0.0,
        'median_psnr': 0.0,
        'std_psnr': 0.0,
        'num_recovered': 0,
        'num_attempted_reconstructions': 0,
        'recovered_threshold': 0.98,
    }


def merge_reconstruction_metrics(current_metrics, candidate_metrics):
    merged = empty_reconstruction_metrics()
    for key in merged:
        if key == 'recovered_threshold':
            merged[key] = candidate_metrics.get(key, current_metrics.get(key, 0.98))
        elif key == 'num_attempted_reconstructions':
            merged[key] = max(current_metrics.get(key, 0), candidate_metrics.get(key, 0))
        else:
            merged[key] = max(current_metrics.get(key, 0.0), candidate_metrics.get(key, 0.0))
    return merged


def tensor_l2_norm(value):
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(torch.linalg.vector_norm(value.detach().float()).item())
    total = 0.0
    if isinstance(value, dict):
        for tensor in value.values():
            if torch.is_tensor(tensor):
                total += float(torch.sum(tensor.detach().float() ** 2).item())
    return float(np.sqrt(total))


def state_delta_norm(current, previous):
    if current is None or previous is None:
        return 0.0
    total = 0.0
    for key, value in current.items():
        if key in previous and torch.is_tensor(value) and torch.is_tensor(previous[key]):
            diff = value.detach().float().cpu() - previous[key].detach().float().cpu()
            total += float(torch.sum(diff ** 2).item())
    return float(np.sqrt(total))


def metrics_reason_code(reason):
    mapping = {
        'ok': 0,
        'validation loss increased too much': 1,
        'validation accuracy dropped too much': 2,
        'candidate too similar to previous approved parent': 3,
        'candidate too different from previous approved parent': 4,
        'candidate has parameter drift but frozen verifier behavior': 5,
        'cra distribution consistency failure': 6,
        'artifact_verification_failed': 7,
        'cra parent commitment failure': 8,
    }
    return int(mapping.get(str(reason or 'unknown').strip().lower(), 255))


def metrics_failed_bitmask(reason, behavior_frozen=False):
    code = metrics_reason_code(reason)
    bit_by_code = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    bitmask = 0
    if code in bit_by_code:
        bitmask |= 1 << bit_by_code[code]
    if behavior_frozen:
        bitmask |= 1 << 4
    return int(bitmask)


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
        'run_id', 'experiment_id', 'experiment_method', 'seed', 'epoch', 'dataset', 'model_name',
        'control_name', 'global_accuracy', 'global_loss', 'local_accuracy_mean',
        'local_loss_mean', 'epoch_time_sec', 'train_time_sec', 'aggregation_time_sec',
        'committee_time_sec', 'dp_time_sec', 'test_time_sec', 'attack_enabled',
        'attack_round', 'committee_enabled', 'committee_approved_this_epoch',
        'candidate_rejected_this_epoch',
    ]
    append_csv_row(cfg['epoch_results_csv'], fieldnames, row)


def write_leakage_result(row):
    fieldnames = [
        'run_id', 'experiment_id', 'experiment_method', 'seed', 'dataset', 'model_name',
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
        'run_id', 'experiment_id', 'experiment_method', 'seed', 'dataset', 'model_name',
        'control_name', 'num_epochs', 'total_runtime_sec', 'mean_epoch_time_sec',
        'mean_train_time_sec', 'mean_aggregation_time_sec', 'mean_committee_time_sec',
        'mean_dp_time_sec', 'mean_test_time_sec',
        'commitment_backend', 'artifact_store_backend', 'ledger_backend',
        'mean_artifact_put_time_sec', 'mean_artifact_get_time_sec',
        'mean_artifact_verify_time_sec', 'mean_ledger_commit_genesis_time_sec',
        'mean_ledger_submit_candidate_time_sec', 'mean_ledger_submit_report_time_sec',
        'mean_ledger_finalize_time_sec', 'mean_blockchain_receipt_wait_time_sec',
        'total_blockchain_gas_used', 'mean_ipfs_add_time_sec', 'mean_ipfs_cat_time_sec',
        'mean_ipfs_pin_time_sec', 'relative_notes',
    ]
    append_csv_row(cfg['overhead_results_csv'], fieldnames, row)


def build_common_result_fields(seed):
    return {
        'run_id': cfg.get('experiment_id') or cfg.get('model_tag') or f'cra_seed_{seed}',
        'experiment_id': cfg['experiment_id'],
        'experiment_method': cfg['experiment_method'],
        'seed': seed,
        'dataset': cfg['data_name'],
        'model_name': cfg['model_name'],
        'control_name': cfg['control_name'],
    }


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def std_or_zero(values):
    return float(np.std(values)) if values else 0.0


def median_or_zero(values):
    return float(np.median(values)) if values else 0.0


def _metric_scalar(value):
    if torch.is_tensor(value):
        return '<tensor>'
    if isinstance(value, np.ndarray):
        return '<array>'
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_metric_scalar_container(value), sort_keys=True)
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _metric_scalar_container(value):
    if torch.is_tensor(value):
        return {'tensor_blocked': True, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {'array_blocked': True, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _metric_scalar_container(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_scalar_container(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_metrics_csv(path, fieldnames, row):
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: _metric_scalar(row.get(key, '')) for key in fieldnames})


def _ensure_metrics_csv_header(path, fieldnames):
    ensure_dir(os.path.dirname(path))
    if os.path.exists(path):
        return
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        csv.DictWriter(csv_file, fieldnames=fieldnames).writeheader()


class CraMetricsLogger:
    def __init__(self, run_id, seed):
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.metrics_dir = cfg['metrics_dir']
        self.plot_ready_dir = cfg['plot_ready_dir']
        ensure_dir(self.metrics_dir)
        ensure_dir(self.plot_ready_dir)
        self.events_path = os.path.join(self.metrics_dir, 'events.jsonl')
        self.rows = {
            'reconstruction_raw': [],
            'reconstruction_summary': [],
            'cohort_convergence_raw': [],
            'cohort_convergence_summary': [],
            'utility_raw': [],
            'utility_summary': [],
            'defense_events': [],
            'defense_summary': [],
            'noise_dp_raw': [],
            'noise_dp_summary': [],
        }
        self.schemas = {
            'reconstruction_raw': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'local_train_size', 'local_epochs', 'global_epochs', 'batch_size',
                'iid_setting', 'model_rates', 'target_cohort', 'epoch', 'user_id',
                'sample_id', 'label', 'pearson', 'psnr', 'recovered_bool',
                'reconstruction_rank', 'layer_id', 'node_id', 'row_id',
            ],
            'reconstruction_summary': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'local_train_size', 'local_epochs', 'global_epochs', 'batch_size',
                'iid_setting', 'model_rates', 'target_cohort', 'best_pearson',
                'avg_pearson', 'median_pearson', 'std_pearson', 'best_psnr',
                'avg_psnr', 'median_psnr', 'std_psnr', 'num_recovered',
                'num_attempted_reconstructions', 'recovered_threshold',
            ],
            'cohort_convergence_raw': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'local_train_size', 'local_epochs', 'global_epochs', 'batch_size',
                'iid_setting', 'model_rates', 'target_cohort', 'epoch', 'cohort_rate',
                'num_clients', 'train_loss', 'train_accuracy', 'test_loss',
                'test_accuracy', 'update_norm', 'parameter_delta_norm',
                'relative_update_norm', 'before_after_cra_extraction',
                'debug_target_update_norm', 'debug_non_target_update_norm',
                'debug_target_to_total_update_ratio', 'debug_target_to_non_target_ratio',
                'debug_aggregate_norm', 'debug_residual_non_target_norm',
            ],
            'cohort_convergence_summary': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'cohort_rate', 'mean_train_loss', 'mean_train_accuracy',
                'mean_test_loss', 'mean_test_accuracy', 'mean_update_norm',
                'mean_parameter_delta_norm', 'mean_relative_update_norm',
            ],
            'utility_raw': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'local_train_size', 'local_epochs', 'global_epochs', 'batch_size',
                'iid_setting', 'model_rates', 'target_cohort', 'epoch',
                'global_train_loss', 'global_train_accuracy', 'global_test_loss',
                'global_test_accuracy', 'best_test_accuracy_so_far',
                'final_test_accuracy', 'round_skipped', 'round_rejected',
            ],
            'utility_summary': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'final_test_accuracy', 'final_test_loss', 'best_test_accuracy',
                'num_epochs', 'num_rejected_rounds', 'num_skipped_rounds',
            ],
            'defense_events': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'epoch', 'candidate_accepted', 'attack_blocked_bool',
                'rejection_reason', 'reason_code', 'failed_checks_bitmask',
                'quorum_rule', 'num_approved', 'num_rejected', 'verifier_user_id',
                'verifier_cohort_rate', 'relative_change', 'prev_loss', 'cand_loss',
                'loss_delta', 'prev_accuracy', 'cand_accuracy', 'accuracy_delta',
                'behavior_frozen', 'candidate_parent_hash', 'previous_parent_hash',
            ],
            'defense_summary': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'blocked_rate', 'accepted_count', 'rejected_count',
                'most_common_rejection_reason', 'blocked_by_noise_level',
                'blocked_by_local_train_size',
            ],
            'noise_dp_raw': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'epoch', 'dp_mode', 'noise_multiplier', 'clip_norm',
                'clipping_enabled', 'number_clipped_updates', 'fraction_clipped',
                'update_norm_before_clip_mean', 'update_norm_after_clip_mean',
                'estimated_noise_std', 'noise_seed', 'attack_side_noise_amount',
                'attack_side_noise_std', 'bias_grad_abs_min',
                'bias_grad_abs_median', 'bias_grad_abs_mean', 'bias_grad_abs_max',
                'weight_grad_abs_mean', 'weight_grad_abs_median',
                'denominator_near_zero_count', 'denominator_near_zero_threshold',
            ],
            'noise_dp_summary': [
                'run_id', 'seed', 'dataset', 'model', 'attack_name', 'defense_mode',
                'dp_mode', 'noise_multiplier', 'clip_norm',
                'mean_fraction_clipped', 'mean_estimated_noise_std',
                'attack_side_noise_amount',
            ],
        }
        for name, schema in self.schemas.items():
            _ensure_metrics_csv_header(os.path.join(self.metrics_dir, f'{name}.csv'), schema)
        self.write_config()

    def common(self):
        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'dataset': cfg['data_name'],
            'model': cfg['model_name'],
            'attack_name': 'CRA',
            'defense_mode': cfg.get('defense_mode', 'none'),
            'local_train_size': cfg.get('local_train_size', ''),
            'local_epochs': cfg['num_epochs']['local'],
            'global_epochs': cfg['num_epochs']['global'],
            'batch_size': safe_cfg_get('batch_size', default={}).get('train', ''),
            'iid_setting': cfg.get('control', {}).get('data_split_mode', cfg.get('data_split_mode', '')),
            'model_rates': json.dumps(cfg.get('model_rate', [])),
            'target_cohort': 0.5,
        }

    def write_config(self):
        path = os.path.join(self.metrics_dir, 'resolved_config.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(_metric_scalar_container(dict(cfg)), handle, indent=2, sort_keys=True)

    def event(self, event_type, payload):
        with open(self.events_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(_metric_scalar_container({
                'run_id': self.run_id,
                'seed': self.seed,
                'event_type': event_type,
                'timestamp': time.time(),
                **dict(payload or {}),
            }), sort_keys=True) + '\n')

    def write(self, name, row):
        full_row = {**self.common(), **dict(row or {})}
        self.rows[name].append(full_row)
        _append_metrics_csv(os.path.join(self.metrics_dir, f'{name}.csv'), self.schemas[name], full_row)

    def write_many(self, name, rows):
        for row in rows or []:
            self.write(name, row)

    def finalize(self, epoch_rows, reconstruction_metrics):
        utility_rows = self.rows['utility_raw']
        best_acc = max([float(r.get('global_test_accuracy') or 0.0) for r in utility_rows], default=0.0)
        final_acc = float(utility_rows[-1].get('global_test_accuracy') or 0.0) if utility_rows else 0.0
        final_loss = float(utility_rows[-1].get('global_test_loss') or 0.0) if utility_rows else 0.0
        self.write('utility_summary', {
            'final_test_accuracy': final_acc,
            'final_test_loss': final_loss,
            'best_test_accuracy': best_acc,
            'num_epochs': len(utility_rows),
            'num_rejected_rounds': sum(1 for r in utility_rows if bool(r.get('round_rejected', False))),
            'num_skipped_rounds': sum(1 for r in utility_rows if bool(r.get('round_skipped', False))),
        })
        self.write('reconstruction_summary', reconstruction_metrics)
        self._finalize_defense_summary()
        self._finalize_noise_summary()
        self._finalize_cohort_summary()
        self._write_plot_ready()

    def _finalize_defense_summary(self):
        rows = self.rows['defense_events']
        if not rows:
            self.write('defense_summary', {'blocked_rate': 0.0, 'accepted_count': 0, 'rejected_count': 0})
            return
        round_rows = [r for r in rows if r.get('verifier_user_id') in ('', None)]
        if not round_rows:
            round_rows = rows
        denom = max(len(round_rows), 1)
        rejected = [r for r in round_rows if not bool(r.get('candidate_accepted', False))]
        accepted = [r for r in round_rows if bool(r.get('candidate_accepted', False))]
        reasons = [str(r.get('rejection_reason', '')) for r in rejected if r.get('rejection_reason')]
        self.write('defense_summary', {
            'blocked_rate': float(len(rejected) / denom),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'most_common_rejection_reason': Counter(reasons).most_common(1)[0][0] if reasons else '',
            'blocked_by_noise_level': cfg.get('attack_noise_amount', ''),
            'blocked_by_local_train_size': cfg.get('local_train_size', ''),
        })

    def _finalize_noise_summary(self):
        rows = self.rows['noise_dp_raw']
        self.write('noise_dp_summary', {
            'dp_mode': cfg.get('dp_mode', 'none'),
            'noise_multiplier': cfg.get('noise_multiplier', ''),
            'clip_norm': cfg.get('clip_norm', ''),
            'mean_fraction_clipped': mean_or_zero([float(r.get('fraction_clipped') or 0.0) for r in rows]),
            'mean_estimated_noise_std': mean_or_zero([float(r.get('estimated_noise_std') or 0.0) for r in rows]),
            'attack_side_noise_amount': cfg.get('attack_noise_amount', 0.0),
        })

    def _finalize_cohort_summary(self):
        rows = self.rows['cohort_convergence_raw']
        by_cohort = {}
        for row in rows:
            by_cohort.setdefault(row.get('cohort_rate', ''), []).append(row)
        if not by_cohort:
            self.write('cohort_convergence_summary', {
                'cohort_rate': '',
                'mean_train_loss': 0.0,
                'mean_train_accuracy': 0.0,
                'mean_test_loss': 0.0,
                'mean_test_accuracy': 0.0,
                'mean_update_norm': 0.0,
                'mean_parameter_delta_norm': 0.0,
                'mean_relative_update_norm': 0.0,
            })
            return
        for cohort, cohort_rows in by_cohort.items():
            self.write('cohort_convergence_summary', {
                'cohort_rate': cohort,
                'mean_train_loss': mean_or_zero([float(r.get('train_loss') or 0.0) for r in cohort_rows]),
                'mean_train_accuracy': mean_or_zero([float(r.get('train_accuracy') or 0.0) for r in cohort_rows]),
                'mean_test_loss': mean_or_zero([float(r.get('test_loss') or 0.0) for r in cohort_rows]),
                'mean_test_accuracy': mean_or_zero([float(r.get('test_accuracy') or 0.0) for r in cohort_rows]),
                'mean_update_norm': mean_or_zero([float(r.get('update_norm') or 0.0) for r in cohort_rows]),
                'mean_parameter_delta_norm': mean_or_zero([float(r.get('parameter_delta_norm') or 0.0) for r in cohort_rows]),
                'mean_relative_update_norm': mean_or_zero([float(r.get('relative_update_norm') or 0.0) for r in cohort_rows]),
            })

    def _copy_plot_ready(self, source_name, dest_name):
        src = os.path.join(self.metrics_dir, f'{source_name}.csv')
        dst = os.path.join(self.plot_ready_dir, dest_name)
        if os.path.exists(src):
            shutil.copy(src, dst)

    def _write_plot_ready(self):
        self._copy_plot_ready('reconstruction_summary', 'cra_privacy_by_local_train_size.csv')
        self._copy_plot_ready('reconstruction_summary', 'cra_privacy_by_noise.csv')
        self._copy_plot_ready('utility_raw', 'cra_utility_by_epoch.csv')
        self._copy_plot_ready('defense_summary', 'cra_defense_block_rate.csv')
        self._copy_plot_ready('noise_dp_summary', 'cra_dp_comparison.csv')
        overhead_src = cfg.get('overhead_results_csv')
        if overhead_src and os.path.exists(overhead_src):
            shutil.copy(overhead_src, os.path.join(self.plot_ready_dir, 'cra_overhead_summary.csv'))


def summarize_commitment_overhead(round_log_dir):
    path = os.path.join(round_log_dir, 'commitment_overhead.jsonl')
    summary = {
        'commitment_backend': cfg.get('commitment_backend', 'local'),
        'artifact_store_backend': cfg.get('artifact_store_backend', 'local'),
        'ledger_backend': cfg.get('ledger_backend', 'json'),
        'mean_artifact_put_time_sec': 0.0,
        'mean_artifact_get_time_sec': 0.0,
        'mean_artifact_verify_time_sec': 0.0,
        'mean_ledger_commit_genesis_time_sec': 0.0,
        'mean_ledger_submit_candidate_time_sec': 0.0,
        'mean_ledger_submit_report_time_sec': 0.0,
        'mean_ledger_finalize_time_sec': 0.0,
        'mean_blockchain_receipt_wait_time_sec': 0.0,
        'total_blockchain_gas_used': 0,
        'mean_ipfs_add_time_sec': 0.0,
        'mean_ipfs_cat_time_sec': 0.0,
        'mean_ipfs_pin_time_sec': 0.0,
    }
    if not os.path.exists(path):
        return summary
    event_durations = {
        'artifact_put': [],
        'artifact_get': [],
        'artifact_verify': [],
        'ledger_commit_genesis': [],
        'ledger_submit_candidate': [],
        'ledger_submit_report': [],
        'ledger_finalize': [],
    }
    metric_values = {'receipt_wait_time_sec': [], 'ipfs_add_time_sec': [], 'ipfs_cat_time_sec': [], 'ipfs_pin_time_sec': []}
    gas_values = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = payload.get('event')
            if event in event_durations:
                event_durations[event].append(float(payload.get('duration_sec', 0.0) or 0.0))
            for key in metric_values:
                if payload.get(key) not in (None, ''):
                    metric_values[key].append(float(payload.get(key) or 0.0))
            if payload.get('gas_used') not in (None, ''):
                gas_values.append(int(payload.get('gas_used') or 0))
    summary.update({
        'mean_artifact_put_time_sec': mean_or_zero(event_durations['artifact_put']),
        'mean_artifact_get_time_sec': mean_or_zero(event_durations['artifact_get']),
        'mean_artifact_verify_time_sec': mean_or_zero(event_durations['artifact_verify']),
        'mean_ledger_commit_genesis_time_sec': mean_or_zero(event_durations['ledger_commit_genesis']),
        'mean_ledger_submit_candidate_time_sec': mean_or_zero(event_durations['ledger_submit_candidate']),
        'mean_ledger_submit_report_time_sec': mean_or_zero(event_durations['ledger_submit_report']),
        'mean_ledger_finalize_time_sec': mean_or_zero(event_durations['ledger_finalize']),
        'mean_blockchain_receipt_wait_time_sec': mean_or_zero(metric_values['receipt_wait_time_sec']),
        'total_blockchain_gas_used': sum(gas_values),
        'mean_ipfs_add_time_sec': mean_or_zero(metric_values['ipfs_add_time_sec']),
        'mean_ipfs_cat_time_sec': mean_or_zero(metric_values['ipfs_cat_time_sec']),
        'mean_ipfs_pin_time_sec': mean_or_zero(metric_values['ipfs_pin_time_sec']),
    })
    return summary


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

    if commitment_backend not in {'local', 'blockchain'}:
        raise ValueError(f'Unsupported commitment_backend for CRA: {commitment_backend}')
    if artifact_backend not in {'local', 'ipfs'}:
        raise ValueError(f'Unsupported artifact_store_backend for CRA: {artifact_backend}')
    if ledger_backend not in {'json', 'ethereum'}:
        raise ValueError(f'Unsupported ledger_backend for CRA: {ledger_backend}')

    print('[CRA-COMMIT] selected commitment backend:', commitment_backend)
    print('[CRA-COMMIT] selected artifact backend:', artifact_backend)
    print('[CRA-COMMIT] selected ledger backend:', ledger_backend)
    if artifact_backend == 'ipfs':
        print('[CRA-COMMIT] IPFS API URL:', cfg.get('ipfs_api_url'))
    if ledger_backend == 'ethereum':
        print('[CRA-COMMIT] blockchain RPC URL:', cfg.get('blockchain_rpc_url'))
        print('[CRA-COMMIT] blockchain chain id:', cfg.get('blockchain_chain_id'))
        print('[CRA-COMMIT] blockchain contract address:', cfg.get('blockchain_contract_address') or '<deploy>')

    if artifact_backend == 'local':
        artifact_store = LocalArtifactStore(
            round_log_dir,
            artifact_dirname=cfg.get('commitment_artifact_dirname', 'artifacts'),
        )
    else:
        artifact_store = IPFSArtifactStore(
            ipfs_api_url=cfg.get('ipfs_api_url', 'http://127.0.0.1:5001'),
            ipfs_gateway_url=cfg.get('ipfs_gateway_url', 'http://127.0.0.1:8081/ipfs'),
            pin_artifacts=cfg.get('ipfs_pin_artifacts', True),
            request_timeout_sec=cfg.get('blockchain_receipt_timeout_sec', 120),
        )

    if ledger_backend == 'json':
        ledger = JsonCommitmentLedger(
            round_log_dir,
            ledger_filename=cfg.get('commitment_ledger_filename', 'commitment_ledger.json'),
        )
    else:
        ledger = EthereumCommitmentLedger(
            round_log_dir=round_log_dir,
            blockchain_rpc_url=cfg.get('blockchain_rpc_url', 'http://127.0.0.1:8545'),
            blockchain_chain_id=cfg.get('blockchain_chain_id', 1337),
            blockchain_private_key=cfg.get('blockchain_private_key'),
            blockchain_account_index=cfg.get('blockchain_account_index', 0),
            blockchain_contract_address=cfg.get('blockchain_contract_address'),
            blockchain_deploy_contract=cfg.get('blockchain_deploy_contract', True),
            blockchain_wait_for_receipt=cfg.get('blockchain_wait_for_receipt', True),
            blockchain_receipt_timeout_sec=cfg.get('blockchain_receipt_timeout_sec', 120),
            blockchain_gas_limit=cfg.get('blockchain_gas_limit', 8000000),
            blockchain_gas_price_wei=cfg.get('blockchain_gas_price_wei'),
            blockchain_store_full_report_json=cfg.get('blockchain_store_full_report_json', False),
            blockchain_report_payload_mode=cfg.get('blockchain_report_payload_mode', 'hash_only'),
        )
        print('[CRA-COMMIT] resolved contract address:', ledger.contract_address)
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
        'previous_parent_hash': previous_parent_record.parent_hash,
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
    run_id = cfg.get('experiment_id') or cfg.get('model_tag') or f'cra_seed_{seed}'
    metrics_logger = CraMetricsLogger(run_id=run_id, seed=seed)
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
        'metrics_logger': metrics_logger,
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
        metrics_logger.write('utility_raw', {
            'epoch': int(epoch),
            'global_train_loss': float(logger.mean.get('train/Local-Loss', 0.0)),
            'global_train_accuracy': float(logger.mean.get('train/Local-Accuracy', 0.0)),
            'global_test_loss': final_global_loss,
            'global_test_accuracy': final_global_accuracy,
            'best_test_accuracy_so_far': max(
                [float(row.get('global_accuracy', 0.0)) for row in experiment_tracker['epoch_rows']]
            ),
            'final_test_accuracy': final_global_accuracy,
            'round_skipped': bool(train_context.get('round_skipped', False)),
            'round_rejected': bool(train_context.get('candidate_rejected_this_epoch', False)),
        })
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
    commitment_overhead_summary = summarize_commitment_overhead(
        cfg.get('round_log_dir') or os.path.join('output', 'round_log', 'prototype2')
    )
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
        **commitment_overhead_summary,
        'relative_notes': 'convergence_mode={} disable_attack_for_convergence={}'.format(
            cfg['convergence_mode'], cfg['disable_attack_for_convergence']
        ),
    })
    metrics_logger.finalize(epoch_rows, reconstruction_metrics)
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
    metrics_logger = experiment_tracker.get('metrics_logger')
    previous_global_state = clone_state_dict(federation.global_parameters)
    if metrics_logger is not None:
        metrics_logger.write('noise_dp_raw', {
            'epoch': int(epoch),
            'dp_mode': cfg.get('dp_mode', 'none'),
            'noise_multiplier': cfg.get('noise_multiplier', ''),
            'clip_norm': cfg.get('clip_norm', ''),
            'clipping_enabled': cfg.get('clip_norm') not in (None, ''),
            'number_clipped_updates': '',
            'fraction_clipped': '',
            'estimated_noise_std': cfg.get('noise_multiplier', ''),
            'attack_side_noise_amount': cfg.get('attack_noise_amount', 0.0),
            'attack_side_noise_std': cfg.get('attack_noise_amount', 0.0),
        })

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
            'round_skipped': True,
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
            if metrics_logger is not None:
                metrics_logger.write('defense_events', {
                    'epoch': int(epoch),
                    'candidate_accepted': False,
                    'attack_blocked_bool': True,
                    'rejection_reason': 'cra distribution consistency failure',
                    'reason_code': metrics_reason_code('cra distribution consistency failure'),
                    'failed_checks_bitmask': metrics_failed_bitmask('cra distribution consistency failure'),
                    'num_approved': 0,
                    'num_rejected': 1,
                    'verifier_user_id': int(event['user_id']),
                    'verifier_cohort_rate': float(event['cohort']),
                    'relative_change': float(event['relative_diff']),
                    'candidate_parent_hash': '',
                    'previous_parent_hash': event['expected_parent_hash'],
                })
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
                'round_skipped': False,
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
    aggregate_delta_norm = state_delta_norm(honest_aggregated_state, previous_global_state)
    if metrics_logger is not None:
        cohort_counts = Counter(float(federation.model_rate[uid]) for uid in user_idx)
        for cohort_rate, num_clients in sorted(cohort_counts.items()):
            cohort_update_norms = []
            for m in range(num_active_users):
                if float(federation.model_rate[user_idx[m]]) == float(cohort_rate):
                    cohort_update_norms.append(state_delta_norm(local_parameters[m], distributed_local_parameters[m]))
            update_norm = mean_or_zero(cohort_update_norms)
            metrics_logger.write('cohort_convergence_raw', {
                'epoch': int(epoch),
                'cohort_rate': float(cohort_rate),
                'num_clients': int(num_clients),
                'train_loss': float(logger.mean.get('train/Local-Loss', 0.0)),
                'train_accuracy': float(logger.mean.get('train/Local-Accuracy', 0.0)),
                'update_norm': update_norm,
                'parameter_delta_norm': aggregate_delta_norm,
                'relative_update_norm': float(update_norm / (aggregate_delta_norm + 1.0e-12)),
                'before_after_cra_extraction': 'after' if attack_round else 'before',
                'debug_aggregate_norm': aggregate_delta_norm,
            })

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
        if metrics_logger is not None:
            metrics_logger.write('defense_events', {
                'epoch': int(epoch),
                'candidate_accepted': bool(commitment_event['approved']),
                'attack_blocked_bool': not bool(commitment_event['approved']),
                'rejection_reason': '' if commitment_event['approved'] else commitment_event.get('quorum_decision_reason', ''),
                'reason_code': metrics_reason_code('ok' if commitment_event['approved'] else 'cra parent commitment failure'),
                'failed_checks_bitmask': 0 if commitment_event['approved'] else metrics_failed_bitmask('cra parent commitment failure'),
                'quorum_rule': commitment_event.get('quorum_rule', ''),
                'num_approved': int(commitment_event.get('decision_num_approved', 0)),
                'num_rejected': int(commitment_event.get('decision_num_rejected', 0)),
                'candidate_parent_hash': commitment_event.get('model_hash', ''),
                'previous_parent_hash': commitment_event.get('previous_parent_hash', ''),
            })
            for verifier_report in commitment_event.get('verifier_reports', []):
                prev_eval = verifier_report.get('prev_eval', {})
                cand_eval = verifier_report.get('cand_eval', {})
                reason = verifier_report.get('reason', '')
                metrics_logger.write('defense_events', {
                    'epoch': int(epoch),
                    'candidate_accepted': bool(commitment_event['approved']),
                    'attack_blocked_bool': not bool(commitment_event['approved']),
                    'rejection_reason': reason,
                    'reason_code': metrics_reason_code(reason),
                    'failed_checks_bitmask': metrics_failed_bitmask(reason, verifier_report.get('behavior_frozen', False)),
                    'quorum_rule': commitment_event.get('quorum_rule', ''),
                    'num_approved': int(commitment_event.get('decision_num_approved', 0)),
                    'num_rejected': int(commitment_event.get('decision_num_rejected', 0)),
                    'verifier_user_id': int(verifier_report.get('user_id', -1)),
                    'verifier_cohort_rate': float(verifier_report.get('cohort_rate', 0.0)),
                    'relative_change': float(verifier_report.get('relative_change', 0.0)),
                    'prev_loss': float(prev_eval.get('Local-Loss', 0.0)),
                    'cand_loss': float(cand_eval.get('Local-Loss', 0.0)),
                    'loss_delta': float(cand_eval.get('Local-Loss', 0.0)) - float(prev_eval.get('Local-Loss', 0.0)),
                    'prev_accuracy': float(prev_eval.get('Local-Accuracy', 0.0)),
                    'cand_accuracy': float(cand_eval.get('Local-Accuracy', 0.0)),
                    'accuracy_delta': float(cand_eval.get('Local-Accuracy', 0.0)) - float(prev_eval.get('Local-Accuracy', 0.0)),
                    'behavior_frozen': bool(verifier_report.get('behavior_frozen', False)),
                    'candidate_parent_hash': commitment_event.get('model_hash', ''),
                    'previous_parent_hash': commitment_event.get('previous_parent_hash', ''),
                })
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
                    if metrics_logger is not None:
                        for raw_row in user_metrics.get('_raw_rows', []):
                            metrics_logger.write('reconstruction_raw', {
                                'epoch': int(epoch),
                                'user_id': int(user_idx[m]),
                                **raw_row,
                            })
                        diagnostics = user_metrics.get('_noise_dp_diagnostics', {})
                        metrics_logger.write('noise_dp_raw', {
                            'epoch': int(epoch),
                            'dp_mode': cfg.get('dp_mode', 'none'),
                            'noise_multiplier': cfg.get('noise_multiplier', ''),
                            'clip_norm': cfg.get('clip_norm', ''),
                            'clipping_enabled': cfg.get('clip_norm') not in (None, ''),
                            'number_clipped_updates': '',
                            'fraction_clipped': '',
                            'estimated_noise_std': cfg.get('noise_multiplier', ''),
                            'attack_side_noise_amount': cfg.get('attack_noise_amount', 0.0),
                            'attack_side_noise_std': cfg.get('attack_noise_amount', 0.0),
                            **diagnostics,
                        })
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
        'round_skipped': False,
    }


def reconstruct_image(weight_grad, bias_grad, img_list):
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0
    threshold = float(safe_cfg_get('recovered_pearson_threshold', default=0.98))
    raw_rows = []
    eps = 1.0e-8
    bias_abs = torch.abs(bias_grad.detach().float()).flatten()
    weight_abs = torch.abs(weight_grad.detach().float()).flatten()
    diagnostics = {
        'bias_grad_abs_min': float(torch.min(bias_abs).item()) if bias_abs.numel() else 0.0,
        'bias_grad_abs_median': float(torch.median(bias_abs).item()) if bias_abs.numel() else 0.0,
        'bias_grad_abs_mean': float(torch.mean(bias_abs).item()) if bias_abs.numel() else 0.0,
        'bias_grad_abs_max': float(torch.max(bias_abs).item()) if bias_abs.numel() else 0.0,
        'weight_grad_abs_mean': float(torch.mean(weight_abs).item()) if weight_abs.numel() else 0.0,
        'weight_grad_abs_median': float(torch.median(weight_abs).item()) if weight_abs.numel() else 0.0,
        'denominator_near_zero_count': int(torch.sum(bias_abs < eps).item()) if bias_abs.numel() else 0,
        'denominator_near_zero_threshold': eps,
    }

    for sample_rank, elem in enumerate(img_list):
        elem_extracted = elem[0].to(cfg["device"])
        max_pearson = 0.0
        max_ssim = 0.0
        max_psnr = 0.0
        pearson = PearsonCorrCoef().to(cfg["device"])
        best_partial_recon = None
        best_node_id = ''

        for i in range(weight_grad.size()[0]):
            denom = bias_grad[i]
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
                best_node_id = int(i)

        if max_pearson >= threshold:
            num_recovered += 1
        raw_rows.append({
            'sample_id': sample_rank,
            'label': '',
            'pearson': max_pearson,
            'psnr': max_psnr,
            'recovered_bool': bool(max_pearson >= threshold),
            'reconstruction_rank': sample_rank,
            'layer_id': 'layers.0',
            'node_id': best_node_id,
            'row_id': best_node_id,
        })
        avg_ssim.append(max_ssim)
        avg_psnr.append(max_psnr)
        avg_pearson.append(max_pearson)
        if best_partial_recon is None:
            continue
        _ = best_partial_recon.reshape(elem_extracted.size())

    if len(avg_pearson) == 0:
        metrics = empty_reconstruction_metrics()
        metrics['_raw_rows'] = raw_rows
        metrics['_noise_dp_diagnostics'] = diagnostics
        return metrics

    return {
        'best_pearson': max(avg_pearson),
        'avg_pearson': sum(avg_pearson) / len(avg_pearson),
        'median_pearson': median_or_zero(avg_pearson),
        'std_pearson': std_or_zero(avg_pearson),
        'best_psnr': max(avg_psnr),
        'avg_psnr': sum(avg_psnr) / len(avg_psnr),
        'median_psnr': median_or_zero(avg_psnr),
        'std_psnr': std_or_zero(avg_psnr),
        'num_recovered': num_recovered,
        'num_attempted_reconstructions': len(avg_pearson),
        'recovered_threshold': threshold,
        '_raw_rows': raw_rows,
        '_noise_dp_diagnostics': diagnostics,
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
