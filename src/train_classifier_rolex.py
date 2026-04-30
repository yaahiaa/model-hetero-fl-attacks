import argparse
import csv
import copy
import datetime
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
from local_dp import apply_local_dp_to_update
from distributed_dp import apply_distributed_dp_to_update

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
RAW_DP_MODE_PRESENT = 'dp_mode' in cfg


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


def add_arg_if_missing(*args, **kwargs):
    option = args[0]
    existing_options = {
        opt
        for action in parser._actions
        for opt in action.option_strings
    }
    if option not in existing_options:
        parser.add_argument(*args, **kwargs)


add_arg_if_missing('--control_name', default=None, type=str)
add_arg_if_missing('--seed', default=None, type=int)
add_arg_if_missing('--global_epochs', default=None, type=int)
add_arg_if_missing('--local_epochs', default=None, type=int)
add_arg_if_missing('--local_train_size', default=None, type=int)
add_arg_if_missing('--train_batch_size', default=None, type=int)
add_arg_if_missing('--test_batch_size', default=None, type=int)

add_arg_if_missing('--experiment_method', default=None, type=str)
add_arg_if_missing('--experiment_id', default=None, type=str)
add_arg_if_missing('--results_dir', default=None, type=str)
add_arg_if_missing('--leakage_results_csv', default=None, type=str)
add_arg_if_missing('--epoch_results_csv', default=None, type=str)
add_arg_if_missing('--overhead_results_csv', default=None, type=str)
add_arg_if_missing('--enable_experiment_logging', default=None, type=str_to_bool)
add_arg_if_missing('--attack_noise_amount', default=None, type=float)
add_arg_if_missing('--noise_scale', default=None, type=float)
add_arg_if_missing('--attack_blocked_zero_metrics', default=None, type=str_to_bool)
add_arg_if_missing('--recovered_pearson_threshold', default=None, type=float)
add_arg_if_missing('--convergence_mode', default=None, type=str_to_bool)
add_arg_if_missing('--disable_attack_for_convergence', default=None, type=str_to_bool)
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

# Prototype4: base RMA with configurable DP defense
cfg.setdefault('attack_source_round', 3)
cfg.setdefault('attack_replay_round', 4)
cfg.setdefault('attack_replay_enabled', True)
cfg.setdefault('attack_model_mode', 'real_replay')
cfg.setdefault('debug_attack_replay', False)
cfg.setdefault('debug_hybrid_trap', False)
# DP defense mode
cfg.setdefault('dp_mode', 'none')
# Local DP defense
cfg.setdefault('local_dp_enabled', False)
cfg.setdefault('ldp_clip_norm', 1.0)
cfg.setdefault('ldp_noise_multiplier', 0.005)
cfg.setdefault('debug_local_dp', False)
# Distributed DP defense
cfg.setdefault('ddp_enabled', False)
cfg.setdefault('ddp_clip_norm', 1.0)
cfg.setdefault('ddp_noise_multiplier', 0.005)
cfg.setdefault('ddp_debug', False)
cfg.setdefault('ddp_use_shared_total_noise', False)
cfg.setdefault('experiment_method', 'unknown')
cfg.setdefault('experiment_id', 'default')
cfg.setdefault('results_dir', 'results')
cfg.setdefault('leakage_results_csv', '{results_dir}/leakage_raw.csv')
cfg.setdefault('epoch_results_csv', '{results_dir}/epoch_raw.csv')
cfg.setdefault('overhead_results_csv', '{results_dir}/overhead_raw.csv')
cfg.setdefault('enable_experiment_logging', True)
cfg.setdefault('attack_noise_amount', 0.0)
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
    resolved = str(path_value).format(results_dir=cfg['results_dir'])
    ensure_dir(os.path.dirname(resolved))
    return resolved


def now_seconds():
    return time.perf_counter()


def infer_experiment_method():
    explicit = str(safe_cfg_get('experiment_method', default='unknown')).lower()
    if explicit not in {'', 'unknown'}:
        return explicit
    dp_mode = get_dp_mode()
    if dp_mode == 'distributed':
        return 'ddp'
    if dp_mode == 'local':
        return 'ldp'
    return 'base'


def attack_execution_enabled():
    return bool(cfg.get('attack_replay_enabled', True)) and not (
        bool(cfg.get('convergence_mode', False))
        and bool(cfg.get('disable_attack_for_convergence', False))
    )


def get_dp_report_fields():
    dp_mode = get_dp_mode()
    if dp_mode == 'distributed':
        return (
            'distributed',
            safe_cfg_get('ddp_clip_norm', default=''),
            safe_cfg_get('ddp_noise_multiplier', default=''),
        )
    if dp_mode == 'local':
        return (
            'local',
            safe_cfg_get('ldp_clip_norm', default=''),
            safe_cfg_get('ldp_noise_multiplier', default=''),
        )
    return ('none', '', '')


def resolve_experiment_defaults():
    cfg['experiment_method'] = infer_experiment_method()
    if cfg.get('experiment_id') in (None, '', 'default'):
        cfg['experiment_id'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args['attack_noise_amount'] is None:
        cfg['attack_noise_amount'] = float(safe_cfg_get('noise_scale', default=0.0) or 0.0)
    if cfg.get('noise_scale') is None:
        cfg['noise_scale'] = float(safe_cfg_get('attack_noise_amount', default=0.0) or 0.0)
    cfg['leakage_results_csv'] = resolve_results_path(cfg['leakage_results_csv'])
    cfg['epoch_results_csv'] = resolve_results_path(cfg['epoch_results_csv'])
    cfg['overhead_results_csv'] = resolve_results_path(cfg['overhead_results_csv'])


def append_csv_row(path, fieldnames, row):
    if not bool(cfg.get('enable_experiment_logging', True)):
        return
    ensure_dir(os.path.dirname(path))
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


def apply_runtime_overrides():
    if args['global_epochs'] is not None:
        cfg['num_epochs']['global'] = int(args['global_epochs'])
    if args['local_epochs'] is not None:
        cfg['num_epochs']['local'] = int(args['local_epochs'])
    if args['train_batch_size'] is not None:
        cfg['batch_size']['train'] = int(args['train_batch_size'])
    if args['test_batch_size'] is not None:
        cfg['batch_size']['test'] = int(args['test_batch_size'])

full_path = os.path.join(os.getcwd(), cfg['file_output'])
ensure_dir(os.path.dirname(full_path))
fp = open(full_path, 'w')
fp.write("N Max_Pearson Avg_Pearson Max_PSNR Avg_PSNR Max_Recovered\n")

def clone_state_dict(state_dict):
    cloned = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned

def add_uniform_noise_to_state_dict(state_dict, noise_scale):
    """
    Add prototype3-h style attack noise to a parent/global state dict.

    For each floating tensor v:
        noise ~ Uniform(-noise_scale * mean(abs(v)),
                         noise_scale * mean(abs(v)))

    Non-floating tensors are left unchanged.
    """
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

def get_dp_mode():
    mode = str(cfg.get('dp_mode', 'none')).lower()
    if mode == 'distributed':
        return 'distributed'
    if mode == 'local':
        return 'local'
    if mode == 'none':
        if RAW_DP_MODE_PRESENT:
            return 'none'
        if bool(cfg.get('ddp_enabled', False)):
            return 'distributed'
        if bool(cfg.get('local_dp_enabled', False)):
            return 'local'
        return 'none'
    if bool(cfg.get('ddp_enabled', False)):
        return 'distributed'
    if bool(cfg.get('local_dp_enabled', False)):
        return 'local'
    raise ValueError(f'Invalid dp_mode: {cfg.get("dp_mode")}')


def get_fcnn_shifted_block_contributor_counts(federation, user_idx, target_rate=0.25):
    active_rates = [float(federation.model_rate[u]) for u in user_idx]

    # Source round: shifted block is updated by cohorts larger than target.
    # Replay round: shifted block is updated by larger cohorts + target cohort.
    num_source_contributors = sum(1 for r in active_rates if r > target_rate)
    num_replay_contributors = sum(1 for r in active_rates if r >= target_rate)

    return num_source_contributors, num_replay_contributors


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
   
    if source_honest_parent is None or replay_result_parent is None:
        return None

    hidden_size = source_honest_parent.size(0)
    client_cap = int(float(model_rate) * hidden_size)

    lower = client_cap
    upper = lower + client_cap

    source_block = source_honest_parent[lower:upper].to(cfg['device'])
    replay_block = replay_result_parent[lower:upper].to(cfg['device'])

    # Undo averaging to isolate the target cohort's trained local model.
    target_trained_model = (
        replay_block * float(num_replay_contributors)
        - source_block * float(num_source_contributors)
    )

    # The inversion code expects "distributed_model - trained_model".
    return distributed_model.to(cfg['device']) - target_trained_model

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

    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        epoch_start = now_seconds()
        logger.safe(True)
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
            experiment_tracker,
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
        experiment_tracker['final_global_accuracy'] = float(logger.mean.get('test/Global-Accuracy', 0.0))
        experiment_tracker['final_global_loss'] = float(logger.mean.get('test/Global-Loss', 0.0))
        epoch_row = {
            **build_common_result_fields(seed),
            'epoch': int(epoch),
            'global_accuracy': experiment_tracker['final_global_accuracy'],
            'global_loss': experiment_tracker['final_global_loss'],
            'local_accuracy_mean': float(logger.mean.get('train/Local-Accuracy', 0.0)),
            'local_loss_mean': float(logger.mean.get('train/Local-Loss', 0.0)),
            'epoch_time_sec': epoch_time_sec,
            'train_time_sec': float(train_context.get('train_time_sec', 0.0)),
            'aggregation_time_sec': float(train_context.get('aggregation_time_sec', 0.0)),
            'committee_time_sec': 0.0,
            'dp_time_sec': float(train_context.get('dp_time_sec', 0.0)),
            'test_time_sec': test_time_sec,
            'attack_enabled': bool(train_context.get('attack_enabled', False)),
            'attack_round': bool(train_context.get('attack_round', False)),
            'committee_enabled': False,
            'committee_approved_this_epoch': '',
            'candidate_rejected_this_epoch': False,
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
    dp_mode, dp_clip_norm, dp_noise_multiplier = get_dp_report_fields()
    write_leakage_result({
        **build_common_result_fields(seed),
        'global_epochs': cfg['num_epochs']['global'],
        'local_epochs': cfg['num_epochs']['local'],
        'local_train_size': cfg['local_train_size'],
        'batch_size_train': safe_cfg_get('batch_size', default={}).get('train', ''),
        'attack_source_round': safe_cfg_get('attack_source_round', default=''),
        'attack_replay_round': safe_cfg_get('attack_replay_round', default=''),
        'attack_noise_amount': cfg['attack_noise_amount'],
        'noise_scale': safe_cfg_get('noise_scale', default=''),
        'dp_mode': dp_mode,
        'dp_clip_norm': dp_clip_norm,
        'dp_noise_multiplier': dp_noise_multiplier,
        'committee_enabled': False,
        'committee_approved': '',
        'attack_blocked': experiment_tracker.get('attack_blocked', False),
        'commit_rejection_response': 'none',
        'best_pearson': experiment_tracker['last_reconstruction']['best_pearson'],
        'avg_pearson': experiment_tracker['last_reconstruction']['avg_pearson'],
        'best_psnr': experiment_tracker['last_reconstruction']['best_psnr'],
        'avg_psnr': experiment_tracker['last_reconstruction']['avg_psnr'],
        'num_recovered': experiment_tracker['last_reconstruction']['num_recovered'],
        'total_runtime_sec': total_runtime_sec,
        'final_global_accuracy': experiment_tracker['final_global_accuracy'],
        'final_global_loss': experiment_tracker['final_global_loss'],
    })
    write_overhead_summary({
        **build_common_result_fields(seed),
        'num_epochs': cfg['num_epochs']['global'],
        'total_runtime_sec': total_runtime_sec,
        'mean_epoch_time_sec': mean_or_zero([float(row['epoch_time_sec']) for row in epoch_rows]),
        'mean_train_time_sec': mean_or_zero([float(row['train_time_sec']) for row in epoch_rows]),
        'mean_aggregation_time_sec': mean_or_zero([float(row['aggregation_time_sec']) for row in epoch_rows]),
        'mean_committee_time_sec': 0.0,
        'mean_dp_time_sec': mean_or_zero([float(row['dp_time_sec']) for row in epoch_rows]),
        'mean_test_time_sec': mean_or_zero([float(row['test_time_sec']) for row in epoch_rows]),
        'relative_notes': 'dp_mode={} convergence_mode={} disable_attack_for_convergence={}'.format(
            dp_mode, cfg['convergence_mode'], cfg['disable_attack_for_convergence']
        ),
    })
    logger.safe(False)
    return


def train(model_history_block2, model_history_fcnn, fcnn_attack_cache, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, experiment_tracker):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    distributed_local_parameters = copy.deepcopy(local_parameters)
    img_list_by_user = {}
    num_active_users = len(local)
    dp_mode = get_dp_mode()
    attack_enabled = attack_execution_enabled()
    attack_round = False
    train_time_sec = 0.0
    dp_time_sec = 0.0
    aggregation_time_sec = 0.0

    lr = optimizer.param_groups[0]['lr']

    start_time = time.time()

    logger.append({
        'info': [
            f'[DP] epoch={epoch}',
            f'[DP] mode={dp_mode}',
            f'[DP] num_active_users={num_active_users}',
        ]
    }, 'train', mean=False)

    if dp_mode == 'distributed' and cfg.get('ddp_debug', False):
        sigma_total = float(cfg.get('ddp_noise_multiplier', 0.005)) * float(cfg.get('ddp_clip_norm', 1.0))
        sigma_client = np.sqrt(num_active_users) * sigma_total
        logger.append({
            'info': [
                f'[DDP] epoch={epoch}',
                f'[DDP] expected aggregate noise std = {sigma_total:.6e} because each of {num_active_users} clients adds sigma_client={sigma_client:.6e} and server averages {num_active_users} clients',
                '[DDP] this is a simulation of distributed client-contributed Gaussian noise, not a secure distributed-DP protocol',
            ]
        }, 'train', mean=False)
        if bool(cfg.get('ddp_use_shared_total_noise', False)):
            logger.append({
                'info': [
                    f'[DDP] epoch={epoch}',
                    '[DDP] ddp_use_shared_total_noise is a reserved compatibility flag in this simulation; sigma_client still uses sqrt(k) * sigma_total',
                ]
            }, 'train', mean=False)

    img_list = None
    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]] # This line modifies the learning rate based on user. 
        local_train_start = now_seconds()
        (trained_parameters, img_data) = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        train_time_sec += now_seconds() - local_train_start

        img_list = copy.deepcopy(img_data)
        img_list_by_user[int(user_idx[m])] = copy.deepcopy(img_data)

        if dp_mode == 'distributed':
            dp_start = now_seconds()
            privatized_parameters, ddp_report = apply_distributed_dp_to_update(
                base_parameters=distributed_local_parameters[m],
                trained_parameters=trained_parameters,
                clip_norm=float(cfg.get('ddp_clip_norm', 1.0)),
                noise_multiplier=float(cfg.get('ddp_noise_multiplier', 0.005)),
                num_active_users=num_active_users,
                enabled=True,
            )
            dp_time_sec += now_seconds() - dp_start
            local_parameters[m] = privatized_parameters

            if cfg.get('ddp_debug', False):
                logger.append({
                    'info': [
                        f'[DDP] epoch={epoch}',
                        f'[DDP] user_id={user_idx[m]}',
                        f'[DDP] model_rate={federation.model_rate[user_idx[m]]}',
                        f'[DDP] num_active_users={ddp_report["ddp_num_active_users"]}',
                        f'[DDP] update_norm_before_clip={ddp_report["ddp_update_norm"]:.6e}',
                        f'[DDP] clip_factor={ddp_report["ddp_clip_factor"]:.6e}',
                        f'[DDP] clip_norm={ddp_report["ddp_clip_norm"]:.6e}',
                        f'[DDP] noise_multiplier={ddp_report["ddp_noise_multiplier"]:.6e}',
                        f'[DDP] sigma_total={ddp_report["ddp_sigma_total"]:.6e}',
                        f'[DDP] sigma_client={ddp_report["ddp_sigma_client"]:.6e}',
                    ]
                }, 'train', mean=False)

        elif dp_mode == 'local':
            dp_start = now_seconds()
            privatized_parameters, ldp_report = apply_local_dp_to_update(
                base_parameters=distributed_local_parameters[m],
                trained_parameters=trained_parameters,
                clip_norm=float(cfg.get('ldp_clip_norm', 1.0)),
                noise_multiplier=float(cfg.get('ldp_noise_multiplier', 0.005)),
                enabled=True,
            )
            dp_time_sec += now_seconds() - dp_start
            local_parameters[m] = privatized_parameters

            if cfg.get('debug_local_dp', False):
                logger.append({
                    'info': [
                        f'[LDP] epoch={epoch}',
                        f'[LDP] user_id={user_idx[m]}',
                        f'[LDP] model_rate={federation.model_rate[user_idx[m]]}',
                        f'[LDP] update_norm={ldp_report["ldp_update_norm"]:.6e}',
                        f'[LDP] clip_factor={ldp_report["ldp_clip_factor"]:.6e}',
                        f'[LDP] clip_norm={ldp_report["ldp_clip_norm"]:.6e}',
                        f'[LDP] noise_multiplier={ldp_report["ldp_noise_multiplier"]:.6e}',
                        f'[LDP] noise_std={ldp_report["ldp_noise_std"]:.6e}',
                    ]
                }, 'train', mean=False)
        else:
            local_parameters[m] = trained_parameters

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
                            'DP Mode: {}'.format(dp_mode),
                            'Epoch Finished Time: {}'.format(epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])   

    aggregation_start = now_seconds()
    federation.combine(local_parameters, param_idx, user_idx)
    aggregation_time_sec = now_seconds() - aggregation_start

    honest_aggregated_state = clone_state_dict(federation.global_parameters)

    attack_source_round = int(cfg.get('attack_source_round', 3))
    attack_replay_round = int(cfg.get('attack_replay_round', 4))

    if cfg['model_name'] == 'fcnn' and epoch == attack_source_round:
        for key in ['layers.0.weight', 'layers.0.bias']:
            fcnn_attack_cache['source_honest_parent'][key] = honest_aggregated_state[key].detach().clone()

        if attack_enabled:
            # This is the malicious server action:
            # instead of using the honest aggregate from the source round as parent for
            # the replay round, replay the real parent that was used at the start of
            # the source round. If configured, apply the same style
            # uniform attack noise to the replayed parent before distributing it.
            attack_noise_scale = float(
                cfg.get(
                    'attack_noise_amount',
                    cfg.get('noise_scale', 0.0),
                ) or 0.0
            )

            if attack_noise_scale <= 0.0:
                attack_noise_scale = float(cfg.get('noise_scale', 0.0) or 0.0)

            federation.global_parameters, replay_noise_report = add_uniform_noise_to_state_dict(
                federation.initial_parent_state,
                attack_noise_scale,
            )

            attack_round = True

            if cfg.get('debug_attack_replay', False):
                print(
                    f'[REPLAY] epoch={epoch}: stored honest source aggregate, '
                    f'but replayed previous real parent for epoch {epoch + 1} '
                    f'with attack_noise_scale={attack_noise_scale} '
                    f'noise_enabled={replay_noise_report["noise_enabled"]} '
                    f'noise_rel_l2={replay_noise_report["noise_rel_l2"]:.6e} '
                    f'noise_max_abs={replay_noise_report["noise_max_abs"]:.6e} '
                    f'noise_mean_abs={replay_noise_report["noise_mean_abs"]:.6e}',
                    flush=True,
                )
        else:
            federation.global_parameters = honest_aggregated_state
            if bool(cfg.get('convergence_mode', False)) and bool(cfg.get('disable_attack_for_convergence', False)):
                experiment_tracker['attack_blocked'] = True

    else:
        federation.global_parameters = honest_aggregated_state

    global_model.load_state_dict(federation.global_parameters)
    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())

    if cfg['model_name'] == 'fcnn' and epoch == attack_replay_round:
        for key in ['layers.0.weight', 'layers.0.bias']:
            fcnn_attack_cache['replay_result_parent'][key] = honest_aggregated_state[key].detach().clone()


    # Append to the model history. 
    if cfg['model_name'] == 'fcnn':
        targetWeights = ['layers.0.weight']
        targetBiases = ['layers.0.bias']

        attack_replay_round = int(cfg.get('attack_replay_round', 4))

        for m in range(num_active_users):
            if federation.model_rate[user_idx[m]] != 0.25:
                continue

            weight_grad = None
            bias_grad = None
            max_pearson_list = []
            avg_pearson_list = []
            max_psnr_list = []
            avg_psnr_list = []
            num_recovered_list = []

            if epoch != attack_replay_round or not attack_enabled:
                continue

            for k in targetWeights + targetBiases:
                distributed_model = distributed_local_parameters[m][k]

                num_source_contributors, num_replay_contributors = (
                    get_fcnn_shifted_block_contributor_counts(
                        federation,
                        user_idx,
                        target_rate=0.25,
                    )
                )

                recovered = fcnn_leakage_from_cache(
                    key=k,
                    model_rate=federation.model_rate[user_idx[m]],
                    distributed_model=distributed_model,
                    source_honest_parent=fcnn_attack_cache['source_honest_parent'].get(k, None),
                    replay_result_parent=fcnn_attack_cache['replay_result_parent'].get(k, None),
                    num_source_contributors=num_source_contributors,
                    num_replay_contributors=num_replay_contributors,
                )

                if cfg.get('debug_attack_replay', False):
                    print(
                        f'[REPLAY][LEAKAGE] epoch={epoch} user={user_idx[m]} key={k} '
                        f'src_count={num_source_contributors} '
                        f'replay_count={num_replay_contributors} '
                        f'have_source={fcnn_attack_cache["source_honest_parent"].get(k, None) is not None} '
                        f'have_replay={fcnn_attack_cache["replay_result_parent"].get(k, None) is not None}',
                        flush=True,
                    )

                if recovered is None:
                    continue

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

                target_img_list = img_list_by_user.get(int(user_idx[m]), None)

                if target_img_list is not None and len(target_img_list) > 0:
                    reconstruction_metrics = reconstruct_image(
                        weight_grad,
                        bias_grad,
                        target_img_list,
                    )
                    experiment_tracker['last_reconstruction'] = merge_reconstruction_metrics(
                        experiment_tracker.get('last_reconstruction', empty_reconstruction_metrics()),
                        reconstruction_metrics,
                    )
                    max_pearson_list.append(reconstruction_metrics['best_pearson'])
                    avg_pearson_list.append(reconstruction_metrics['avg_pearson'])
                    max_psnr_list.append(reconstruction_metrics['best_psnr'])
                    avg_psnr_list.append(reconstruction_metrics['avg_psnr'])
                    num_recovered_list.append(reconstruction_metrics['num_recovered'])
                else:
                    print(
                        f'RECONSTRUCT: skipped because no image batch was recorded '
                        f'for target user {user_idx[m]}',
                        flush=True,
                    )

            if len(max_pearson_list) > 0:
                fp.write(
                    "%s %s %s %s %s %s\n" % (
                        cfg['local_train_size'],
                        max(max_pearson_list),
                        mean_or_zero(avg_pearson_list),
                        max(max_psnr_list),
                        mean_or_zero(avg_psnr_list),
                        max(num_recovered_list),
                    )
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
        'dp_time_sec': dp_time_sec,
        'attack_enabled': attack_enabled,
        'attack_round': attack_round,
    }

def reconstruct_image(weight_grad, bias_grad, img_list):
    count = 0
    avg_psnr = []
    avg_ssim = []
    avg_pearson = []
    num_recovered = 0
    recovered_threshold = float(cfg.get('recovered_pearson_threshold', 0.98))

    if weight_grad is None or bias_grad is None or img_list is None:
        return empty_reconstruction_metrics()

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

            partial_recon = torch.divide(weight_grad[i], denom)

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

        if best_partial_recon is None:
            continue

        if max_pearson >= recovered_threshold:
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

    if len(avg_pearson) == 0:
        print("RECONSTRUCT: no valid partial reconstructions", flush=True)
        return empty_reconstruction_metrics()
    print("RECONSTRUCT: best_ssim across images = %s"%(max(avg_ssim)))
    print("RECONSTRUCT: best_psnr across images = %s"%(max(avg_psnr)))
    print("RECONSTRUCT: best_pearson across images = %s"%(max(avg_pearson)))
    print("RECONSTRUCT: num_recovered = %s"%(num_recovered))

    print("RECONSTRUCT: avg_ssim across images = %s"%(sum(avg_ssim) / len(avg_ssim)))
    print("RECONSTRUCT: avg_psnr across images = %s"%(sum(avg_psnr) / len(avg_psnr)))
    print("RECONSTRUCT: avg_pearson across images = %s"%(sum(avg_pearson) / len(avg_pearson)))

    # plt.show() 
    # plt.savefig('orig_recon_rolex_n%s_v2_noise.png'%(cfg['local_train_size']))

    return {
        'best_pearson': float(max(avg_pearson)),
        'avg_pearson': float(sum(avg_pearson) / len(avg_pearson)),
        'best_psnr': float(max(avg_psnr)),
        'avg_psnr': float(sum(avg_psnr) / len(avg_psnr)),
        'num_recovered': int(num_recovered),
    }



def fcnn_leakage(k, user, num_active_users, model_rate, local_params, model_history, distributed_model):
    hidden_layer_size = model_history[0].size()[0]
    client_cap = int(model_rate * hidden_layer_size)
    lower = client_cap
    upper = lower + client_cap
    
    agg_val_rd_0 = torch.multiply(model_history[0][lower:upper], num_active_users - 1) # A + B
    agg_val_rd_1 = torch.multiply(model_history[1][lower:upper], num_active_users) # A + B + C

    agg_diff = torch.subtract(agg_val_rd_1, agg_val_rd_0)
    # print("FCNN_LEAKAGE: agg_diff = %s and local_params = %s"%(agg_diff, local_params))
    server_error = torch.abs(torch.subtract(agg_diff, local_params))
    # print("FCNN_LEAKAGE: server_error = %s"%(server_error))

    malicious_model_sent = distributed_model.to(cfg["device"])
    user_grad = torch.subtract(malicious_model_sent, agg_diff)
    # print("FCNN_LEAKAGE: user_grad = %s"%(user_grad))
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


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]

    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split 

    def train(self, local_parameters, lr, logger):
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
