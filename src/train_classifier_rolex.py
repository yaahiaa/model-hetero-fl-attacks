import argparse
import copy
import datetime
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
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
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

full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Max_PSNR Max_Recovered\n")

def clone_state_dict(state_dict):
    cloned = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned


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
        logger.safe(True)
        federation = Federation(epoch, global_parameters, cfg['model_rate'], label_split)
        train(model_history_block2, model_history_fcnn, fcnn_attack_cache, dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch)
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


def train(model_history_block2, model_history_fcnn, fcnn_attack_cache, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    distributed_local_parameters = copy.deepcopy(local_parameters)
    img_list_by_user = {}
    num_active_users = len(local)
    dp_mode = get_dp_mode()

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
        (trained_parameters, img_data) = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))

        img_list = copy.deepcopy(img_data)
        img_list_by_user[int(user_idx[m])] = copy.deepcopy(img_data)

        if dp_mode == 'distributed':
            privatized_parameters, ddp_report = apply_distributed_dp_to_update(
                base_parameters=distributed_local_parameters[m],
                trained_parameters=trained_parameters,
                clip_norm=float(cfg.get('ddp_clip_norm', 1.0)),
                noise_multiplier=float(cfg.get('ddp_noise_multiplier', 0.005)),
                num_active_users=num_active_users,
                enabled=True,
            )
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
            privatized_parameters, ldp_report = apply_local_dp_to_update(
                base_parameters=distributed_local_parameters[m],
                trained_parameters=trained_parameters,
                clip_norm=float(cfg.get('ldp_clip_norm', 1.0)),
                noise_multiplier=float(cfg.get('ldp_noise_multiplier', 0.005)),
                enabled=True,
            )
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
    
    federation.combine(local_parameters, param_idx, user_idx)

    honest_aggregated_state = clone_state_dict(federation.global_parameters)

    attack_source_round = int(cfg.get('attack_source_round', 3))
    attack_replay_round = int(cfg.get('attack_replay_round', 4))

    if cfg['model_name'] == 'fcnn' and epoch == attack_source_round:
        for key in ['layers.0.weight', 'layers.0.bias']:
            fcnn_attack_cache['source_honest_parent'][key] = honest_aggregated_state[key].detach().clone()

        if bool(cfg.get('attack_replay_enabled', True)):
            # This is the malicious server action:
            # instead of using the honest aggregate from round 3 as parent for round 4,
            # replay the real parent that was used at the start of round 3.
            federation.global_parameters = clone_state_dict(federation.initial_parent_state)

            if cfg.get('debug_attack_replay', False):
                print(
                    f'[REPLAY] epoch={epoch}: stored honest source aggregate, '
                    f'but replayed previous real parent for epoch {epoch + 1}',
                    flush=True,
                )
        else:
            federation.global_parameters = honest_aggregated_state

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
            max_psnr_list = []
            num_recovered_list = []

            if epoch != attack_replay_round:
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
                    max_pearson, max_psnr, num_recovered = reconstruct_image(
                        weight_grad,
                        bias_grad,
                        target_img_list,
                    )

                    max_pearson_list.append(max_pearson)
                    max_psnr_list.append(max_psnr)
                    num_recovered_list.append(num_recovered)
                else:
                    print(
                        f'RECONSTRUCT: skipped because no image batch was recorded '
                        f'for target user {user_idx[m]}',
                        flush=True,
                    )

            if len(max_pearson_list) > 0:
                fp.write(
                    "%s %s %s %s\n" % (
                        cfg['local_train_size'],
                        max(max_pearson_list),
                        max(max_psnr_list),
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

    if len(avg_pearson) == 0:
        print("RECONSTRUCT: no valid partial reconstructions", flush=True)
        return (0.0, 0.0, 0)
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
