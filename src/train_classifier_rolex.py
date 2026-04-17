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
from round_log import TransparencyLog, hash_json

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
full_path = os.getcwd() + "/" + cfg['file_output']
fp = open(full_path, 'w')
fp.write("N Max_Pearson Max_PSNR Max_Recovered\n")

def sample_active_users():
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    return torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()


def validate_logged_round(user_id, expected_rate, round_log, server_manifest, server_checkpoint, previous_checkpoint):
    latest_entry = round_log.get_latest_entry()
    if latest_entry is None:
        return False, "round log is empty"

    log_manifest = latest_entry["manifest"]
    log_checkpoint = latest_entry["checkpoint"]

    # Client queries the log directly and checks the server-provided
    # metadata against the independently observed latest log state.
    if hash_json(server_manifest) != hash_json(log_manifest):
        return False, "server manifest differs from log manifest"

    if server_checkpoint != log_checkpoint:
        return False, "server checkpoint differs from latest log checkpoint"

    if not round_log.verify_manifest_against_checkpoint(server_manifest, server_checkpoint):
        return False, "checkpoint does not authenticate manifest"

    if previous_checkpoint is not None and not round_log.verify_checkpoint_extension(previous_checkpoint, server_checkpoint):
        return False, "checkpoint is stale or does not extend prior local checkpoint"

    active_users = server_manifest["active_users"]
    if int(user_id) not in active_users:
        return False, f"user {user_id} not listed as active in manifest"

    logged_rate = float(server_manifest["active_user_model_rates"][str(int(user_id))])
    if not np.isclose(logged_rate, float(expected_rate)):
        return False, f"expected rate {expected_rate}, but log says {logged_rate}"

    return True, "ok"


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

    round_log = TransparencyLog(cfg['round_log_dir'])
    client_round_state = {int(u): None for u in range(cfg['num_users'])}

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
        federation = Federation(epoch, global_parameters, cfg['model_rate'], label_split)
        train(model_history_block2, model_history_fcnn, dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, round_log, client_round_state)
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


def train(model_history_block2, model_history_fcnn, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, round_log, client_round_state):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    user_idx = sample_active_users()
    round_manifest = federation.build_round_manifest(user_idx)
    round_checkpoint = round_log.append_round(round_manifest)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, user_idx, round_log, round_manifest, round_checkpoint, client_round_state)
    num_active_users = len(local)

    lr = optimizer.param_groups[0]['lr']

    start_time = time.time()

    img_list = None
    for m in range(num_active_users):
        lr = cfg['lr_map'][federation.model_rate[user_idx[m]]] # This line modifies the learning rate based on user. 
        (local_parameters[m], img_data) = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
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
    global_model.load_state_dict(federation.global_parameters)


    global_model_state_dict_copy = copy.deepcopy(global_model.state_dict())


    # Append to the model history. 
    if cfg['model_name'] == 'fcnn':
        targetWeights = ['layers.0.weight']
        targetBiases = ['layers.0.bias']
        for m in range(num_active_users):
            weight_grad = None
            bias_grad = None
            max_pearson_overall = 0.0
            max_psnr_overall = 0.0
            max_pearson_list = []
            max_psnr_list = []
            num_recovered_list = []
            for k, v in global_model_state_dict_copy.items():
                if federation.model_rate[user_idx[m]] == 0.25:
                    if (k in targetWeights or k in targetBiases):
                            model_history_fcnn[k].append(global_model_state_dict_copy[k])
                            if (epoch == 2): 
                                
                                if k in targetWeights:
                                    weight_grad = fcnn_leakage(k, user_idx[m], num_active_users, federation.model_rate[user_idx[m]], local_parameters[m][k], model_history_fcnn[k], federation.model_to_distribute[k])
                                else:
                                    bias_grad = fcnn_leakage(k, user_idx[m], num_active_users, federation.model_rate[user_idx[m]], local_parameters[m][k], model_history_fcnn[k], federation.model_to_distribute[k])
                                
                                # print("For epoch %s, weight_grad = %s and bias_grad = %s"%(epoch, weight_grad, bias_grad))

                            # max_overall_avgs = []
                            if (weight_grad is not None and bias_grad is not None):
                                bias_grad_sum = torch.abs(torch.sum(bias_grad)).item()
                                if bias_grad_sum != 0.0:
                                    (max_pearson, max_psnr, num_recovered) = reconstruct_image(weight_grad, bias_grad, img_list)
                                    max_pearson_overall = max(max_pearson_overall, max_pearson)
                                    max_psnr_overall = max(max_psnr_overall, max_psnr)
                                    max_pearson_list.append(max_pearson)
                                    max_psnr_list.append(max_psnr)
                                    num_recovered_list.append(num_recovered)
            if len(max_pearson_list) > 0:
                N_Table = cfg['local_train_size']
                Max_Pearson_Table = max(max_pearson_list)
                Max_PSNR_Table = max(max_psnr_list)
                Max_Recovered_Table = max(num_recovered_list)
                fp.write("%s %s %s %s\n"%(N_Table, Max_Pearson_Table, Max_PSNR_Table, Max_Recovered_Table))


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


def make_local(dataset, data_split, label_split, federation, user_idx, round_log, round_manifest, round_checkpoint, client_round_state):
    num_active_users = len(user_idx)

    # This is the committed parent/global model for phase two.
    # In the prototype, we "send" the full parent model to every active client.
    parent_parameters = copy.deepcopy(federation.global_parameters)

    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]

    for m in range(num_active_users):
        uid = int(user_idx[m])
        model_rate_m = federation.model_rate[uid]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[uid])})['train']

        # Phase-one metadata validation
        metadata_valid, metadata_reason = validate_logged_round(
            user_id=uid,
            expected_rate=model_rate_m,
            round_log=round_log,
            server_manifest=round_manifest,
            server_checkpoint=round_checkpoint,
            previous_checkpoint=client_round_state[uid],
        )

        # Phase-two actual submodel validation
        if metadata_valid:
            expected_local_parameters = federation.extract_expected_local_parameters(
                parent_parameters=parent_parameters,
                param_idx_for_user=param_idx[m],
            )
            submodel_valid, submodel_reason = federation.compare_local_parameters(
                expected_local_parameters=expected_local_parameters,
                received_local_parameters=local_parameters[m],
                atol=0.0,
                rtol=0.0,
            )
        else:
            submodel_valid, submodel_reason = False, f"metadata invalid: {metadata_reason}"

        trusted_round = metadata_valid and submodel_valid
        validation_reason = "ok" if trusted_round else (
            metadata_reason if not metadata_valid else submodel_reason
        )

        if trusted_round:
            client_round_state[uid] = round_checkpoint

        local[m] = Local(
            model_rate=model_rate_m,
            data_loader=data_loader_m,
            label_split=label_split[uid],
            user_id=uid,
            trusted_round=trusted_round,
            validation_reason=validation_reason,
            committed_parent_parameters=parent_parameters,
            expected_param_idx=param_idx[m],
        )

    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split, user_id, trusted_round=True, validation_reason="ok", committed_parent_parameters=None, expected_param_idx=None):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.user_id = user_id
        self.trusted_round = trusted_round
        self.validation_reason = validation_reason
        self.committed_parent_parameters = committed_parent_parameters
        self.expected_param_idx = expected_param_idx

    def train(self, local_parameters, lr, logger):
        if not self.trusted_round:
            info = {
                'info': [
                    f'Client {self.user_id} rejected round before training',
                    f'Reason: {self.validation_reason}',
                    f'Validation response: {cfg["validation_response"]}',
                ]
            }
            logger.append(info, 'train', mean=False)

            if cfg['validation_response'] == 'strict_abort':
                raise RuntimeError(
                    f'Client {self.user_id} rejected round metadata/submodel: {self.validation_reason}'
                )

            # zero_change mode:
            # return the received local parameters unchanged,
            # but move them onto the aggregation device so combine() does not crash
            safe_local_parameters = copy.deepcopy(local_parameters)
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