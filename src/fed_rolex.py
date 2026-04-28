import copy
import hashlib
import torch
import numpy as np
from config import cfg
from collections import OrderedDict

def _clone_state_dict(state_dict):
    cloned = OrderedDict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cloned[k] = v.detach().clone()
        else:
            cloned[k] = copy.deepcopy(v)
    return cloned

class Federation:
    def __init__(self, epoch, global_parameters, rate, label_split):
        self.rd = epoch
        self.global_parameters = global_parameters
        self.initial_parent_state = _clone_state_dict(global_parameters)
        self.last_distribution_debug = []
        self.target_weights = ['blocks.2.weight']
        self.target_weights_fcnn = ['layers.0.weight']
        self.target_biases = ['blocks.0.bias', 'blocks.2.bias']
        self.target_biases_fcnn = ['layers.0.bias']
        self.rate = rate
        self.label_split = label_split
        self.model_to_distribute = OrderedDict()
        self.hybrid_trap_cache = {}

        self.make_model_rate()

    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx):
        if cfg['model_name'] == 'fcnn':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]

            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)

                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]

                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]

                                if k in self.target_weights_fcnn:
                                    replay_round = int(cfg.get('attack_replay_round', 4))

                                    if self.model_rate[user_idx[m]] == 0.25 and self.rd == replay_round:
                                        # Assign this client the malicious weights. 
                                        output_idx_i_m = (output_idx_i_m + int((self.model_rate[user_idx[m]]) * v.size()[0])) % v.size()[0]
                                        (output_idx_i_m, sorted_indeces) = torch.sort(output_idx_i_m)

                                idx[m][k] = output_idx_i_m, input_idx_i_m 
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:
                                input_idx_i_m = idx_i[m] # Length of bias tensor should be the same as the length of the weight tensor. 
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'conv':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx[m][k] = output_idx_i_m, input_idx_i_m   # Basically just a tensor containing the indeces to update for a given layer. 
                                idx_i[m] = output_idx_i_m  # Update idx_i, because the output of layer j is the input of layer j+1. 
                            else:
                                input_idx_i_m = idx_i[m]

                                if k in self.target_weights:
                                    input_idx_i_m = (input_idx_i_m + (self.rd - 1)) % v.size()[0]
                                    (input_idx_i_m, sorted_indeces) = torch.sort(input_idx_i_m)

                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:
                                input_idx_i_m = idx_i[m] # Length of bias tensor should be the same as the length of the weight tensor. 
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]

                                if k in self.target_biases:
                                    input_idx_i_m = (input_idx_i_m + (self.rd - 1)) % v.size()[0]
                                    (input_idx_i_m, sorted_indeces) = torch.sort(input_idx_i_m)

                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif 'resnet' in cfg['model_name']:
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'conv1' in k or 'conv2' in k:
                                    if idx_i[m] is None:
                                        idx_i[m] = torch.arange(input_size, device=v.device)
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                elif 'shortcut' in k:
                                    input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                    output_idx_i_m = idx_i[m]
                                elif 'linear' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    raise ValueError('Not valid k')
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'linear' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'transformer':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'embedding' in k.split('.')[-2]:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_input_size = int(np.ceil(input_size * scaler_rate))
                                    input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                                    idx_i[m] = input_idx_i_m
                                elif 'decoder' in k and 'linear2' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                    * scaler_rate))
                                    output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                                        cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                    idx_i[m] = output_idx_i_m
                                else:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                                if 'linear_v' not in k:
                                    idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        else:
            raise ValueError('Not valid model name')
        return idx

    def _extract_from_param_idx(self, user_idx, param_idx):
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]

        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]

            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(
                                v[torch.meshgrid(param_idx[m][k])]
                            )
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)

        return local_parameters


    def extract_honest_local_parameters(self, user_idx, param_idx=None):
        self.make_model_rate()
        if param_idx is None:
            param_idx = self.split_model(user_idx)

        local_parameters = self._extract_from_param_idx(user_idx, param_idx)
        return local_parameters, param_idx

    def _hybrid_trap_enabled_for_round(self):
        if not bool(cfg.get('attack_replay_enabled', True)):
            return False
        if cfg.get('attack_model_mode', 'real_replay') != 'hybrid_trap':
            return False
        if cfg['model_name'] != 'fcnn':
            return False
        attack_source_round = int(cfg.get('attack_source_round', 3))
        attack_replay_round = int(cfg.get('attack_replay_round', 4))
        return int(self.rd) in (attack_source_round, attack_replay_round)

    def _get_fcnn_attack_block_bounds(self, target_rate=0.25):
        weight = self.global_parameters.get('layers.0.weight', None)
        if weight is None:
            return None

        hidden_layer_size = int(weight.size(0))
        client_cap = int(float(target_rate) * hidden_layer_size)
        block_start = client_cap
        block_end = min(block_start + client_cap, hidden_layer_size)

        if client_cap <= 0 or block_start >= block_end:
            return None

        return client_cap, block_start, block_end

    def _generate_pts_with_rng(self, rng, num_elements, down_scale_factor=0.95, mu=0.0, sigma=0.5):
        vector = rng.normal(mu, sigma, num_elements)
        abs_vector = abs(vector) * (-1)
        num_pos = np.floor(num_elements / 2).astype(int)
        pos_indices = rng.choice(num_elements, num_pos, replace=False)
        negative_elements = np.delete(abs_vector, pos_indices)
        abs_vector[pos_indices] = -down_scale_factor * negative_elements
        return abs_vector

    def _get_hybrid_trap_block_tensor(self, key, template_tensor, block_start, block_end):
        cache_key = (key, int(block_start), int(block_end), tuple(template_tensor.shape), str(template_tensor.dtype))
        if cache_key in self.hybrid_trap_cache:
            return self.hybrid_trap_cache[cache_key]

        block_size = int(block_end - block_start)
        seed_material = '{}:{}:{}:{}'.format(
            cfg.get('model_tag', cfg.get('init_seed', 0)),
            key,
            int(block_start),
            int(block_end),
        )
        seed_bytes = hashlib.sha256(seed_material.encode('utf-8')).digest()[:4]
        rng = np.random.RandomState(int.from_bytes(seed_bytes, byteorder='little', signed=False))

        if template_tensor.dim() > 1:
            trap_np = np.zeros((block_size, template_tensor.size(1)), dtype=np.float32)
            for i in range(template_tensor.size(1)):
                trap_np[:, i] = self._generate_pts_with_rng(
                    rng, block_size, down_scale_factor=0.95, mu=0.0, sigma=0.5
                )
        else:
            trap_np = self._generate_pts_with_rng(
                rng, block_size, down_scale_factor=0.95, mu=0.0, sigma=0.5
            ).astype(np.float32)

        trap_tensor = torch.from_numpy(trap_np).to(device=template_tensor.device, dtype=template_tensor.dtype)
        self.hybrid_trap_cache[cache_key] = trap_tensor
        return trap_tensor

    def _apply_hybrid_trap_overrides(self, local_parameters, param_idx, user_idx):
        bounds = self._get_fcnn_attack_block_bounds(target_rate=0.25)
        if bounds is None:
            return []

        _, block_start, block_end = bounds
        trap_events = []

        for m, uid in enumerate(user_idx):
            for key in ['layers.0.weight', 'layers.0.bias']:
                if key not in local_parameters[m] or key not in param_idx[m]:
                    continue

                if key == 'layers.0.weight':
                    output_idx = param_idx[m][key][0]
                else:
                    output_idx = param_idx[m][key]

                local_row_mask = (output_idx >= block_start) & (output_idx < block_end)
                if not bool(torch.any(local_row_mask).item()):
                    continue

                local_row_idx = torch.nonzero(local_row_mask, as_tuple=False).flatten().long()
                trap_row_idx = (output_idx[local_row_mask] - block_start).long()
                local_tensor = local_parameters[m][key].detach().clone()
                trap_block = self._get_hybrid_trap_block_tensor(
                    key,
                    self.global_parameters[key],
                    block_start,
                    block_end,
                )

                local_tensor[local_row_idx] = trap_block[trap_row_idx].to(
                    device=local_tensor.device,
                    dtype=local_tensor.dtype,
                )
                local_parameters[m][key] = local_tensor

                trap_event = {
                    'round': int(self.rd),
                    'user_id': int(uid),
                    'model_rate': float(self.model_rate[uid]),
                    'key': key,
                    'block_start': int(block_start),
                    'block_end': int(block_end),
                }
                trap_events.append(trap_event)

                if cfg.get('debug_hybrid_trap', False):
                    print(f'[HYBRID_TRAP] round={trap_event["round"]}', flush=True)
                    print(f'[HYBRID_TRAP] user_id={trap_event["user_id"]}', flush=True)
                    print(f'[HYBRID_TRAP] model_rate={trap_event["model_rate"]}', flush=True)
                    print(f'[HYBRID_TRAP] key={trap_event["key"]}', flush=True)
                    print(f'[HYBRID_TRAP] block_start={trap_event["block_start"]}', flush=True)
                    print(f'[HYBRID_TRAP] block_end={trap_event["block_end"]}', flush=True)

        return trap_events


    def distribute(self, user_idx):
        # Prototype4: no fake uniform tensors here.
        # The attack is produced by replaying a real parent model across rounds,
        # while the target cohort is shifted in attack_replay_round.
        # hybrid_trap optionally overrides only the attacked FCNN first-layer block.
        local_parameters, param_idx = self.extract_honest_local_parameters(user_idx)
        hybrid_trap_events = []

        if self._hybrid_trap_enabled_for_round():
            hybrid_trap_events = self._apply_hybrid_trap_overrides(local_parameters, param_idx, user_idx)

        self.last_distribution_debug = []
        replay_round = int(cfg.get('attack_replay_round', 4))
        hybrid_trap_events_by_user = {}
        for event in hybrid_trap_events:
            hybrid_trap_events_by_user.setdefault(event['user_id'], []).append(event)

        for m, uid in enumerate(user_idx):
            self.last_distribution_debug.append({
                'round': int(self.rd),
                'user_id': int(uid),
                'model_rate': float(self.model_rate[uid]),
                'attack_model_mode': cfg.get('attack_model_mode', 'real_replay'),
                'target_shift_applied': bool(
                    cfg['model_name'] == 'fcnn'
                    and float(self.model_rate[uid]) == 0.25
                    and int(self.rd) == replay_round
                ),
                'hybrid_trap_applied': int(uid) in hybrid_trap_events_by_user,
                'hybrid_trap_keys': [
                    event['key'] for event in hybrid_trap_events_by_user.get(int(uid), [])
                ],
            })

        return local_parameters, param_idx

    def generate_pts(self, num_elements, down_scale_factor=0.95, mu=0.0, sigma=0.5):
        vector = np.random.normal(mu, sigma, num_elements)
        abs_vector = abs(vector) * (-1)
        num_pos = np.floor(num_elements / 2).astype(int)
        pos_indices = np.random.choice(num_elements, num_pos, replace=False)
        negative_elements = np.delete(abs_vector, pos_indices)
        abs_vector[pos_indices] = -down_scale_factor * negative_elements  # set the negative values and turn them positive
        return abs_vector

    def initialize_weights(self, weight_shape, k):
        num_rows = weight_shape[0] # 250
        num_cols = weight_shape[1] # 784

        # print("INITIALIZE_PARAMS: num_rows = %s and num_cols = %s"%(num_rows, num_cols), flush=True)

        weights = np.zeros((num_rows, num_cols))
        for i in range(num_cols):
            weights[:, i] = self.generate_pts(num_rows, down_scale_factor=0.95, mu=0.0, sigma=0.5)
        # print("INITIALIZE_PARAMS: weights[:, 0] = %s"%(weights[:, 0]))
        
        self.model_to_distribute[k] = torch.from_numpy(weights)
    
    def initialize_biases(self, bias_shape, k):
        num_elems = bias_shape

        # print("INITIALIZE_BIASES: num_elems %s"%(num_elems), flush=True)

        biases = np.zeros(num_elems)
        biases = self.generate_pts(num_elems, down_scale_factor=0.95, mu=0.0, sigma=0.5)
        # print("INITIALIZE_PARAMS: biases head = %s"%(biases[:5]))
        
        self.model_to_distribute[k] = torch.from_numpy(biases)


    # Takes in local and global parameters for a particular key and outputs the gradient. 
    def compute_gradient(self, key, client, local_params, global_params):
        global_params_resized = copy.deepcopy(global_params)
        global_params_resized = global_params_resized[:local_params.size()[0]]
        user_grad = torch.subtract(global_params_resized, local_params)
        return user_grad

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        if cfg['model_name'] == 'fcnn':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                my_tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                my_tmp_v += v
                                my_tmp_v[param_idx[m][k]] += user_grad
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'conv':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                my_tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if (k == 'blocks.0.bias'):
                        user_grad = self.compute_gradient(k, user_idx[m], local_parameters[m][k], self.global_parameters[k])

                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                user_grad = self.compute_gradient(k, user_idx[m], local_parameters[m][k], self.global_parameters[k])

                                my_tmp_v += v
                                my_tmp_v[param_idx[m][k]] += user_grad
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1

                                v_diff = torch.subtract(self.global_parameters[k], v)
                                grad_nongrad_diff = torch.subtract(my_tmp_v, tmp_v)
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif 'resnet' in cfg['model_name']:
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'linear' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'linear' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'transformer':
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if k.split('.')[-2] == 'embedding':
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                elif 'decoder' in k and 'linear2' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError('Not valid model name')

        return
