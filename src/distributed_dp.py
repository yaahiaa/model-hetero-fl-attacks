from collections import OrderedDict
import math
import torch


def _is_float_tensor(x):
    return torch.is_tensor(x) and torch.is_floating_point(x)


def update_l2_norm(base_parameters, trained_parameters):
    total = None

    for k in trained_parameters:
        if k not in base_parameters:
            continue
        if not _is_float_tensor(trained_parameters[k]):
            continue

        delta = trained_parameters[k].detach() - base_parameters[k].detach()
        sq = torch.sum(delta.float() * delta.float())
        total = sq if total is None else total + sq.to(total.device)

    if total is None:
        return 0.0

    return float(torch.sqrt(total).item())


def apply_distributed_dp_to_update(
    base_parameters,
    trained_parameters,
    clip_norm,
    noise_multiplier,
    num_active_users,
    enabled=True,
):
    """
    Simulated distributed-DP style client-side perturbation.

    This is not a secure distributed-DP protocol. It does not implement secure
    aggregation, threshold cryptography, jointly generated randomness, or formal
    privacy accounting. It only simulates the effect of client-contributed
    Gaussian noise shares under server-side averaging.

    Delta = trained_parameters - base_parameters
    Delta is clipped to L2 norm <= clip_norm.
    Each client adds only a share of the total Gaussian noise:
      sigma_total = noise_multiplier * clip_norm
      sigma_client = sqrt(num_active_users) * sigma_total

    If the server averages k independent noisy client models, the aggregate
    noise standard deviation is approximately sigma_total.
    """
    clip_norm = float(clip_norm)
    noise_multiplier = float(noise_multiplier)
    num_active_users = int(num_active_users)

    if clip_norm <= 0:
        raise ValueError('ddp_clip_norm must be > 0')
    if noise_multiplier < 0:
        raise ValueError('ddp_noise_multiplier must be >= 0')
    if num_active_users <= 0:
        raise ValueError('num_active_users must be > 0')

    norm = update_l2_norm(base_parameters, trained_parameters)

    if not enabled:
        sigma_total = noise_multiplier * clip_norm
        sigma_client = math.sqrt(num_active_users) * sigma_total
        return trained_parameters, {
            'ddp_enabled': False,
            'ddp_update_norm': norm,
            'ddp_clip_factor': 1.0,
            'ddp_clip_norm': clip_norm,
            'ddp_noise_multiplier': noise_multiplier,
            'ddp_num_active_users': num_active_users,
            'ddp_sigma_total': float(sigma_total),
            'ddp_sigma_client': float(sigma_client),
        }

    clip_factor = min(1.0, clip_norm / (norm + 1.0e-12))
    sigma_total = noise_multiplier * clip_norm
    sigma_client = math.sqrt(num_active_users) * sigma_total

    privatized = OrderedDict()

    for k, trained_v in trained_parameters.items():
        if k not in base_parameters or not _is_float_tensor(trained_v):
            privatized[k] = trained_v
            continue

        base_v = base_parameters[k].detach()
        delta = trained_v.detach() - base_v
        clipped_delta = delta * clip_factor

        if sigma_client > 0:
            noise = torch.normal(
                mean=0.0,
                std=sigma_client,
                size=clipped_delta.shape,
                device=clipped_delta.device,
                dtype=clipped_delta.dtype,
            )
            clipped_delta = clipped_delta + noise

        privatized[k] = (base_v + clipped_delta).to(trained_v.dtype)

    return privatized, {
        'ddp_enabled': True,
        'ddp_update_norm': norm,
        'ddp_clip_factor': float(clip_factor),
        'ddp_clip_norm': clip_norm,
        'ddp_noise_multiplier': noise_multiplier,
        'ddp_num_active_users': num_active_users,
        'ddp_sigma_total': float(sigma_total),
        'ddp_sigma_client': float(sigma_client),
    }
