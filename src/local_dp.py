from collections import OrderedDict
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

        total = sq if total is None else total + sq.to(sq.device)

    if total is None:
        return 0.0

    return float(torch.sqrt(total).item())


def apply_local_dp_to_update(
    base_parameters,
    trained_parameters,
    clip_norm,
    noise_multiplier,
    enabled=True,
):
    """
    Local-DP style client-side perturbation.

    Delta = trained_parameters - base_parameters
    Delta is clipped to L2 norm <= clip_norm.
    Gaussian noise is added locally before the update reaches the server.

    Sent model = base_parameters + clipped_delta + N(0, (noise_multiplier * clip_norm)^2)
    """
    if not enabled:
        return trained_parameters, {
            'ldp_enabled': False,
            'ldp_update_norm': update_l2_norm(base_parameters, trained_parameters),
            'ldp_clip_factor': 1.0,
            'ldp_clip_norm': float(clip_norm),
            'ldp_noise_multiplier': float(noise_multiplier),
            'ldp_noise_std': 0.0,
        }

    clip_norm = float(clip_norm)
    noise_multiplier = float(noise_multiplier)

    if clip_norm <= 0:
        raise ValueError('ldp_clip_norm must be > 0')
    if noise_multiplier < 0:
        raise ValueError('ldp_noise_multiplier must be >= 0')

    norm = update_l2_norm(base_parameters, trained_parameters)
    clip_factor = min(1.0, clip_norm / (norm + 1.0e-12))
    noise_std = noise_multiplier * clip_norm

    privatized = OrderedDict()

    for k, trained_v in trained_parameters.items():
        if k not in base_parameters or not _is_float_tensor(trained_v):
            privatized[k] = trained_v
            continue

        base_v = base_parameters[k].detach()
        delta = trained_v.detach() - base_v
        clipped_delta = delta * clip_factor

        if noise_std > 0:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=clipped_delta.shape,
                device=clipped_delta.device,
                dtype=clipped_delta.dtype,
            )
            clipped_delta = clipped_delta + noise

        privatized[k] = (base_v + clipped_delta).to(trained_v.dtype)

    return privatized, {
        'ldp_enabled': True,
        'ldp_update_norm': norm,
        'ldp_clip_factor': float(clip_factor),
        'ldp_clip_norm': clip_norm,
        'ldp_noise_multiplier': noise_multiplier,
        'ldp_noise_std': float(noise_std),
    }