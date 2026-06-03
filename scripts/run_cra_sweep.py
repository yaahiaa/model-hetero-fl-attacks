import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def csv_values(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(',') if value.strip()]


def main():
    parser = argparse.ArgumentParser(description='Run CRA experiment sweeps.')
    parser.add_argument('--output_root', default='results/cra_sweep')
    parser.add_argument('--seeds', default='1')
    parser.add_argument('--local_train_sizes', default='5,10,15,20')
    parser.add_argument('--attack_noise_values', default='0.0')
    parser.add_argument('--defense_modes', default='none,committee')
    parser.add_argument('--dp_noise_multipliers', default='0.0')
    parser.add_argument('--global_epochs', default=5, type=int)
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--data_name', default='MNIST')
    parser.add_argument('--model_name', default='fcnn')
    parser.add_argument('--control_name', default='1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / 'src'
    output_root = Path(args.output_root)
    seeds = csv_values(args.seeds, int)
    local_sizes = csv_values(args.local_train_sizes, int)
    attack_noise_values = csv_values(args.attack_noise_values, float)
    defense_modes = csv_values(args.defense_modes, str)
    dp_noise_multipliers = csv_values(args.dp_noise_multipliers, float)

    for seed, local_size, attack_noise, defense_mode, dp_noise in itertools.product(
        seeds, local_sizes, attack_noise_values, defense_modes, dp_noise_multipliers
    ):
        run_id = f'cra_{defense_mode}_seed{seed}_n{local_size}_noise{attack_noise}_dp{dp_noise}'
        results_dir = output_root / run_id
        command = [
            sys.executable, '-u', 'train_classifier_fed.py',
            '--data_name', args.data_name,
            '--model_name', args.model_name,
            '--control_name', args.control_name,
            '--experiment_method', 'committee_cra' if defense_mode == 'committee' else 'cra_base',
            '--experiment_id', run_id,
            '--experiment-tag', run_id,
            '--seed', str(seed),
            '--global_epochs', str(args.global_epochs),
            '--local_epochs', str(args.local_epochs),
            '--local_train_size', str(local_size),
            '--train_batch_size', str(args.train_batch_size),
            '--attack-noise-amount', str(attack_noise),
            '--defense-mode', defense_mode,
            '--dp-mode', defense_mode if defense_mode in {'ldp', 'ddp'} else 'none',
            '--noise-multiplier', str(dp_noise),
            '--results_dir', str(results_dir),
            '--round_log_dir', str(results_dir / 'round_log'),
        ]
        if defense_mode == 'committee':
            command.extend([
                '--cra_committee_enabled', 'true',
                '--cra_distribution_consistency_enabled', 'true',
                '--cra_parent_commit_enabled', 'true',
            ])
        else:
            command.extend(['--cra_committee_enabled', 'false'])

        print(' '.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=src_dir, check=True)


if __name__ == '__main__':
    main()
