# RMA Experiment Commands

These commands use the existing `src/train_classifier_rolex.py` entry point and only change runtime configuration. This branch exposes three useful experiment modes through the same script:

- `base`: replay attack active, no DP defense
- `ldp`: replay attack active, local DP defense active
- `ddp`: replay attack active, distributed DP defense active

Replace `DATASET`, `MODEL`, and `CONTROL` with the same values you use in your other branches so the runs stay comparable.

## Common Flags

The key sweep knobs are already CLI-configurable:

- `--local_train_size`
- `--global_epochs`
- `--local_epochs`
- `--seed`
- `--attack_source_round`
- `--attack_replay_round`
- `--attack_noise_amount`
- `--noise_scale`
- `--experiment_method`
- `--experiment_id`
- `--results_dir`

## One-Off Examples

Base:

```bash
python src/train_classifier_rolex.py \
  --data_name DATASET \
  --model_name MODEL \
  --control_name CONTROL \
  --dp_mode none \
  --local_dp_enabled False \
  --ddp_enabled False \
  --experiment_method base \
  --experiment_id base_seed31 \
  --seed 31 \
  --global_epochs 2 \
  --local_epochs 1 \
  --local_train_size 10 \
  --attack_source_round 1 \
  --attack_replay_round 2 \
  --attack_noise_amount 0.05 \
  --noise_scale 0.05 \
  --results_dir results
```

LDP:

```bash
python src/train_classifier_rolex.py \
  --data_name DATASET \
  --model_name MODEL \
  --control_name CONTROL \
  --dp_mode local \
  --local_dp_enabled True \
  --ddp_enabled False \
  --ldp_clip_norm 1.0 \
  --ldp_noise_multiplier 0.005 \
  --experiment_method ldp \
  --experiment_id ldp_seed31 \
  --seed 31 \
  --global_epochs 2 \
  --local_epochs 1 \
  --local_train_size 10 \
  --attack_source_round 1 \
  --attack_replay_round 2 \
  --attack_noise_amount 0.05 \
  --noise_scale 0.05 \
  --results_dir results
```

DDP:

```bash
python src/train_classifier_rolex.py \
  --data_name DATASET \
  --model_name MODEL \
  --control_name CONTROL \
  --dp_mode distributed \
  --local_dp_enabled False \
  --ddp_enabled True \
  --ddp_clip_norm 1.0 \
  --ddp_noise_multiplier 0.005 \
  --experiment_method ddp \
  --experiment_id ddp_seed31 \
  --seed 31 \
  --global_epochs 2 \
  --local_epochs 1 \
  --local_train_size 10 \
  --attack_source_round 1 \
  --attack_replay_round 2 \
  --attack_noise_amount 0.05 \
  --noise_scale 0.05 \
  --results_dir results
```

## Leakage Dataset-Size Sweep

Suggested setup:

- `local_train_size`: `5 10 15 20`
- `attack_source_round=1`
- `attack_replay_round=2`
- `global_epochs=2`
- `local_epochs=1`
- `30` seeds

Example bash loop for DDP:

```bash
for N in 5 10 15 20; do
  for SEED in $(seq 1 30); do
    python src/train_classifier_rolex.py \
      --data_name DATASET \
      --model_name MODEL \
      --control_name CONTROL \
      --dp_mode distributed \
      --ddp_enabled True \
      --experiment_method ddp \
      --experiment_id ddp_N${N}_seed${SEED} \
      --seed ${SEED} \
      --global_epochs 2 \
      --local_epochs 1 \
      --local_train_size ${N} \
      --attack_source_round 1 \
      --attack_replay_round 2 \
      --attack_noise_amount 0.00 \
      --noise_scale 0.00 \
      --results_dir results
  done
done
```

Switch the DP flags and `experiment_method` to run the same sweep for `base` or `ldp`.

## Leakage Attack-Noise Sweep

Suggested setup:

- fixed `local_train_size=10`
- noise values: `0.00 0.05 0.10 0.15 0.20 0.25 0.30`
- `attack_source_round=1`
- `attack_replay_round=2`
- `global_epochs=2`
- `local_epochs=1`
- `30` seeds

Example bash loop for LDP:

```bash
for NOISE in 0.00 0.05 0.10 0.15 0.20 0.25 0.30; do
  for SEED in $(seq 1 30); do
    python src/train_classifier_rolex.py \
      --data_name DATASET \
      --model_name MODEL \
      --control_name CONTROL \
      --dp_mode local \
      --local_dp_enabled True \
      --ddp_enabled False \
      --experiment_method ldp \
      --experiment_id ldp_noise${NOISE}_seed${SEED} \
      --seed ${SEED} \
      --global_epochs 2 \
      --local_epochs 1 \
      --local_train_size 10 \
      --attack_source_round 1 \
      --attack_replay_round 2 \
      --attack_noise_amount ${NOISE} \
      --noise_scale ${NOISE} \
      --results_dir results
  done
done
```

## Convergence And Overhead

Use the same method flag you want to evaluate, but disable the malicious replay path while keeping the method behavior active:

- `attack disabled`
- `30` to `50` global epochs
- `3` seeds per method
- `convergence_mode=true`
- `disable_attack_for_convergence=true`

Example DDP convergence run:

```bash
python src/train_classifier_rolex.py \
  --data_name DATASET \
  --model_name MODEL \
  --control_name CONTROL \
  --dp_mode distributed \
  --ddp_enabled True \
  --experiment_method ddp \
  --experiment_id ddp_conv_seed31 \
  --seed 31 \
  --global_epochs 30 \
  --local_epochs 1 \
  --local_train_size 10 \
  --convergence_mode true \
  --disable_attack_for_convergence true \
  --results_dir results
```

Equivalent settings:

- `base`: `--dp_mode none --local_dp_enabled False --ddp_enabled False --experiment_method base`
- `ldp`: `--dp_mode local --local_dp_enabled True --ddp_enabled False --experiment_method ldp`
- `ddp`: `--dp_mode distributed --local_dp_enabled False --ddp_enabled True --experiment_method ddp`

## Plotting

After runs have appended to the CSV files in `results/`, generate the summary plots with:

```bash
python scripts/plot_rma_experiments.py \
  --results_dir results \
  --out_dir results/plots
```

This writes:

- `results/leakage_raw.csv`
- `results/epoch_raw.csv`
- `results/overhead_raw.csv`
- `results/dataset_size_agg.csv`
- `results/noise_agg.csv`
- `results/convergence_agg.csv`
- `results/overhead_agg.csv`
- plot files in `results/plots/`
