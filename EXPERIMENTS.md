# Experiments

This branch uses the Rolex RMA entrypoint in `src/train_classifier_rolex.py`.

Run commands from the `src/` directory so `config.py` can resolve `config.yml` correctly:

```bash
cd src
```

## Common Flags

- `--data_name`
- `--model_name`
- `--control_name`
- `--seed`
- `--global_epochs`
- `--local_epochs`
- `--local_train_size`
- `--train_batch_size`
- `--test_batch_size`
- `--attack_source_round`
- `--attack_replay_round`
- `--attack_noise_amount`
- `--experiment_method`
- `--experiment_id`
- `--results_dir`
- `--convergence_mode`
- `--disable_attack_for_convergence`

This branch is a committee-style RMA defense branch, so a typical method tag is:

```bash
--experiment_method committee
```

## Leakage Dataset-Size Sweep

Suggested settings:

- `local_train_size`: `5`, `10`, `15`, `20`
- `attack_source_round=1`
- `attack_replay_round=2`
- `global_epochs=2`
- `local_epochs=1`
- `30` seeds

Example single run:

```bash
python train_classifier_rolex.py \
  --data_name MNIST \
  --model_name fcnn \
  --control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
  --seed 31 \
  --global_epochs 2 \
  --local_epochs 1 \
  --local_train_size 10 \
  --train_batch_size 10 \
  --test_batch_size 50 \
  --attack_source_round 1 \
  --attack_replay_round 2 \
  --experiment_method committee \
  --experiment_id leakage_n10_seed31 \
  --results_dir results
```

Example sweep loop:

```bash
for N in 5 10 15 20; do
  for SEED in $(seq 1 30); do
    python train_classifier_rolex.py \
      --data_name MNIST \
      --model_name fcnn \
      --control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
      --seed "$SEED" \
      --global_epochs 2 \
      --local_epochs 1 \
      --local_train_size "$N" \
      --train_batch_size "$N" \
      --test_batch_size 50 \
      --attack_source_round 1 \
      --attack_replay_round 2 \
      --experiment_method committee \
      --experiment_id "leakage_n${N}_seed${SEED}" \
      --results_dir results
  done
done
```

## Leakage Attack-Noise Sweep

Suggested settings:

- fixed `local_train_size=10`
- `attack_noise_amount`: `0.00`, `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.30`
- `attack_source_round=1`
- `attack_replay_round=2`
- `global_epochs=2`
- `local_epochs=1`
- `30` seeds

Example sweep loop:

```bash
for NOISE in 0.00 0.05 0.10 0.15 0.20 0.25 0.30; do
  for SEED in $(seq 1 30); do
    python train_classifier_rolex.py \
      --data_name MNIST \
      --model_name fcnn \
      --control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
      --seed "$SEED" \
      --global_epochs 2 \
      --local_epochs 1 \
      --local_train_size 10 \
      --train_batch_size 10 \
      --test_batch_size 50 \
      --attack_source_round 1 \
      --attack_replay_round 2 \
      --attack_noise_amount "$NOISE" \
      --experiment_method committee \
      --experiment_id "noise_${NOISE}_seed${SEED}" \
      --results_dir results
  done
done
```

## Convergence And Overhead

Suggested settings:

- attack disabled
- `30` to `50` global epochs
- `3` seeds per method
- `convergence_mode=true`
- `disable_attack_for_convergence=true`

Example:

```bash
python train_classifier_rolex.py \
  --data_name MNIST \
  --model_name fcnn \
  --control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
  --seed 31 \
  --global_epochs 30 \
  --local_epochs 1 \
  --local_train_size 10 \
  --train_batch_size 10 \
  --test_batch_size 50 \
  --experiment_method committee \
  --experiment_id convergence_seed31 \
  --results_dir results \
  --convergence_mode true \
  --disable_attack_for_convergence true
```

## Plotting

After runs complete:

```bash
cd ..
python scripts/plot_rma_experiments.py --results_dir src/results --out_dir results/plots
```

If you run training from the repo root and redirect outputs elsewhere, adjust `--results_dir` accordingly.
