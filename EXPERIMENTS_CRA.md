# CRA Experiments

This branch uses the CRA entrypoint in `src/train_classifier_fed.py`.

Run commands from the `src/` directory so `config.py` can resolve `config.yml` correctly:

```bash
cd src
```

## A. Baseline CRA, No Committee

```bash
python -u train_classifier_fed.py \
  --data_name MNIST \
  --model_name fcnn \
  --control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1 \
  --experiment_method cra_base \
  --cra_committee_enabled false \
  --global_epochs 10 \
  --local_epochs 1 \
  --local_train_size 10
```

## B. CRA With Committee

```bash
python -u train_classifier_fed.py \
  --data_name MNIST \
  --model_name fcnn \
  --control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1 \
  --experiment_method committee_cra \
  --cra_committee_enabled true \
  --cra_distribution_consistency_enabled true \
  --cra_parent_commit_enabled true \
  --global_epochs 10 \
  --local_epochs 1 \
  --local_train_size 10 \
  --round_log_dir ../results/cra_committee_test/round_log \
  --results_dir ../results/cra_committee_test
```

## C. CRA Committee Convergence With Attack Disabled

```bash
python -u train_classifier_fed.py \
  --data_name MNIST \
  --model_name fcnn \
  --control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1 \
  --experiment_method committee_cra \
  --cra_committee_enabled true \
  --convergence_mode true \
  --disable_attack_for_convergence true \
  --global_epochs 30 \
  --local_train_size 100 \
  --train_batch_size 16
```
