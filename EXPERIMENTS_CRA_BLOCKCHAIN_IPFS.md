# CRA Blockchain/IPFS Commitment Experiments

This mode keeps CRA training, attack, reconstruction, and verifier threshold logic unchanged. It swaps only the commitment infrastructure:

- parent model artifacts are stored in IPFS
- genesis, candidate, compact verifier attestations, and final decisions are stored on an Ethereum-compatible local ledger
- local/json remains the default fast mode

Install optional dependencies only for Blockchain/IPFS mode:

```bash
pip install -r requirements-blockchain.txt
```

## IPFS/Kubo Setup

Use:

- API: `http://127.0.0.1:5001`
- Gateway: `http://127.0.0.1:8081/ipfs`

Example:

```bash
ipfs init
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
ipfs config Addresses.Gateway /ip4/127.0.0.1/tcp/8081
ipfs daemon > ipfs.log 2>&1 &
```

## Ganache Setup

Use:

- RPC: `http://127.0.0.1:8545`
- chain id: `1337`

```bash
npm install -g ganache
ganache --host 127.0.0.1 --port 8545 --chain.chainId 1337 --wallet.deterministic > ganache.log 2>&1 &
```

## Backend Check

```bash
python scripts/check_blockchain_ipfs_backends.py \
  --ipfs_api_url http://127.0.0.1:5001 \
  --blockchain_rpc_url http://127.0.0.1:8545 \
  --blockchain_chain_id 1337 \
  --deploy_contract
```

Expected output includes `IPFS add/cat OK`, `Blockchain RPC OK`, `Contract compiled OK`, and `Contract deployed at <address>`.

## CRA Blockchain/IPFS Smoke Test

From `src/`:

```bash
python -u train_classifier_fed.py \
--data_name MNIST \
--model_name fcnn \
--control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1 \
--experiment_method committee_cra \
--cra_committee_enabled true \
--cra_distribution_consistency_enabled true \
--cra_parent_commit_enabled true \
--global_epochs 5 \
--local_epochs 1 \
--local_train_size 10 \
--train_batch_size 1 \
--commitment_backend blockchain \
--artifact_store_backend ipfs \
--ledger_backend ethereum \
--ipfs_api_url http://127.0.0.1:5001 \
--ipfs_gateway_url http://127.0.0.1:8081/ipfs \
--blockchain_rpc_url http://127.0.0.1:8545 \
--blockchain_chain_id 1337 \
--blockchain_deploy_contract true \
--blockchain_wait_for_receipt true \
--round_log_dir ../results/cra_blockchain_ipfs_smoke/round_log \
--results_dir ../results/cra_blockchain_ipfs_smoke
```

## CRA Local/JSON Fast Mode

From `src/`:

```bash
python -u train_classifier_fed.py \
--data_name MNIST \
--model_name fcnn \
--control_name 1_5_1_non-iid-2_fix_a1-b1-c1_bn_1_1 \
--experiment_method committee_cra \
--cra_committee_enabled true \
--cra_distribution_consistency_enabled true \
--cra_parent_commit_enabled true \
--global_epochs 2 \
--local_epochs 1 \
--local_train_size 10 \
--train_batch_size 1 \
--commitment_backend local \
--artifact_store_backend local \
--ledger_backend json \
--round_log_dir ../results/cra_local_json_smoke/round_log \
--results_dir ../results/cra_local_json_smoke
```

## Compact Attestations

Verifier reports are committed as compact attestations. Detailed verifier-side measurements are computed locally to make the decision but are not stored on-chain or uploaded to IPFS. The candidate commitment stores previous and candidate parent artifact identifiers once per round.

Per-event infrastructure timing is written to `<round_log_dir>/commitment_overhead.jsonl`, and summary columns are appended to `<results_dir>/overhead_raw.csv`.
