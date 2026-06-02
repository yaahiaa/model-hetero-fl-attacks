# Blockchain/IPFS Commitment Experiments

This mode keeps the existing committee-based RMA defense intact, but swaps the commitment infrastructure:

- model artifacts are serialized and uploaded to IPFS
- the ledger stores only hashes, CIDs, compact metadata, reports, decisions, gas, and block metadata
- local/json remains the default fast-testing mode

Install optional Python dependencies only when using these backends:

```bash
pip install -r requirements-blockchain.txt
```

## IPFS/Kubo Setup

Example for Colab/Linux:

```bash
wget https://dist.ipfs.tech/kubo/v0.28.0/kubo_v0.28.0_linux-amd64.tar.gz
tar -xzf kubo_v0.28.0_linux-amd64.tar.gz
sudo bash kubo/install.sh
ipfs init
ipfs config Addresses.API /ip4/127.0.0.1/tcp/5001
ipfs config Addresses.Gateway /ip4/127.0.0.1/tcp/8080
ipfs daemon > ipfs.log 2>&1 &
```

The training flags assume:

- IPFS API: `http://127.0.0.1:5001`
- IPFS gateway: `http://127.0.0.1:8080/ipfs`

## Ganache Setup

```bash
npm install -g ganache
ganache --host 127.0.0.1 --port 8545 --chain.chainId 1337 --wallet.deterministic > ganache.log 2>&1 &
```

Hardhat or Anvil also work if they expose an Ethereum JSON-RPC endpoint and unlocked local accounts.

## Backend Check

From the repository root:

```bash
python scripts/check_blockchain_ipfs_backends.py \
  --ipfs_api_url http://127.0.0.1:5001 \
  --blockchain_rpc_url http://127.0.0.1:8545 \
  --blockchain_chain_id 1337 \
  --deploy_contract
```

If either service is unavailable, the script reports the failing URL and the missing service.

Expected success output includes:

```text
IPFS add/cat OK
Blockchain RPC OK
Contract compiled OK
Contract deployed at <address>
```

The Solidity contract is compiled with `solc 0.8.24`, optimizer enabled, `optimize_runs=200`, and `viaIR=true`. The `viaIR` setting is required because default Solidity compilation can hit a "Stack too deep" error for the commitment contract.

## Smoke Test

```bash
cd src

python -u train_classifier_rolex.py \
--data_name MNIST \
--model_name fcnn \
--control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
--experiment_method committee \
--experiment_id blockchain_ipfs_smoke_seed31 \
--seed 31 \
--global_epochs 2 \
--local_epochs 1 \
--local_train_size 10 \
--train_batch_size 10 \
--attack_source_round 1 \
--attack_replay_round 2 \
--commitment_backend blockchain \
--artifact_store_backend ipfs \
--ledger_backend ethereum \
--ipfs_api_url http://127.0.0.1:5001 \
--blockchain_rpc_url http://127.0.0.1:8545 \
--blockchain_deploy_contract true \
--blockchain_wait_for_receipt true \
--round_log_dir ../results/blockchain_ipfs_smoke_seed31/round_log \
--results_dir ../results/blockchain_ipfs_smoke_seed31
```

## Overhead/Convergence Run

```bash
cd src

python -u train_classifier_rolex.py \
--data_name MNIST \
--model_name fcnn \
--control_name 1_3_1_iid_fix_a1-b1-c1_bn_1_1 \
--experiment_method committee \
--experiment_id blockchain_ipfs_convergence_seed31 \
--seed 31 \
--global_epochs 30 \
--local_epochs 1 \
--local_train_size 100 \
--train_batch_size 16 \
--convergence_mode true \
--disable_attack_for_convergence true \
--commitment_backend blockchain \
--artifact_store_backend ipfs \
--ledger_backend ethereum \
--ipfs_api_url http://127.0.0.1:5001 \
--blockchain_rpc_url http://127.0.0.1:8545 \
--blockchain_deploy_contract true \
--blockchain_wait_for_receipt true \
--round_log_dir ../results/blockchain_ipfs_convergence_seed31/round_log \
--results_dir ../results/blockchain_ipfs_convergence_seed31
```

## Outputs

Per-event timings are written to:

```text
<round_log_dir>/commitment_overhead.jsonl
```

The experiment summary appends optional columns to:

```text
<results_dir>/overhead_raw.csv
```

Important columns include IPFS add/cat/pin latency, artifact put/get/verify latency, Ethereum submit/finalize latency, receipt wait time, and total gas used.

## Local Fast Mode

The default remains:

```bash
--commitment_backend local \
--artifact_store_backend local \
--ledger_backend json
```

This mode does not require `web3`, `py-solc-x`, `requests`, IPFS, or a blockchain node.
