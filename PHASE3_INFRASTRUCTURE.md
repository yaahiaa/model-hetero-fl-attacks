# lite Infrastructure Overhead

## What This Phase Implements

- Optional IPFS/Kubo artifact backend for model artifacts.
- Optional Fabric-style ledger latency estimator for commitment ledger operations.
- A separate infrastructure benchmark that measures commitment-layer overhead without running FL convergence training.
- No real Hyperledger Fabric deployment.

This phase is intentionally lightweight so thesis claims can distinguish algorithmic behavior from deployment overhead:

> Convergence and defense effectiveness are evaluated using the deterministic local commitment backend. Deployment overhead is evaluated separately using an IPFS artifact backend and a Fabric-style ledger latency model.

## Local Artifact Store + JSON Ledger

```bash
python scripts/benchmark_commitment_infrastructure.py \
  --out_dir results/prototype5_infra_local \
  --artifact_store_backend local \
  --ledger_backend json \
  --num_rounds 10 \
  --num_verifiers 3 \
  --model_size_mb 5
```

## Local IPFS/Kubo Setup

### Option A: Installed Kubo

```bash
ipfs daemon
```

### Option B: Docker

```bash
docker run -d --name ipfs_host \
  -p 4001:4001 -p 5001:5001 -p 8080:8080 \
  ipfs/kubo:latest
```

Then run the benchmark:

```bash
python scripts/benchmark_commitment_infrastructure.py \
  --out_dir results/prototype5_infra_ipfs_fabric_est \
  --artifact_store_backend ipfs \
  --ledger_backend fabric_estimate \
  --num_rounds 10 \
  --num_verifiers 3 \
  --model_size_mb 5 \
  --ipfs_api_url http://127.0.0.1:5001
```

## How To Interpret Results

- `local/json` = algorithmic overhead plus local file persistence overhead.
- `ipfs/json` = artifact-store deployment overhead added by IPFS.
- `local/fabric_estimate` = estimated permissioned-ledger latency added on top of the local ledger implementation.
- `ipfs/fabric_estimate` = estimated hybrid deployment overhead for IPFS artifacts plus a Fabric-style ledger latency model.

Important:

```text
fabric_estimate is not a real blockchain. It is a configurable latency model used to estimate deployment overhead from a permissioned ledger.
```

## Output Files

The infrastructure benchmark writes files that are intentionally named for infrastructure-only reporting:

- `infrastructure_overhead_raw.csv`
- `infrastructure_overhead_summary.json`
- `commitment_overhead.jsonl`
- `commitment_ledger.json`

Use these outputs for deployment-overhead discussion, not for convergence or accuracy reporting.

## Optional Training Integration

Normal training still defaults to the prototype4 local backend combination:

```bash
python src/train_classifier_rolex.py \
  --artifact_store_backend local \
  --ledger_backend json
```

Optional deployment-style runs can enable IPFS artifacts and the Fabric-style estimator:

```bash
python src/train_classifier_rolex.py \
  --artifact_store_backend ipfs \
  --ledger_backend fabric_estimate \
  --fabric_estimate_sleep False
```

If IPFS is not installed, normal training still works as long as `artifact_store_backend=local`.
