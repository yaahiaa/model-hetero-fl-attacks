import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from commitment_architecture import (  # noqa: E402
    VerificationReport,
    apply_commitment_config_defaults,
    create_commitment_service_from_cfg,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value')


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cleanup_output_dir(out_dir: Path) -> None:
    for name in [
        'commitment_overhead.jsonl',
        'commitment_ledger.json',
        'commitment_events.jsonl',
        'fabric_estimate_ledger_overhead.jsonl',
        'infrastructure_overhead_raw.csv',
        'infrastructure_overhead_summary.json',
    ]:
        target = out_dir / name
        if target.exists():
            target.unlink()


def build_state_dict(model_size_mb: float, marker: float) -> OrderedDict:
    target_bytes = max(4, int(float(model_size_mb) * 1024 * 1024))
    numel = max(1, target_bytes // 4)
    tensor = torch.zeros(numel, dtype=torch.float32)
    tensor[0] = float(marker)
    if numel > 1:
        tensor[1] = float(model_size_mb)
    return OrderedDict({'benchmark.weight': tensor})


def approval_for_round(round_id: int, pattern: str) -> bool:
    if pattern == 'all_approve':
        return True
    if pattern == 'alternating':
        return round_id % 2 == 1
    raise ValueError(f'Unsupported approval_pattern: {pattern}')


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_summary(args, overhead_rows):
    round_rows = [row for row in overhead_rows if int(row.get('round', 0)) > 0]
    artifact_get_rows = [
        row for row in round_rows
        if row.get('event') in {'artifact_get_candidate', 'artifact_get_previous'}
    ]
    artifact_verify_rows = [
        row for row in round_rows
        if row.get('event') in {'artifact_verify_candidate', 'artifact_verify_previous'}
    ]
    ledger_rows = [
        row for row in round_rows
        if str(row.get('event', '')).startswith('ledger_')
    ]
    per_round_measured = {}
    per_round_estimated = {}
    per_round_fabric = {}
    for row in round_rows:
        round_id = int(row['round'])
        per_round_measured.setdefault(round_id, 0.0)
        per_round_estimated.setdefault(round_id, 0.0)
        per_round_fabric.setdefault(round_id, 0.0)
        per_round_measured[round_id] += float(row.get('duration_sec', 0.0) or 0.0)
        estimated = row.get('estimated_deployment_duration_sec')
        if estimated not in (None, ''):
            estimated_value = float(estimated)
            per_round_estimated[round_id] += estimated_value
            if row.get('backend') == 'fabric_estimate':
                per_round_fabric[round_id] += estimated_value

    summary = {
        'artifact_store_backend': args.artifact_store_backend,
        'ledger_backend': args.ledger_backend,
        'num_rounds': args.num_rounds,
        'num_verifiers': args.num_verifiers,
        'model_size_mb': args.model_size_mb,
        'approval_pattern': args.approval_pattern,
        'mean_artifact_put_sec': mean(
            row['duration_sec'] for row in round_rows if row.get('event') == 'artifact_put'
        ),
        'mean_artifact_get_sec': mean(row['duration_sec'] for row in artifact_get_rows),
        'mean_artifact_verify_sec': mean(row['duration_sec'] for row in artifact_verify_rows),
        'mean_ledger_local_sec': mean(row['duration_sec'] for row in ledger_rows),
        'mean_estimated_fabric_sec_per_round': mean(per_round_fabric.values()),
        'mean_total_measured_sec_per_round': mean(per_round_measured.values()),
        'mean_total_estimated_deployment_sec_per_round': mean(per_round_estimated.values()),
    }
    if args.baseline_round_time_sec is not None:
        baseline = float(args.baseline_round_time_sec)
        summary['baseline_round_time_sec'] = baseline
        summary['measured_local_overhead_percentage'] = (
            summary['mean_total_measured_sec_per_round'] / baseline * 100.0
        )
        summary['estimated_deployment_overhead_percentage'] = (
            summary['mean_total_estimated_deployment_sec_per_round'] / baseline * 100.0
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark commitment infrastructure overhead without FL training.')
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--artifact_store_backend', default='local', choices=['local', 'ipfs'])
    parser.add_argument('--ledger_backend', default='json', choices=['json', 'fabric_estimate'])
    parser.add_argument('--num_rounds', default=10, type=int)
    parser.add_argument('--num_verifiers', default=3, type=int)
    parser.add_argument('--model_size_mb', default=5.0, type=float)
    parser.add_argument('--approval_pattern', default='alternating', choices=['all_approve', 'alternating'])
    parser.add_argument('--baseline_round_time_sec', default=None, type=float)
    parser.add_argument('--ipfs_api_url', default='http://127.0.0.1:5001', type=str)
    parser.add_argument('--ipfs_gateway_url', default='http://127.0.0.1:8080', type=str)
    parser.add_argument('--ipfs_pin', default=True, type=str_to_bool)
    parser.add_argument('--ipfs_timeout_sec', default=60.0, type=float)
    parser.add_argument('--ipfs_cid_version', default=1, type=int)
    parser.add_argument('--fabric_estimate_tx_latency_sec', default=2.13, type=float)
    parser.add_argument('--fabric_estimate_throughput_tps', default=28.0, type=float)
    parser.add_argument('--fabric_estimate_storage_per_update_mb', default=1.8, type=float)
    parser.add_argument('--fabric_estimate_sleep', default=False, type=str_to_bool)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    cleanup_output_dir(out_dir)

    cfg = {
        'round_log_dir': str(out_dir),
        'commitment_backend': 'local',
        'artifact_store_backend': args.artifact_store_backend,
        'ledger_backend': args.ledger_backend,
        'commitment_architecture_enabled': True,
        'commitment_artifact_dirname': 'artifacts',
        'commitment_ledger_filename': 'commitment_ledger.json',
        'commitment_quorum_rule': 'majority',
        'ipfs_api_url': args.ipfs_api_url,
        'ipfs_gateway_url': args.ipfs_gateway_url,
        'ipfs_pin': args.ipfs_pin,
        'ipfs_timeout_sec': args.ipfs_timeout_sec,
        'ipfs_cid_version': args.ipfs_cid_version,
        'fabric_estimate_tx_latency_sec': args.fabric_estimate_tx_latency_sec,
        'fabric_estimate_throughput_tps': args.fabric_estimate_throughput_tps,
        'fabric_estimate_storage_per_update_mb': args.fabric_estimate_storage_per_update_mb,
        'fabric_estimate_sleep': args.fabric_estimate_sleep,
    }
    apply_commitment_config_defaults(cfg)
    service = create_commitment_service_from_cfg(cfg)

    genesis_state = build_state_dict(args.model_size_mb, marker=0.0)
    service.ensure_genesis_parent(
        genesis_state,
        metadata={'source': 'infra_benchmark', 'role': 'genesis'},
    )

    latest_state = genesis_state
    for round_id in range(1, args.num_rounds + 1):
        approved = approval_for_round(round_id, args.approval_pattern)
        previous_parent, previous_state = service.fetch_previous_approved_parent()
        if previous_parent is None or previous_state is None:
            raise RuntimeError('Missing approved parent during infrastructure benchmark.')
        latest_state = previous_state
        candidate_state = OrderedDict(
            (name, tensor.detach().cpu().clone()) for name, tensor in latest_state.items()
        )
        candidate_state['benchmark.weight'][0] = float(round_id)
        if candidate_state['benchmark.weight'].numel() > 2:
            candidate_state['benchmark.weight'][2] = float(time.time() % 1.0)

        candidate_record, _ = service.submit_candidate_parent(
            round_id=round_id,
            candidate_state_dict=candidate_state,
            previous_parent_hash=previous_parent.parent_hash,
            schedule_id=f'infra-round-{round_id}',
            active_cohorts=[1.0],
            committee_ids=list(range(args.num_verifiers)),
            metadata={
                'source': 'infra_benchmark',
                'approval_target': approved,
                'model_size_mb': args.model_size_mb,
            },
        )
        fetched_record, fetched_state = service.fetch_candidate_parent(round_id)
        if fetched_record.candidate_parent_hash != candidate_record.candidate_parent_hash:
            raise RuntimeError(f'Fetched candidate hash mismatch for round {round_id}.')
        if not torch.equal(fetched_state['benchmark.weight'], candidate_state['benchmark.weight']):
            raise RuntimeError(f'Fetched candidate tensor mismatch for round {round_id}.')

        for verifier_idx in range(args.num_verifiers):
            report = VerificationReport(
                round_id=round_id,
                verifier_user_id=verifier_idx,
                cohort_rate=1.0,
                approved=approved,
                reason='benchmark_approve' if approved else 'benchmark_reject',
                relative_change=0.1,
                prev_eval={'Local-Loss': 0.0, 'Local-Accuracy': 100.0},
                cand_eval={'Local-Loss': 0.0, 'Local-Accuracy': 100.0},
                metrics={'source': 'infra_benchmark'},
                timestamp=time.time(),
            )
            service.submit_verification_report(report)

        decision = service.finalize_round(round_id, quorum_rule='majority')
        if bool(decision.approved) != bool(approved):
            raise RuntimeError(
                f'Unexpected quorum decision for round {round_id}: '
                f'expected={approved} observed={decision.approved}'
            )

        latest_parent = service.get_latest_approved_parent()
        if latest_parent is None:
            raise RuntimeError(f'Latest approved parent missing after round {round_id}.')
        if approved and latest_parent.parent_hash != candidate_record.candidate_parent_hash:
            raise RuntimeError(f'Approved round {round_id} did not advance latest parent.')
        if not approved and latest_parent.parent_hash != previous_parent.parent_hash:
            raise RuntimeError(f'Rejected round {round_id} incorrectly advanced latest parent.')

    overhead_rows = load_jsonl(out_dir / 'commitment_overhead.jsonl')
    write_csv(out_dir / 'infrastructure_overhead_raw.csv', overhead_rows)
    summary = build_summary(args, overhead_rows)
    with open(out_dir / 'infrastructure_overhead_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'Wrote benchmark outputs to {out_dir}')


if __name__ == '__main__':
    main()