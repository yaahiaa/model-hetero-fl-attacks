import argparse
import sys
import tempfile
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
    is_ipfs_api_reachable,
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


def build_dummy_state(marker: float) -> OrderedDict:
    tensor = torch.arange(4096, dtype=torch.float32)
    tensor[0] = float(marker)
    return OrderedDict({'smoke.weight': tensor})


def run_backend_case(base_dir: Path, artifact_store_backend: str, ledger_backend: str, ipfs_api_url: str) -> None:
    cfg = {
        'round_log_dir': str(base_dir),
        'commitment_backend': 'local',
        'artifact_store_backend': artifact_store_backend,
        'ledger_backend': ledger_backend,
        'commitment_architecture_enabled': True,
        'commitment_artifact_dirname': 'artifacts',
        'commitment_ledger_filename': 'commitment_ledger.json',
        'commitment_quorum_rule': 'majority',
        'ipfs_api_url': ipfs_api_url,
        'ipfs_gateway_url': 'http://127.0.0.1:8080',
        'ipfs_pin': True,
        'ipfs_timeout_sec': 30.0,
        'ipfs_cid_version': 1,
        'fabric_estimate_tx_latency_sec': 0.01,
        'fabric_estimate_throughput_tps': 100.0,
        'fabric_estimate_storage_per_update_mb': 0.1,
        'fabric_estimate_sleep': False,
    }
    apply_commitment_config_defaults(cfg)
    service = create_commitment_service_from_cfg(cfg)

    genesis_state = build_dummy_state(0.0)
    genesis_parent = service.ensure_genesis_parent(genesis_state, metadata={'source': 'smoke'})
    if genesis_parent is None:
        raise RuntimeError('Genesis parent was not committed.')

    previous_parent, previous_state = service.fetch_previous_approved_parent()
    if previous_parent is None or previous_state is None:
        raise RuntimeError('Failed to fetch approved genesis parent.')
    if not torch.equal(previous_state['smoke.weight'], genesis_state['smoke.weight']):
        raise RuntimeError('Genesis artifact roundtrip mismatch.')

    candidate_state = build_dummy_state(1.0)
    candidate_record, _ = service.submit_candidate_parent(
        round_id=1,
        candidate_state_dict=candidate_state,
        previous_parent_hash=previous_parent.parent_hash,
        schedule_id='smoke-round-1',
        active_cohorts=[1.0],
        committee_ids=[0, 1, 2],
        metadata={'source': 'smoke'},
    )
    if not service.artifact_store.verify_artifact(candidate_record.candidate_artifact):
        raise RuntimeError('Artifact verification failed in smoke test.')

    _, fetched_candidate = service.fetch_candidate_parent(1)
    if not torch.equal(fetched_candidate['smoke.weight'], candidate_state['smoke.weight']):
        raise RuntimeError('Candidate artifact roundtrip mismatch.')

    for verifier_idx in range(3):
        service.submit_verification_report(
            VerificationReport(
                round_id=1,
                verifier_user_id=verifier_idx,
                cohort_rate=1.0,
                approved=True,
                reason='smoke_pass',
                relative_change=0.1,
                prev_eval={'Local-Loss': 0.0, 'Local-Accuracy': 100.0},
                cand_eval={'Local-Loss': 0.0, 'Local-Accuracy': 100.0},
                metrics={'source': 'smoke'},
                timestamp=time.time(),
            )
        )

    decision = service.finalize_round(1, quorum_rule='majority')
    if not decision.approved:
        raise RuntimeError(f'Smoke test quorum decision failed for {artifact_store_backend}/{ledger_backend}.')

    latest_parent = service.get_latest_approved_parent()
    if latest_parent is None or latest_parent.parent_hash != candidate_record.candidate_parent_hash:
        raise RuntimeError('Latest approved parent was not updated after approval.')


def main():
    parser = argparse.ArgumentParser(description='Smoke test commitment backends.')
    parser.add_argument('--out_dir', default=None, type=str)
    parser.add_argument('--ipfs', default=False, type=str_to_bool)
    parser.add_argument('--ipfs_api_url', default='http://127.0.0.1:5001', type=str)
    args = parser.parse_args()

    managed_tmp = None
    if args.out_dir is None:
        managed_tmp = tempfile.TemporaryDirectory(prefix='commitment-smoke-')
        root_dir = Path(managed_tmp.name)
    else:
        root_dir = Path(args.out_dir).resolve()
        root_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ('local', 'json'),
        ('local', 'fabric_estimate'),
    ]
    if args.ipfs or is_ipfs_api_reachable(args.ipfs_api_url, timeout_sec=5.0):
        cases.append(('ipfs', 'json'))

    for artifact_store_backend, ledger_backend in cases:
        case_dir = root_dir / f'{artifact_store_backend}_{ledger_backend}'
        case_dir.mkdir(parents=True, exist_ok=True)
        run_backend_case(case_dir, artifact_store_backend, ledger_backend, args.ipfs_api_url)
        print(f'PASS {artifact_store_backend}/{ledger_backend}')

    if managed_tmp is not None:
        managed_tmp.cleanup()


if __name__ == '__main__':
    main()