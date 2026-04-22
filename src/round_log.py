import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch


def hash_state_dict(state_dict):
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode('utf-8'))
        hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
        hasher.update(str(tensor.dtype).encode('utf-8'))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


class TransparencyLog:
    """
    Prototype-2 transparency log.

    The server is not trusted to define the benchmark parent model by itself.
    Instead, the benchmark used by clients in round r is the latest *approved*
    parent commitment recorded in this log.

    A candidate parent for round r+1 is only promoted if verifier reports from
    at least one active client per cohort approve it.
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.log_dir / 'parent_snapshots'
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / 'events.jsonl'
        self.latest_approved_path = self.log_dir / 'latest_approved_parent.json'

    def _append_event(self, event):
        with open(self.events_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')

    def _write_latest(self, event):
        with open(self.latest_approved_path, 'w', encoding='utf-8') as f:
            json.dump(event, f, indent=2, sort_keys=True)

    def load_parent_state_dict(self, commitment_or_path):
        if commitment_or_path is None:
            return None
        if isinstance(commitment_or_path, (str, Path)):
            snapshot_path = Path(commitment_or_path)
            state_dict = torch.load(snapshot_path, map_location='cpu')
            return state_dict
        snapshot_path = self.snapshots_dir / commitment_or_path['snapshot_file']
        state_dict = torch.load(snapshot_path, map_location='cpu')
        observed_hash = hash_state_dict(state_dict)
        if observed_hash != commitment_or_path['model_hash']:
            raise RuntimeError(
                f"Logged snapshot hash mismatch: expected={commitment_or_path['model_hash']} observed={observed_hash}"
            )
        return state_dict

    def get_latest_approved_parent(self):
        if not self.latest_approved_path.exists():
            return None
        with open(self.latest_approved_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def bootstrap_initial_parent(self, state_dict, parent_for_round=1):
        existing = self.get_latest_approved_parent()
        if existing is not None:
            return existing

        model_hash = hash_state_dict(state_dict)
        snapshot_file = f'parent_round_{int(parent_for_round)}_{model_hash}.pt'
        torch.save(copy.deepcopy(state_dict), self.snapshots_dir / snapshot_file)

        event = {
            'event_type': 'approved_parent',
            'source': 'bootstrap',
            'round_produced': 0,
            'parent_for_round': int(parent_for_round),
            'model_hash': model_hash,
            'snapshot_file': snapshot_file,
            'required_cohort_rates': [],
            'approved_cohort_rates': [],
            'verifier_reports': [],
            'approved': True,
            'timestamp': int(time.time()),
        }
        event['commitment_id'] = hashlib.sha256(
            json.dumps(event, sort_keys=True).encode('utf-8')
        ).hexdigest()
        self._append_event(event)
        self._write_latest(event)
        return event


    def record_candidate_parent(
        self,
        *,
        epoch,
        state_dict,
        active_users,
        active_user_model_rates,
        verifier_reports,
        candidate_metadata=None,
    ):
        model_hash = hash_state_dict(state_dict)
        snapshot_file = f'parent_round_{int(epoch) + 1}_{model_hash}.pt'
        torch.save(copy.deepcopy(state_dict), self.snapshots_dir / snapshot_file)

        required = sorted({float(r['cohort_rate']) for r in verifier_reports})
        approved = sorted({float(r['cohort_rate']) for r in verifier_reports if r.get('approved', False)})
        quorum_ok = set(required).issubset(set(approved))

        event = {
            'event_type': 'candidate_parent',
            'round_produced': int(epoch),
            'parent_for_round': int(epoch) + 1,
            'model_hash': model_hash,
            'snapshot_file': snapshot_file,
            'active_users': [int(u) for u in active_users],
            'active_user_model_rates': {str(int(k)): float(v) for k, v in active_user_model_rates.items()},
            'required_cohort_rates': required,
            'approved_cohort_rates': approved,
            'verifier_reports': verifier_reports,
            'approved': quorum_ok,
            'timestamp': int(time.time()),
        }

        if candidate_metadata is not None:
            event['candidate_metadata'] = candidate_metadata

        event['commitment_id'] = hashlib.sha256(
            json.dumps(event, sort_keys=True).encode('utf-8')
        ).hexdigest()

        self._append_event(event)

        if quorum_ok:
            approved_event = copy.deepcopy(event)
            approved_event['event_type'] = 'approved_parent'
            self._append_event(approved_event)
            self._write_latest(approved_event)
            return approved_event

        return event
