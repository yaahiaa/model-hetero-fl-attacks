import copy
import hashlib
import io
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import url2pathname

import torch

from round_log import hash_state_dict


@dataclass
class ArtifactRef:
    cid: str
    uri: str
    sha256: str
    size_bytes: int
    encrypted: bool = False
    encryption_alg: Optional[str] = None


@dataclass
class CandidateRecord:
    round_id: int
    previous_parent_hash: str
    candidate_parent_hash: str
    candidate_artifact: ArtifactRef
    schedule_id: str
    active_cohorts: List[float]
    committee_ids: List[int]
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class VerificationReport:
    round_id: int
    verifier_user_id: int
    cohort_rate: float
    approved: bool
    reason: str
    relative_change: float
    prev_eval: Dict[str, Any]
    cand_eval: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: float


@dataclass
class QuorumDecision:
    round_id: int
    approved: bool
    reason: str
    num_approved: int
    num_rejected: int
    quorum_rule: str
    approved_parent_hash: Optional[str]
    approved_artifact: Optional[ArtifactRef]
    timestamp: float


@dataclass
class CommittedParent:
    round_id: int
    parent_hash: str
    artifact: ArtifactRef
    metadata: Dict[str, Any]
    timestamp: float


def _clone_state_dict_cpu(state_dict) -> OrderedDict:
    cloned = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _state_dict_semantic_hash(state_dict) -> str:
    return hash_state_dict(_clone_state_dict_cpu(state_dict))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _serialize_state_dict_bytes(state_dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(_clone_state_dict_cpu(state_dict), buffer)
    return buffer.getvalue()


def _json_safe(value):
    if is_dataclass(value):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {
            'tensor_blocked': True,
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + '\n')


def _artifact_ref_from_dict(payload: Dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        cid=str(payload['cid']),
        uri=str(payload['uri']),
        sha256=str(payload['sha256']),
        size_bytes=int(payload['size_bytes']),
        encrypted=bool(payload.get('encrypted', False)),
        encryption_alg=payload.get('encryption_alg'),
    )


def _candidate_record_from_dict(payload: Dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        round_id=int(payload['round_id']),
        previous_parent_hash=str(payload['previous_parent_hash']),
        candidate_parent_hash=str(payload['candidate_parent_hash']),
        candidate_artifact=_artifact_ref_from_dict(payload['candidate_artifact']),
        schedule_id=str(payload['schedule_id']),
        active_cohorts=[float(v) for v in payload.get('active_cohorts', [])],
        committee_ids=[int(v) for v in payload.get('committee_ids', [])],
        metadata=dict(payload.get('metadata', {})),
        timestamp=float(payload['timestamp']),
    )


def _verification_report_from_dict(payload: Dict[str, Any]) -> VerificationReport:
    return VerificationReport(
        round_id=int(payload['round_id']),
        verifier_user_id=int(payload['verifier_user_id']),
        cohort_rate=float(payload['cohort_rate']),
        approved=bool(payload['approved']),
        reason=str(payload['reason']),
        relative_change=float(payload['relative_change']),
        prev_eval=dict(payload.get('prev_eval', {})),
        cand_eval=dict(payload.get('cand_eval', {})),
        metrics=dict(payload.get('metrics', {})),
        timestamp=float(payload['timestamp']),
    )


def _quorum_decision_from_dict(payload: Dict[str, Any]) -> QuorumDecision:
    artifact_payload = payload.get('approved_artifact')
    return QuorumDecision(
        round_id=int(payload['round_id']),
        approved=bool(payload['approved']),
        reason=str(payload['reason']),
        num_approved=int(payload['num_approved']),
        num_rejected=int(payload['num_rejected']),
        quorum_rule=str(payload['quorum_rule']),
        approved_parent_hash=payload.get('approved_parent_hash'),
        approved_artifact=_artifact_ref_from_dict(artifact_payload) if artifact_payload else None,
        timestamp=float(payload['timestamp']),
    )


def _committed_parent_from_dict(payload: Dict[str, Any]) -> CommittedParent:
    return CommittedParent(
        round_id=int(payload['round_id']),
        parent_hash=str(payload['parent_hash']),
        artifact=_artifact_ref_from_dict(payload['artifact']),
        metadata=dict(payload.get('metadata', {})),
        timestamp=float(payload['timestamp']),
    )


def _file_uri_to_path(uri: str) -> Path:
    if '://' not in uri:
        return Path(uri)
    parsed = urlparse(uri)
    if parsed.scheme != 'file':
        raise ValueError(f'Unsupported artifact URI scheme: {uri}')
    path_text = url2pathname(parsed.path)
    if os.name == 'nt' and path_text.startswith('/') and len(path_text) > 2 and path_text[2] == ':':
        path_text = path_text[1:]
    if parsed.netloc:
        path_text = f'//{parsed.netloc}{path_text}'
    return Path(path_text)


class ArtifactStore:
    def put_model(self, state_dict, metadata: Dict[str, Any]) -> ArtifactRef:
        raise NotImplementedError

    def get_model(self, artifact_ref: ArtifactRef):
        raise NotImplementedError

    def verify_artifact(self, artifact_ref: ArtifactRef) -> bool:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, round_log_dir, artifact_dirname: str = 'artifacts'):
        self.round_log_dir = Path(round_log_dir or Path('output') / 'round_log' / 'prototype2')
        self.round_log_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.round_log_dir / artifact_dirname
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path_for_state(self, state_dict) -> Path:
        model_hash = _state_dict_semantic_hash(state_dict)
        return self.artifacts_dir / f'model_{model_hash}.pt'

    def put_model(self, state_dict, metadata: Dict[str, Any]) -> ArtifactRef:
        artifact_path = self._artifact_path_for_state(state_dict)
        if not artifact_path.exists():
            artifact_bytes = _serialize_state_dict_bytes(state_dict)
            temp_path = artifact_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as handle:
                handle.write(artifact_bytes)
            os.replace(temp_path, artifact_path)
        sha256 = _sha256_file(artifact_path)
        return ArtifactRef(
            cid=f'local-{sha256[:12]}',
            uri=artifact_path.resolve().as_uri(),
            sha256=sha256,
            size_bytes=int(artifact_path.stat().st_size),
            encrypted=False,
            encryption_alg=None,
        )

    def get_model(self, artifact_ref: ArtifactRef):
        artifact_path = _file_uri_to_path(artifact_ref.uri)
        state_dict = torch.load(artifact_path, map_location='cpu')
        if isinstance(state_dict, OrderedDict):
            return state_dict
        if isinstance(state_dict, dict):
            return OrderedDict(state_dict.items())
        raise TypeError(f'Unsupported stored artifact type: {type(state_dict)!r}')

    def verify_artifact(self, artifact_ref: ArtifactRef) -> bool:
        artifact_path = _file_uri_to_path(artifact_ref.uri)
        if not artifact_path.exists():
            return False
        return _sha256_file(artifact_path) == artifact_ref.sha256


class CommitmentLedger:
    def submit_candidate(self, candidate_record: CandidateRecord) -> str:
        raise NotImplementedError

    def submit_verifier_report(self, report: VerificationReport) -> str:
        raise NotImplementedError

    def finalize_round(self, round_id: int, quorum_rule: str = 'majority') -> QuorumDecision:
        raise NotImplementedError

    def get_candidate(self, round_id: int) -> CandidateRecord:
        raise NotImplementedError

    def get_reports(self, round_id: int) -> List[VerificationReport]:
        raise NotImplementedError

    def get_latest_approved_parent(self) -> Optional[CommittedParent]:
        raise NotImplementedError

    def commit_genesis_parent(self, parent_hash: str, artifact: ArtifactRef, metadata: Dict[str, Any]) -> CommittedParent:
        raise NotImplementedError


class JsonCommitmentLedger(CommitmentLedger):
    def __init__(self, round_log_dir, ledger_filename: str = 'commitment_ledger.json'):
        self.round_log_dir = Path(round_log_dir or Path('output') / 'round_log' / 'prototype2')
        self.round_log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.round_log_dir / ledger_filename
        self.events_path = self.round_log_dir / 'commitment_events.jsonl'
        if not self.ledger_path.exists():
            self._write_state(self._empty_state())

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            'schema_version': 1,
            'genesis_parent': None,
            'candidates': {},
            'reports': {},
            'decisions': {},
            'latest_approved_parent': None,
        }

    def _read_state(self) -> Dict[str, Any]:
        with open(self.ledger_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def _write_state(self, state: Dict[str, Any]) -> None:
        with open(self.ledger_path, 'w', encoding='utf-8') as handle:
            json.dump(_json_safe(state), handle, indent=2, sort_keys=True)

    @staticmethod
    def _record_id(prefix: str, payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(json.dumps(_json_safe(payload), sort_keys=True).encode('utf-8')).hexdigest()
        return f'{prefix}-{digest[:16]}'

    def _append_event(self, event_type: str, payload: Dict[str, Any], record_id: str) -> None:
        _append_jsonl(
            self.events_path,
            {
                'event_type': event_type,
                'record_id': record_id,
                'timestamp': time.time(),
                'payload': _json_safe(payload),
            },
        )

    def submit_candidate(self, candidate_record: CandidateRecord) -> str:
        state = self._read_state()
        round_key = str(candidate_record.round_id)
        payload = _json_safe(candidate_record)
        state['candidates'][round_key] = payload
        state['reports'][round_key] = []
        state['decisions'].pop(round_key, None)
        record_id = self._record_id('candidate', payload)
        self._write_state(state)
        self._append_event('candidate_submitted', payload, record_id)
        return record_id

    def submit_verifier_report(self, report: VerificationReport) -> str:
        state = self._read_state()
        round_key = str(report.round_id)
        reports = state['reports'].setdefault(round_key, [])
        payload = _json_safe(report)
        reports.append(payload)
        record_id = self._record_id('report', payload)
        self._write_state(state)
        self._append_event('report_submitted', payload, record_id)
        return record_id

    def finalize_round(self, round_id: int, quorum_rule: str = 'majority') -> QuorumDecision:
        state = self._read_state()
        round_key = str(round_id)
        candidate_payload = state['candidates'].get(round_key)
        if candidate_payload is None:
            raise KeyError(f'No candidate record found for round {round_id}')
        candidate_record = _candidate_record_from_dict(candidate_payload)
        reports = [_verification_report_from_dict(v) for v in state['reports'].get(round_key, [])]

        num_approved = sum(1 for report in reports if report.approved)
        num_rejected = sum(1 for report in reports if not report.approved)
        approved = False
        reason = 'rejected_no_reports'

        normalized_rule = str(quorum_rule or 'majority').lower()
        if normalized_rule == 'majority':
            approved = num_approved > num_rejected
            reason = 'approved_by_majority' if approved else 'rejected_by_majority'
        elif normalized_rule in {'committee_unanimous', 'all_committee', 'all', 'required_cohorts'}:
            approved = len(reports) > 0 and num_rejected == 0
            reason = 'approved_all_committee' if approved else 'rejected_committee_dissent'
        else:
            raise ValueError(f'Unsupported quorum rule: {quorum_rule}')

        approved_parent_hash = candidate_record.candidate_parent_hash if approved else None
        approved_artifact = candidate_record.candidate_artifact if approved else None
        decision = QuorumDecision(
            round_id=int(round_id),
            approved=approved,
            reason=reason,
            num_approved=num_approved,
            num_rejected=num_rejected,
            quorum_rule=normalized_rule,
            approved_parent_hash=approved_parent_hash,
            approved_artifact=approved_artifact,
            timestamp=time.time(),
        )
        payload = _json_safe(decision)
        record_id = self._record_id('decision', payload)
        state['decisions'][round_key] = payload

        if approved:
            committed_parent = CommittedParent(
                round_id=int(round_id),
                parent_hash=candidate_record.candidate_parent_hash,
                artifact=candidate_record.candidate_artifact,
                metadata=dict(candidate_record.metadata),
                timestamp=decision.timestamp,
            )
            state['latest_approved_parent'] = _json_safe(committed_parent)
        self._write_state(state)
        self._append_event('round_finalized', payload, record_id)
        return decision

    def get_candidate(self, round_id: int) -> CandidateRecord:
        state = self._read_state()
        payload = state['candidates'].get(str(round_id))
        if payload is None:
            raise KeyError(f'No candidate record found for round {round_id}')
        return _candidate_record_from_dict(payload)

    def get_reports(self, round_id: int) -> List[VerificationReport]:
        state = self._read_state()
        return [
            _verification_report_from_dict(payload)
            for payload in state['reports'].get(str(round_id), [])
        ]

    def get_latest_approved_parent(self) -> Optional[CommittedParent]:
        state = self._read_state()
        payload = state.get('latest_approved_parent')
        if payload is None:
            return None
        return _committed_parent_from_dict(payload)

    def commit_genesis_parent(self, parent_hash: str, artifact: ArtifactRef, metadata: Dict[str, Any]) -> CommittedParent:
        state = self._read_state()
        existing = state.get('latest_approved_parent')
        if existing is not None:
            return _committed_parent_from_dict(existing)

        committed_parent = CommittedParent(
            round_id=0,
            parent_hash=str(parent_hash),
            artifact=artifact,
            metadata=dict(metadata or {}),
            timestamp=time.time(),
        )
        payload = _json_safe(committed_parent)
        state['genesis_parent'] = payload
        state['latest_approved_parent'] = payload
        self._write_state(state)
        record_id = self._record_id('genesis', payload)
        self._append_event('genesis_committed', payload, record_id)
        return committed_parent


class CommitmentService:
    def __init__(self, artifact_store: ArtifactStore, ledger: CommitmentLedger, cfg: Dict[str, Any]):
        self.artifact_store = artifact_store
        self.ledger = ledger
        self.cfg = cfg
        round_log_dir = cfg.get('round_log_dir') or os.path.join('output', 'round_log', 'prototype2')
        self.round_log_dir = Path(round_log_dir)
        self.round_log_dir.mkdir(parents=True, exist_ok=True)
        self.overhead_path = self.round_log_dir / 'commitment_overhead.jsonl'

    def _record_overhead(self, round_id: int, event: str, duration_sec: float, **payload) -> None:
        _append_jsonl(
            self.overhead_path,
            {
                'round': int(round_id),
                'event': str(event),
                'duration_sec': float(duration_sec),
                **_json_safe(payload),
            },
        )

    def ensure_genesis_parent(self, state_dict, metadata: Optional[Dict[str, Any]] = None) -> CommittedParent:
        latest = self.ledger.get_latest_approved_parent()
        if latest is not None:
            return latest

        artifact_start = time.perf_counter()
        artifact = self.artifact_store.put_model(state_dict, metadata or {})
        self._record_overhead(0, 'artifact_put', time.perf_counter() - artifact_start, size_bytes=artifact.size_bytes, cid=artifact.cid)

        ledger_start = time.perf_counter()
        committed_parent = self.ledger.commit_genesis_parent(
            parent_hash=_state_dict_semantic_hash(state_dict),
            artifact=artifact,
            metadata=dict(metadata or {}),
        )
        self._record_overhead(0, 'ledger_commit_genesis', time.perf_counter() - ledger_start, parent_hash=committed_parent.parent_hash)
        return committed_parent

    def submit_candidate_parent(
        self,
        round_id: int,
        candidate_state_dict,
        previous_parent_hash: str,
        schedule_id: str,
        active_cohorts: List[float],
        committee_ids: List[int],
        metadata: Dict[str, Any],
    ) -> Tuple[CandidateRecord, str]:
        artifact_start = time.perf_counter()
        artifact = self.artifact_store.put_model(candidate_state_dict, metadata or {})
        self._record_overhead(
            round_id,
            'artifact_put',
            time.perf_counter() - artifact_start,
            size_bytes=artifact.size_bytes,
            cid=artifact.cid,
        )

        candidate_record = CandidateRecord(
            round_id=int(round_id),
            previous_parent_hash=str(previous_parent_hash),
            candidate_parent_hash=_state_dict_semantic_hash(candidate_state_dict),
            candidate_artifact=artifact,
            schedule_id=str(schedule_id),
            active_cohorts=[float(v) for v in active_cohorts],
            committee_ids=[int(v) for v in committee_ids],
            metadata=dict(metadata or {}),
            timestamp=time.time(),
        )

        ledger_start = time.perf_counter()
        record_id = self.ledger.submit_candidate(candidate_record)
        self._record_overhead(round_id, 'ledger_submit_candidate', time.perf_counter() - ledger_start, candidate_id=record_id)
        return candidate_record, record_id

    def fetch_candidate_parent(self, round_id: int):
        candidate_record = self.ledger.get_candidate(round_id)

        verify_start = time.perf_counter()
        verified = self.artifact_store.verify_artifact(candidate_record.candidate_artifact)
        self._record_overhead(
            round_id,
            'artifact_verify',
            time.perf_counter() - verify_start,
            cid=candidate_record.candidate_artifact.cid,
            verified=verified,
        )
        if not verified:
            raise RuntimeError(f'Candidate artifact verification failed for round {round_id}')

        load_start = time.perf_counter()
        candidate_state = self.artifact_store.get_model(candidate_record.candidate_artifact)
        self._record_overhead(
            round_id,
            'artifact_get',
            time.perf_counter() - load_start,
            cid=candidate_record.candidate_artifact.cid,
            size_bytes=candidate_record.candidate_artifact.size_bytes,
        )

        observed_hash = _state_dict_semantic_hash(candidate_state)
        if observed_hash != candidate_record.candidate_parent_hash:
            raise RuntimeError(
                f'Candidate parent hash mismatch for round {round_id}: '
                f'expected={candidate_record.candidate_parent_hash} observed={observed_hash}'
            )

        return candidate_record, candidate_state

    def fetch_previous_approved_parent(self):
        committed_parent = self.ledger.get_latest_approved_parent()
        if committed_parent is None:
            return None, None

        verify_start = time.perf_counter()
        verified = self.artifact_store.verify_artifact(committed_parent.artifact)
        self._record_overhead(
            committed_parent.round_id,
            'artifact_verify',
            time.perf_counter() - verify_start,
            cid=committed_parent.artifact.cid,
            verified=verified,
        )
        if not verified:
            raise RuntimeError(f'Approved parent artifact verification failed for round {committed_parent.round_id}')

        load_start = time.perf_counter()
        parent_state = self.artifact_store.get_model(committed_parent.artifact)
        self._record_overhead(
            committed_parent.round_id,
            'artifact_get',
            time.perf_counter() - load_start,
            cid=committed_parent.artifact.cid,
            size_bytes=committed_parent.artifact.size_bytes,
        )

        observed_hash = _state_dict_semantic_hash(parent_state)
        if observed_hash != committed_parent.parent_hash:
            raise RuntimeError(
                f'Approved parent hash mismatch: expected={committed_parent.parent_hash} observed={observed_hash}'
            )

        return committed_parent, parent_state

    def submit_verification_report(self, report: VerificationReport) -> str:
        ledger_start = time.perf_counter()
        record_id = self.ledger.submit_verifier_report(report)
        self._record_overhead(report.round_id, 'ledger_submit_report', time.perf_counter() - ledger_start, report_id=record_id)
        return record_id

    def finalize_round(self, round_id: int, quorum_rule: Optional[str] = None) -> QuorumDecision:
        rule = quorum_rule or self.cfg.get('commitment_quorum_rule', 'majority')
        ledger_start = time.perf_counter()
        decision = self.ledger.finalize_round(round_id, quorum_rule=rule)
        self._record_overhead(
            round_id,
            'ledger_finalize',
            time.perf_counter() - ledger_start,
            approved=decision.approved,
            num_approved=decision.num_approved,
            num_rejected=decision.num_rejected,
            quorum_rule=decision.quorum_rule,
        )
        return decision

    def get_latest_approved_parent(self) -> Optional[CommittedParent]:
        return self.ledger.get_latest_approved_parent()

    def get_latest_approved_parent_state(self):
        _, state_dict = self.fetch_previous_approved_parent()
        return state_dict
