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

try:
    from blockchain_compile import (
        SOLC_OPTIMIZE,
        SOLC_OPTIMIZE_RUNS,
        SOLC_VERSION,
        SOLC_VIA_IR,
        compile_commitment_ledger_contract,
    )
except ImportError:
    from .blockchain_compile import (
        SOLC_OPTIMIZE,
        SOLC_OPTIMIZE_RUNS,
        SOLC_VERSION,
        SOLC_VIA_IR,
        compile_commitment_ledger_contract,
    )
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


_OVERHEAD_RESERVED_KEYS = {'round', 'round_id', 'event', 'duration_sec'}


def _sanitize_backend_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    sanitized = dict(metrics or {})
    if 'duration_sec' in sanitized:
        sanitized[f'{prefix}_backend_duration_sec'] = sanitized.pop('duration_sec')
    for key in list(sanitized.keys()):
        if key in _OVERHEAD_RESERVED_KEYS:
            sanitized[f'{prefix}_{key}'] = sanitized.pop(key)
    return sanitized


def _sanitize_overhead_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(payload or {})
    for key in list(sanitized.keys()):
        if key in _OVERHEAD_RESERVED_KEYS:
            sanitized[f'payload_{key}'] = sanitized.pop(key)
    return sanitized


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
        self.backend_name = 'local'
        self.last_metrics: Dict[str, Any] = {}

    def _artifact_path_for_state(self, state_dict) -> Path:
        model_hash = _state_dict_semantic_hash(state_dict)
        return self.artifacts_dir / f'model_{model_hash}.pt'

    def put_model(self, state_dict, metadata: Dict[str, Any]) -> ArtifactRef:
        started = time.perf_counter()
        artifact_path = self._artifact_path_for_state(state_dict)
        if not artifact_path.exists():
            artifact_bytes = _serialize_state_dict_bytes(state_dict)
            temp_path = artifact_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as handle:
                handle.write(artifact_bytes)
            os.replace(temp_path, artifact_path)
        sha256 = _sha256_file(artifact_path)
        self.last_metrics = {
            'artifact_put_time_sec': time.perf_counter() - started,
        }
        return ArtifactRef(
            cid=f'local-{sha256[:12]}',
            uri=artifact_path.resolve().as_uri(),
            sha256=sha256,
            size_bytes=int(artifact_path.stat().st_size),
            encrypted=False,
            encryption_alg=None,
        )

    def get_model(self, artifact_ref: ArtifactRef):
        started = time.perf_counter()
        artifact_path = _file_uri_to_path(artifact_ref.uri)
        state_dict = torch.load(artifact_path, map_location='cpu')
        self.last_metrics = {
            'artifact_get_time_sec': time.perf_counter() - started,
        }
        if isinstance(state_dict, OrderedDict):
            return state_dict
        if isinstance(state_dict, dict):
            return OrderedDict(state_dict.items())
        raise TypeError(f'Unsupported stored artifact type: {type(state_dict)!r}')

    def verify_artifact(self, artifact_ref: ArtifactRef) -> bool:
        started = time.perf_counter()
        artifact_path = _file_uri_to_path(artifact_ref.uri)
        if not artifact_path.exists():
            self.last_metrics = {
                'artifact_verify_time_sec': time.perf_counter() - started,
            }
            return False
        verified = _sha256_file(artifact_path) == artifact_ref.sha256
        self.last_metrics = {
            'artifact_verify_time_sec': time.perf_counter() - started,
        }
        return verified


class IPFSArtifactStore(ArtifactStore):
    def __init__(
        self,
        ipfs_api_url: str = 'http://127.0.0.1:5001',
        ipfs_gateway_url: str = 'http://127.0.0.1:8080/ipfs',
        pin_artifacts: bool = True,
        request_timeout_sec: int = 120,
    ):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                'IPFS artifact backend requires requests. Install optional dependencies with: '
                'pip install -r requirements-blockchain.txt'
            ) from exc

        self.requests = requests
        self.ipfs_api_url = str(ipfs_api_url).rstrip('/')
        self.ipfs_gateway_url = str(ipfs_gateway_url).rstrip('/')
        self.pin_artifacts = bool(pin_artifacts)
        self.request_timeout_sec = int(request_timeout_sec)
        self.backend_name = 'ipfs'
        self.last_metrics: Dict[str, Any] = {}

    def _post(self, path: str, **kwargs):
        url = f'{self.ipfs_api_url}{path}'
        try:
            response = self.requests.post(url, timeout=self.request_timeout_sec, **kwargs)
            response.raise_for_status()
            return response
        except self.requests.RequestException as exc:
            raise RuntimeError(
                f'IPFS API request failed at {url}. Is IPFS/Kubo running at '
                f'{self.ipfs_api_url}? Start it with `ipfs daemon` after `ipfs init`.'
            ) from exc

    def _cat_bytes(self, cid: str) -> Tuple[bytes, float]:
        started = time.perf_counter()
        response = self._post('/api/v0/cat', params={'arg': cid})
        return response.content, time.perf_counter() - started

    def put_model(self, state_dict, metadata: Dict[str, Any]) -> ArtifactRef:
        started = time.perf_counter()
        artifact_bytes = _serialize_state_dict_bytes(state_dict)
        sha256 = _sha256_bytes(artifact_bytes)

        add_started = time.perf_counter()
        files = {'file': ('model.pt', artifact_bytes, 'application/octet-stream')}
        response = self._post('/api/v0/add', files=files)
        ipfs_add_time_sec = time.perf_counter() - add_started
        add_payload = response.json()
        cid = add_payload.get('Hash') or add_payload.get('Cid') or add_payload.get('Name')
        if not cid:
            raise RuntimeError(f'IPFS add response did not include a CID: {add_payload}')

        ipfs_pin_time_sec = 0.0
        if self.pin_artifacts:
            pin_started = time.perf_counter()
            self._post('/api/v0/pin/add', params={'arg': cid})
            ipfs_pin_time_sec = time.perf_counter() - pin_started

        self.last_metrics = {
            'ipfs_add_time_sec': ipfs_add_time_sec,
            'ipfs_pin_time_sec': ipfs_pin_time_sec,
            'artifact_put_time_sec': time.perf_counter() - started,
        }
        return ArtifactRef(
            cid=str(cid),
            uri=f'ipfs://{cid}',
            sha256=sha256,
            size_bytes=len(artifact_bytes),
            encrypted=False,
            encryption_alg=None,
        )

    def get_model(self, artifact_ref: ArtifactRef):
        started = time.perf_counter()
        artifact_bytes, ipfs_cat_time_sec = self._cat_bytes(artifact_ref.cid)
        observed_sha256 = _sha256_bytes(artifact_bytes)
        if observed_sha256 != artifact_ref.sha256:
            raise RuntimeError(
                f'IPFS artifact SHA256 mismatch for cid={artifact_ref.cid}: '
                f'expected={artifact_ref.sha256} observed={observed_sha256}'
            )
        state_dict = torch.load(io.BytesIO(artifact_bytes), map_location='cpu')
        self.last_metrics = {
            'ipfs_cat_time_sec': ipfs_cat_time_sec,
            'artifact_get_time_sec': time.perf_counter() - started,
        }
        if isinstance(state_dict, OrderedDict):
            return state_dict
        if isinstance(state_dict, dict):
            return OrderedDict(state_dict.items())
        raise TypeError(f'Unsupported stored artifact type: {type(state_dict)!r}')

    def verify_artifact(self, artifact_ref: ArtifactRef) -> bool:
        started = time.perf_counter()
        artifact_bytes, ipfs_cat_time_sec = self._cat_bytes(artifact_ref.cid)
        verified = _sha256_bytes(artifact_bytes) == artifact_ref.sha256
        self.last_metrics = {
            'ipfs_cat_time_sec': ipfs_cat_time_sec,
            'artifact_verify_time_sec': time.perf_counter() - started,
        }
        return verified


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


def _require_hex_bytes32(hex_value: str, field_name: str) -> bytes:
    cleaned = str(hex_value).strip().lower()
    if cleaned.startswith('0x'):
        cleaned = cleaned[2:]
    if len(cleaned) != 64:
        raise ValueError(f'{field_name} must be a 32-byte hex string, got {hex_value!r}')
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(f'{field_name} must be hex, got {hex_value!r}') from exc


def _bytes32_to_hex(value) -> str:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str):
        return value[2:] if value.startswith('0x') else value
    return bytes(value).hex()


def _scale_rate(value: float) -> int:
    return int(round(float(value) * 1000000))


def _scale_relative_change(value: float) -> int:
    return int(round(float(value) * 1000000000000))


class EthereumCommitmentLedger(CommitmentLedger):
    def __init__(
        self,
        round_log_dir,
        blockchain_rpc_url: str = 'http://127.0.0.1:8545',
        blockchain_chain_id: int = 1337,
        blockchain_private_key: Optional[str] = None,
        blockchain_account_index: int = 0,
        blockchain_contract_address: Optional[str] = None,
        blockchain_deploy_contract: bool = True,
        blockchain_wait_for_receipt: bool = True,
        blockchain_receipt_timeout_sec: int = 120,
        blockchain_gas_limit: int = 8000000,
        blockchain_gas_price_wei: Optional[int] = None,
        blockchain_store_full_report_json: bool = False,
        blockchain_report_payload_mode: str = 'hash_only',
    ):
        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError(
                'Ethereum ledger backend requires web3. Install optional dependencies with: '
                'pip install -r requirements-blockchain.txt'
            ) from exc

        self.Web3 = Web3
        self.round_log_dir = Path(round_log_dir or Path('output') / 'round_log' / 'prototype2')
        self.round_log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.round_log_dir / 'commitment_events.jsonl'
        self.rpc_url = str(blockchain_rpc_url)
        self.chain_id = int(blockchain_chain_id)
        self.private_key = blockchain_private_key
        self.account_index = int(blockchain_account_index)
        self.wait_for_receipt = bool(blockchain_wait_for_receipt)
        self.receipt_timeout_sec = int(blockchain_receipt_timeout_sec)
        self.gas_limit = int(blockchain_gas_limit)
        self.gas_price_wei = None if blockchain_gas_price_wei in (None, '') else int(blockchain_gas_price_wei)
        self.store_full_report_json = bool(blockchain_store_full_report_json) or str(blockchain_report_payload_mode).lower() == 'full_json'
        self.report_payload_mode = str(blockchain_report_payload_mode or 'hash_only').lower()
        if self.report_payload_mode not in {'hash_only', 'full_json'}:
            raise ValueError('blockchain_report_payload_mode must be one of: hash_only, full_json')
        self.backend_name = 'ethereum'
        self.last_metrics: Dict[str, Any] = {}
        self.last_receipt = None

        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.web3.is_connected():
            raise RuntimeError(
                f'Ethereum RPC is unavailable at {self.rpc_url}. Start a local node such as '
                '`ganache --host 127.0.0.1 --port 8545 --chain.chainId 1337 --wallet.deterministic`.'
            )

        if self.private_key:
            self.account = self.web3.eth.account.from_key(self.private_key).address
        else:
            accounts = list(self.web3.eth.accounts)
            if self.account_index >= len(accounts):
                raise RuntimeError(
                    f'Ethereum account index {self.account_index} is unavailable at {self.rpc_url}; '
                    f'node returned {len(accounts)} unlocked accounts.'
                )
            self.account = accounts[self.account_index]

        self.abi, self.bytecode = self._load_contract_artifacts()
        if blockchain_contract_address:
            self.contract_address = self.web3.to_checksum_address(blockchain_contract_address)
            self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)
        elif blockchain_deploy_contract:
            self.contract_address, self.contract = self._deploy_contract()
        else:
            raise RuntimeError(
                'ledger_backend=ethereum requires blockchain_contract_address or blockchain_deploy_contract=true'
            )

    def _load_contract_artifacts(self):
        contract_path = Path(__file__).resolve().parent / 'contracts' / 'CommitmentLedger.sol'
        print('[COMMIT][SOLC] contract path:', contract_path)
        print('[COMMIT][SOLC] solc version:', SOLC_VERSION)
        print('[COMMIT][SOLC] optimize:', SOLC_OPTIMIZE)
        print('[COMMIT][SOLC] optimize_runs:', SOLC_OPTIMIZE_RUNS)
        print('[COMMIT][SOLC] via_ir:', SOLC_VIA_IR)
        abi, bytecode = compile_commitment_ledger_contract(contract_path)
        print('[COMMIT][SOLC] Contract compiled OK')
        return abi, bytecode

    def _tx_options(self) -> Dict[str, Any]:
        options = {
            'from': self.account,
            'gas': self.gas_limit,
            'chainId': self.chain_id,
        }
        if self.gas_price_wei is not None:
            options['gasPrice'] = self.gas_price_wei
        return options

    def _send_transaction(self, function_or_constructor, event_type: str, payload: Dict[str, Any]) -> str:
        started = time.perf_counter()
        options = self._tx_options()
        if self.private_key:
            options['nonce'] = self.web3.eth.get_transaction_count(self.account)
            tx = function_or_constructor.build_transaction(options)
            signed = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
            raw_tx = getattr(signed, 'rawTransaction', None) or getattr(signed, 'raw_transaction')
            tx_hash = self.web3.eth.send_raw_transaction(raw_tx)
        else:
            tx_hash = function_or_constructor.transact(options)

        tx_hash_hex = tx_hash.hex()
        receipt_payload: Dict[str, Any] = {
            'tx_hash': tx_hash_hex,
            'success': True,
        }
        if self.wait_for_receipt:
            receipt_started = time.perf_counter()
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=self.receipt_timeout_sec,
            )
            self.last_receipt = receipt
            receipt_wait_time_sec = time.perf_counter() - receipt_started
            receipt_payload.update({
                'block_number': int(receipt.get('blockNumber')) if receipt.get('blockNumber') is not None else None,
                'gas_used': int(receipt.get('gasUsed')) if receipt.get('gasUsed') is not None else None,
                'effective_gas_price': int(receipt.get('effectiveGasPrice')) if receipt.get('effectiveGasPrice') is not None else None,
                'receipt_wait_time_sec': receipt_wait_time_sec,
                'success': int(receipt.get('status', 1)) == 1,
            })
            if not receipt_payload['success']:
                raise RuntimeError(f'Ethereum transaction failed: {tx_hash_hex}')

        self.last_metrics = {
            **receipt_payload,
            'duration_sec': time.perf_counter() - started,
        }
        self._append_event(event_type, payload, tx_hash_hex, receipt_payload)
        return tx_hash_hex

    def _deploy_contract(self):
        print('[COMMIT][ETH] deploying CommitmentLedger contract')
        contract_factory = self.web3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx_hash = self._send_transaction(contract_factory.constructor(), 'contract_deployed', {})
        if not self.wait_for_receipt:
            raise RuntimeError('blockchain_wait_for_receipt must be true when deploying the contract')
        receipt = self.last_receipt
        address = receipt.get('contractAddress')
        if not address:
            raise RuntimeError(f'Contract deployment did not return an address for tx {tx_hash}')
        print(
            '[COMMIT][ETH] deployment succeeded: '
            f'address={address} tx_hash={tx_hash} '
            f'block_number={receipt.get("blockNumber")} gas_used={receipt.get("gasUsed")} '
            f'receipt_wait_time_sec={self.last_metrics.get("receipt_wait_time_sec")}'
        )
        return address, self.web3.eth.contract(address=address, abi=self.abi)

    def _append_event(self, event_type: str, payload: Dict[str, Any], record_id: str, tx_payload: Dict[str, Any]) -> None:
        _append_jsonl(
            self.events_path,
            {
                'event_type': event_type,
                'record_id': record_id,
                'timestamp': time.time(),
                'contract_address': self.contract_address if hasattr(self, 'contract_address') else None,
                'payload': _json_safe(payload),
                'tx': _json_safe(tx_payload),
            },
        )

    def submit_candidate(self, candidate_record: CandidateRecord) -> str:
        payload = _json_safe(candidate_record)
        metadata_hash = _require_hex_bytes32(_sha256_bytes(json.dumps(payload, sort_keys=True).encode('utf-8')), 'metadata_hash')
        return self._send_transaction(
            self.contract.functions.submitCandidate(
                int(candidate_record.round_id),
                _require_hex_bytes32(candidate_record.previous_parent_hash, 'previous_parent_hash'),
                _require_hex_bytes32(candidate_record.candidate_parent_hash, 'candidate_parent_hash'),
                str(candidate_record.candidate_artifact.cid),
                _require_hex_bytes32(candidate_record.candidate_artifact.sha256, 'artifact_sha256'),
                int(candidate_record.candidate_artifact.size_bytes),
                str(candidate_record.schedule_id),
                [_scale_rate(v) for v in candidate_record.active_cohorts],
                [int(v) for v in candidate_record.committee_ids],
                metadata_hash,
            ),
            'candidate_submitted',
            payload,
        )

    def submit_verifier_report(self, report: VerificationReport) -> str:
        payload = _json_safe(report)
        report_json = json.dumps(payload, sort_keys=True)
        report_hash = _require_hex_bytes32(_sha256_bytes(report_json.encode('utf-8')), 'report_hash')
        report_json_or_empty = report_json if self.store_full_report_json else ''
        return self._send_transaction(
            self.contract.functions.submitVerificationReport(
                int(report.round_id),
                int(report.verifier_user_id),
                _scale_rate(report.cohort_rate),
                bool(report.approved),
                str(report.reason),
                _scale_relative_change(report.relative_change),
                report_hash,
                report_json_or_empty,
            ),
            'report_submitted',
            payload,
        )

    def finalize_round(self, round_id: int, quorum_rule: str = 'majority') -> QuorumDecision:
        tx_hash = self._send_transaction(
            self.contract.functions.finalizeRound(int(round_id), str(quorum_rule or 'majority')),
            'round_finalized',
            {'round_id': int(round_id), 'quorum_rule': str(quorum_rule or 'majority')},
        )
        decision = self.get_decision(round_id)
        self.last_metrics['tx_hash'] = tx_hash
        return decision

    def get_candidate(self, round_id: int) -> CandidateRecord:
        candidate = self.contract.functions.getCandidate(int(round_id)).call()
        if not candidate[0]:
            raise KeyError(f'No candidate record found for round {round_id}')
        artifact = ArtifactRef(
            cid=str(candidate[4]),
            uri=f'ipfs://{candidate[4]}',
            sha256=_bytes32_to_hex(candidate[5]),
            size_bytes=int(candidate[6]),
        )
        return CandidateRecord(
            round_id=int(candidate[1]),
            previous_parent_hash=_bytes32_to_hex(candidate[2]),
            candidate_parent_hash=_bytes32_to_hex(candidate[3]),
            candidate_artifact=artifact,
            schedule_id=str(candidate[7]),
            active_cohorts=[int(v) / 1000000.0 for v in candidate[8]],
            committee_ids=[int(v) for v in candidate[9]],
            metadata={'metadata_hash': _bytes32_to_hex(candidate[12])},
            timestamp=float(candidate[10]),
        )

    def get_reports(self, round_id: int) -> List[VerificationReport]:
        reports = self.contract.functions.getReports(int(round_id)).call()
        result = []
        for report in reports:
            if not report[0]:
                continue
            result.append(VerificationReport(
                round_id=int(report[1]),
                verifier_user_id=int(report[2]),
                cohort_rate=int(report[3]) / 1000000.0,
                approved=bool(report[4]),
                reason=str(report[5]),
                relative_change=int(report[6]) / 1000000000000.0,
                prev_eval={},
                cand_eval={},
                metrics={'report_hash': _bytes32_to_hex(report[7])},
                timestamp=float(report[10]),
            ))
        return result

    def get_decision(self, round_id: int) -> QuorumDecision:
        decision = self.contract.functions.getDecision(int(round_id)).call()
        if not decision[0]:
            raise KeyError(f'No decision found for round {round_id}')
        approved_artifact = None
        if decision[2]:
            candidate = self.get_candidate(round_id)
            approved_artifact = candidate.candidate_artifact
        return QuorumDecision(
            round_id=int(decision[1]),
            approved=bool(decision[2]),
            reason=str(decision[3]),
            num_approved=int(decision[4]),
            num_rejected=int(decision[5]),
            quorum_rule=str(decision[6]),
            approved_parent_hash=_bytes32_to_hex(decision[7]) if decision[2] else None,
            approved_artifact=approved_artifact,
            timestamp=float(decision[9]),
        )

    def get_latest_approved_parent(self) -> Optional[CommittedParent]:
        latest = self.contract.functions.getLatestApprovedParent().call()
        if not latest[0]:
            return None
        artifact = ArtifactRef(
            cid=str(latest[2]),
            uri=f'ipfs://{latest[2]}',
            sha256=_bytes32_to_hex(latest[3]),
            size_bytes=int(latest[4]),
        )
        return CommittedParent(
            round_id=-1,
            parent_hash=_bytes32_to_hex(latest[1]),
            artifact=artifact,
            metadata={},
            timestamp=time.time(),
        )

    def commit_genesis_parent(self, parent_hash: str, artifact: ArtifactRef, metadata: Dict[str, Any]) -> CommittedParent:
        existing = self.get_latest_approved_parent()
        if existing is not None:
            return existing
        payload = {
            'parent_hash': str(parent_hash),
            'artifact': _json_safe(artifact),
            'metadata': _json_safe(metadata or {}),
        }
        metadata_hash = _require_hex_bytes32(_sha256_bytes(json.dumps(payload, sort_keys=True).encode('utf-8')), 'metadata_hash')
        self._send_transaction(
            self.contract.functions.commitGenesisParent(
                _require_hex_bytes32(parent_hash, 'parent_hash'),
                str(artifact.cid),
                _require_hex_bytes32(artifact.sha256, 'artifact_sha256'),
                int(artifact.size_bytes),
                metadata_hash,
            ),
            'genesis_committed',
            payload,
        )
        latest = self.get_latest_approved_parent()
        if latest is None:
            raise RuntimeError('Genesis parent was committed but latest approved parent is empty on chain')
        return latest


class CommitmentService:
    def __init__(self, artifact_store: ArtifactStore, ledger: CommitmentLedger, cfg: Dict[str, Any]):
        self.artifact_store = artifact_store
        self.ledger = ledger
        self.cfg = cfg
        round_log_dir = cfg.get('round_log_dir') or os.path.join('output', 'round_log', 'prototype2')
        self.round_log_dir = Path(round_log_dir)
        self.round_log_dir.mkdir(parents=True, exist_ok=True)
        self.overhead_path = self.round_log_dir / 'commitment_overhead.jsonl'

    def _artifact_metrics(self) -> Dict[str, Any]:
        return _sanitize_backend_metrics(
            getattr(self.artifact_store, 'last_metrics', {}) or {},
            'artifact',
        )

    def _ledger_metrics(self) -> Dict[str, Any]:
        return _sanitize_backend_metrics(
            getattr(self.ledger, 'last_metrics', {}) or {},
            'ledger',
        )

    def _record_overhead(self, overhead_round_id: int, overhead_event: str, overhead_duration_sec: float, **payload) -> None:
        payload = _sanitize_overhead_payload(payload)
        _append_jsonl(
            self.overhead_path,
            {
                'backend': str(self.cfg.get('commitment_backend', 'local')),
                'commitment_backend': str(self.cfg.get('commitment_backend', 'local')),
                'artifact_store_backend': str(self.cfg.get('artifact_store_backend', getattr(self.artifact_store, 'backend_name', 'local'))),
                'ledger_backend': str(self.cfg.get('ledger_backend', getattr(self.ledger, 'backend_name', 'json'))),
                'round': int(overhead_round_id),
                'event': str(overhead_event),
                'duration_sec': float(overhead_duration_sec),
                'success': bool(payload.pop('success', True)),
                **_json_safe(payload),
            },
        )

    def ensure_genesis_parent(self, state_dict, metadata: Optional[Dict[str, Any]] = None) -> CommittedParent:
        latest = self.ledger.get_latest_approved_parent()
        if latest is not None:
            return latest

        artifact_start = time.perf_counter()
        artifact = self.artifact_store.put_model(state_dict, metadata or {})
        self._record_overhead(
            0,
            'artifact_put',
            time.perf_counter() - artifact_start,
            size_bytes=artifact.size_bytes,
            cid=artifact.cid,
            sha256=artifact.sha256,
            **self._artifact_metrics(),
        )

        ledger_start = time.perf_counter()
        committed_parent = self.ledger.commit_genesis_parent(
            parent_hash=_state_dict_semantic_hash(state_dict),
            artifact=artifact,
            metadata=dict(metadata or {}),
        )
        self._record_overhead(
            0,
            'ledger_commit_genesis',
            time.perf_counter() - ledger_start,
            parent_hash=committed_parent.parent_hash,
            cid=artifact.cid,
            sha256=artifact.sha256,
            **self._ledger_metrics(),
        )
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
            sha256=artifact.sha256,
            **self._artifact_metrics(),
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
        self._record_overhead(
            round_id,
            'ledger_submit_candidate',
            time.perf_counter() - ledger_start,
            candidate_id=record_id,
            cid=artifact.cid,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            **self._ledger_metrics(),
        )
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
            sha256=candidate_record.candidate_artifact.sha256,
            verified=verified,
            success=verified,
            **self._artifact_metrics(),
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
            sha256=candidate_record.candidate_artifact.sha256,
            **self._artifact_metrics(),
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
            sha256=committed_parent.artifact.sha256,
            verified=verified,
            success=verified,
            **self._artifact_metrics(),
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
            sha256=committed_parent.artifact.sha256,
            **self._artifact_metrics(),
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
        self._record_overhead(
            report.round_id,
            'ledger_submit_report',
            time.perf_counter() - ledger_start,
            report_id=record_id,
            **self._ledger_metrics(),
        )
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
            **self._ledger_metrics(),
        )
        return decision

    def get_latest_approved_parent(self) -> Optional[CommittedParent]:
        return self.ledger.get_latest_approved_parent()

    def get_latest_approved_parent_state(self):
        _, state_dict = self.fetch_previous_approved_parent()
        return state_dict
