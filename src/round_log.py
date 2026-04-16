from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(data: Any) -> str:
    return sha256_hex(canonical_json(data).encode("utf-8"))


def hash_state_dict(state_dict) -> str:
    """
    Stable hash of a PyTorch state_dict.
    """
    hasher = hashlib.sha256()
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(str(tuple(arr.shape)).encode("utf-8"))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


class TransparencyLog:
    """
    Minimal append-only round log for the prototype.

    This is not a full Merkle transparency log yet.
    It is a linear hash chain:
        root_t = H(root_{t-1} || entry_hash_t)

    Good enough for phase one:
    - append-only checkpoints
    - anti-replay / anti-fork foundation
    - externally queryable latest checkpoint + manifest
    """
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.entries_path = self.log_dir / "entries.jsonl"
        self.latest_path = self.log_dir / "latest.json"

        self._latest_entry: Optional[Dict[str, Any]] = None
        if self.latest_path.exists():
            with open(self.latest_path, "r", encoding="utf-8") as f:
                self._latest_entry = json.load(f)

    def append_round(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        prev_checkpoint = None if self._latest_entry is None else self._latest_entry["checkpoint"]
        prev_root = "0" * 64 if prev_checkpoint is None else prev_checkpoint["root_hash"]

        entry_hash = hash_json(manifest)
        root_hash = sha256_hex(f"{prev_root}:{entry_hash}".encode("utf-8"))

        checkpoint = {
            "round": int(manifest["round"]),
            "tree_size": 1 if prev_checkpoint is None else int(prev_checkpoint["tree_size"]) + 1,
            "prev_root": prev_root,
            "entry_hash": entry_hash,
            "root_hash": root_hash,
            "timestamp": int(time.time()),
        }

        entry = {
            "checkpoint": checkpoint,
            "manifest": manifest,
        }

        with open(self.entries_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        with open(self.latest_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, sort_keys=True, indent=2)

        self._latest_entry = entry
        return copy.deepcopy(checkpoint)

    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        return None if self._latest_entry is None else copy.deepcopy(self._latest_entry)

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        latest = self.get_latest_entry()
        if latest is None:
            return None
        return latest["checkpoint"]

    def verify_checkpoint_extension(
        self,
        previous_checkpoint: Optional[Dict[str, Any]],
        new_checkpoint: Dict[str, Any],
    ) -> bool:
        if previous_checkpoint is None:
            return int(new_checkpoint["tree_size"]) == 1

        return (
            int(new_checkpoint["tree_size"]) > int(previous_checkpoint["tree_size"])
            and new_checkpoint["prev_root"] == previous_checkpoint["root_hash"]
            and int(new_checkpoint["round"]) > int(previous_checkpoint["round"])
        )

    def verify_manifest_against_checkpoint(
        self,
        manifest: Dict[str, Any],
        checkpoint: Dict[str, Any],
    ) -> bool:
        return hash_json(manifest) == checkpoint["entry_hash"]