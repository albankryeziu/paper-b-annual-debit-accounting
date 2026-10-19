from __future__ import annotations


# ============================================================================
# tpwrs_v3.exceptions (consolidated; see SOURCE_MAP.json)
# ============================================================================
class TPWRSV3Error(RuntimeError):
    """Base exception for fail-closed scientific contract failures."""


class ContractError(TPWRSV3Error):
    pass


class ProvenanceError(TPWRSV3Error):
    pass


class SolverFailure(TPWRSV3Error):
    pass


class ReplayFailure(TPWRSV3Error):
    pass


class MethodologyContractConflict(TPWRSV3Error):
    pass


# ============================================================================
# tpwrs_v3.hashing (consolidated; see SOURCE_MAP.json)
# ============================================================================

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(array))
    meta = f"{a.dtype.str}|{a.shape}".encode("ascii")
    return sha256_bytes(meta + memoryview(a).tobytes())


def combined_hash(parts: Mapping[str, str]) -> str:
    return sha256_json(dict(sorted((str(k), str(v)) for k, v in parts.items())))


# ============================================================================
# tpwrs_v3.io_utils (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _atomic_replace(source: str | Path, destination: str | Path) -> None:
    """Replace an artifact, tolerating brief Windows sync/indexer locks."""
    delays = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
    for attempt, delay in enumerate(delays):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(delays) - 1:
                raise
            time.sleep(delay)



def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        _atomic_replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def artifact_metadata(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
