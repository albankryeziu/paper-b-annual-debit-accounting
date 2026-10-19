"""Fail-closed compatibility for immutable results produced by an older source closure.

The compatibility manifest never changes an artifact's identity.  It only
authorizes one exact relative path, byte hash, and recorded fingerprint for a
declared downstream use under the current release fingerprint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tpwrs.config import ROOT, scientific_fingerprint
from tpwrs.storage import sha256_file


FINAL_COMPATIBILITY_PATH = ROOT / "artifacts/provenance/FINAL_R10_COMPATIBILITY.json"


def _relative(path: str | Path, root: Path = ROOT) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Compatibility path is outside the release: {target}") from exc


def _recorded_fingerprint(payload: dict[str, Any], fields: Iterable[str]) -> tuple[str, str] | None:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str) and value:
            return field, value
    return None


def compatibility_manifest(path: Path = FINAL_COMPATIBILITY_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Final compatibility manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "TPWRS_FINAL_R10_COMPATIBILITY_V1":
        raise RuntimeError("Final compatibility manifest schema is not recognized.")
    if payload.get("status") != "PASS":
        raise RuntimeError("Final compatibility manifest is not PASS.")
    if payload.get("final_scientific_fingerprint") != scientific_fingerprint():
        raise RuntimeError("Final compatibility manifest belongs to another source closure.")
    return payload


def compatibility_entry(
    path: str | Path,
    *,
    use: str,
    payload: dict[str, Any] | None = None,
    fingerprint_fields: tuple[str, ...] = (
        "scientific_fingerprint",
        "implementation_hash_recorded_by_run",
        "implementation_hash",
    ),
) -> dict[str, Any]:
    """Return a verified authorization for an unchanged historical artifact."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    relative = _relative(target)
    obj = payload or json.loads(target.read_text(encoding="utf-8"))
    recorded = _recorded_fingerprint(obj, fingerprint_fields)
    if recorded is None:
        raise RuntimeError(f"Compatible artifact has no recorded fingerprint: {relative}")
    field, value = recorded
    manifest = compatibility_manifest()
    candidates = [
        row
        for row in manifest.get("artifacts", [])
        if row.get("relative_path") == relative
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Compatibility authorization must contain exactly one entry for {relative}."
        )
    entry = candidates[0]
    if entry.get("sha256") != sha256_file(target):
        raise RuntimeError(f"Compatible artifact bytes changed: {relative}")
    if entry.get("fingerprint_field") != field or entry.get("original_fingerprint") != value:
        raise RuntimeError(f"Compatible artifact fingerprint changed: {relative}")
    if entry.get("final_fingerprint") != scientific_fingerprint():
        raise RuntimeError(f"Compatible artifact final fingerprint is stale: {relative}")
    if use not in entry.get("allowed_downstream_use", []):
        raise RuntimeError(f"Compatibility entry does not authorize {use}: {relative}")
    if not str(entry.get("dependency_justification", "")).strip():
        raise RuntimeError(f"Compatibility entry lacks a dependency justification: {relative}")
    return entry


def require_compatible_status(
    path: str | Path,
    *,
    allowed: tuple[str, ...] = ("PASS",),
    use: str,
    fingerprint_fields: tuple[str, ...] = (
        "scientific_fingerprint",
        "implementation_hash_recorded_by_run",
        "implementation_hash",
    ),
) -> tuple[dict[str, Any], str]:
    """Require PASS plus either current identity or exact manifest authorization."""
    target = Path(path)
    if not target.is_file():
        raise RuntimeError(f"Prerequisite missing: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("status") not in allowed:
        raise RuntimeError(
            f"Prerequisite did not pass: {target}: {payload.get('status')}"
        )
    recorded = _recorded_fingerprint(payload, fingerprint_fields)
    if recorded is not None and recorded[1] == scientific_fingerprint():
        return payload, "CURRENT_FINGERPRINT"
    compatibility_entry(
        target,
        use=use,
        payload=payload,
        fingerprint_fields=fingerprint_fields,
    )
    return payload, "FINAL_R10_EXACT_HASH_COMPATIBILITY"
