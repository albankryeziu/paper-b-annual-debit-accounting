"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

import sys

sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, csv, math
import numpy as np
import pandas as pd

BASE = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "VERIFICATION_REPORTS").is_dir() and (p / "GATE1_CERTIFICATES").is_dir()
)
ROOT = Path("C:/Users/Alban/OneDrive - University of Groningen/Desktop/Final Paper Script")
A = ROOT / "PAPER B/TPWRS_FINAL_ARCHIVE"
R = A / "RESULTS"
REAL = A / "IMPLEMENTATION/frozen_inputs/artifacts/real_case"
GATE = BASE / "GATE1_CERTIFICATES"
SEEN = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def register(p):
    p = Path(p).resolve()
    key = str(p)
    if key not in SEEN:
        digest = sha(p)
        SEEN[key] = {"path": key, "sha256": digest, "bytes": p.stat().st_size}
        with (GATE / "ACCESSED_INPUT_HASHES.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(SEEN[key]) + "\n")
    return p


def read(p):
    return json.loads(register(p).read_text(encoding="utf8"))


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf8")


def shj(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}".encode("ascii") + a.tobytes()).hexdigest()


def physical():
    m = read(REAL / "physical_case/FINAL_DUTCH_CASE/physical_case.json")
    path = register(REAL / "physical_case/FINAL_DUTCH_CASE" / m["arrays_file"])
    assert sha(path) == m["arrays_sha256"]
    z = np.load(path, allow_pickle=False)
    data = {
        k: z[k]
        for k in [
            "capacities_mw",
            "requests_mw",
            "contract_mask",
            "eligibility_mask",
            "xbar_mw",
            "p_background_mw",
            "A",
        ]
    }
    data["H"] = np.vstack([z[f"H_{v['index']}"] for v in m["network_states"]])
    data["h"] = np.concatenate([z[f"h_{v['index']}"] for v in m["network_states"]])
    data["F"] = np.concatenate([z[f"rating_{v['index']}"] for v in m["network_states"]])
    data["G"] = data["H"] @ data["A"]
    data["dates"] = pd.to_datetime(m["time_grid"]["timestamps_utc"], utc=True)
    states = {}
    rows = []
    for st in m["network_states"]:
        i = st["index"]
        states[st["state_id"]] = shj(
            {
                "state_id": st["state_id"],
                "row_ids": st["row_ids"],
                "H": sha_arr(z[f"H_{i}"]),
                "h": sha_arr(z[f"h_{i}"]),
                "ratings": sha_arr(z[f"rating_{i}"]),
                "kind": st["state_kind"],
                "outage": st["outaged_element_id"],
            }
        )
        assert st["outaged_element_id"] is None or st["outaged_element_id"] not in st["row_ids"]
        rows.extend([st["state_id"] + "::" + r for r in st["row_ids"]])
    data["network_hash"] = shj(states)
    data["row_ids"] = rows
    assert len(set(rows)) == len(rows) == 8837 and len(states) == 94
    assert (
        data["A"].shape == (45, 114) and np.all(data["F"] > 0) and np.all(data["capacities_mw"] > 0)
    )
    assert np.array_equal(
        data["xbar_mw"],
        data["eligibility_mask"] * np.minimum(data["requests_mw"], data["capacities_mw"]),
    )
    for k, a in data.items():
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.number):
            assert np.isfinite(a).all()
    data["meta"] = m
    return data


def case_identity(data, ix):
    m = data["meta"]
    tg = {
        "timestamps": [m["time_grid"]["timestamps_utc"][int(i)] for i in ix],
        "delta": [m["time_grid"]["delta_t_hours"][int(i)] for i in ix],
    }
    parts = {
        "time": shj(dict(tg, timezone=m["time_grid"]["timezone"])),
        "cap": sha_arr(data["capacities_mw"]),
        "req": sha_arr(data["requests_mw"][ix]),
        "ctr": sha_arr(data["contract_mask"][ix].astype(np.uint8)),
        "elig": sha_arr(data["eligibility_mask"][ix].astype(np.uint8)),
        "pbg": sha_arr(data["p_background_mw"][ix]),
        "A": sha_arr(data["A"]),
        "rows": shj(data["row_ids"]),
        "network": data["network_hash"],
        "balancing": m["balancing_key_hash"],
        "participant_set": m["participant_set_id"],
        "contingency_set": m["contingency_set_id"],
        "sources": shj(dict(sorted(m["source_hashes"].items()))),
    }
    return shj(parts), "time_" + shj(tg)[:16]


def fraction_sum(values):
    from fractions import Fraction

    total = 0
    k = 0
    for x in values:
        n, d = float(x).as_integer_ratio()
        j = d.bit_length() - 1
        if j > k:
            total <<= j - k
            k = j
        total += n << (k - j)
    return Fraction(total, 1 << k)


def gamma(n):
    u = np.finfo(float).eps / 2
    return (float(n) * u) / (1 - float(n) * u)
