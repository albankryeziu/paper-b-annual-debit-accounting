"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

import sys

sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, csv

G = Path(__file__).resolve().parent
R = Path(
    "C:/Users/Alban/OneDrive - University of Groningen/Desktop/Final Paper Script/PAPER B/TPWRS_FINAL_ARCHIVE/RESULTS"
)
records = []


def hashed(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    records.append({"path": str(p), "sha256": h.hexdigest(), "bytes": p.stat().st_size})
    return h.hexdigest()


def read(p):
    hashed(p)
    return json.loads(p.read_text(encoding="utf8"))


result = []
for y in [2029, 2030, 2031]:
    root = R / f"annual_primary/year_{y}/rho_0.15/annual"
    one = read(root / "M1_LP/EXPERIMENT_CASE.json")["experiment"]
    two = read(root / "M2/EXPERIMENT_CASE.json")["experiment"]
    for k in ["physical_case_hash", "time_grid_id"]:
        assert one[k] == two[k]
    for k in ["values_hours", "participant_ids", "contract_mask_hash"]:
        assert one["allowance"][k] == two["allowance"][k]
    for rule in ["M1_LP", "M2"]:
        info = read(root / rule / "solution.json")
        assert hashed(root / rule / "solution_arrays.npz") == info["arrays_sha256"]
    result.append(
        {
            "year": y,
            "M1LP_M2_physical_and_time_identity": True,
            "same_positionwise_allowance_and_order": True,
            "M1LP_and_M2_solution_array_hashes_pass": True,
        }
    )
(G / "ENDPOINT_MODEL_IDENTITY.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf8"
)
with (G / "ENDPOINT_IDENTITY_SOURCE_HASHES.csv").open("w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes"])
    w.writeheader()
    w.writerows(records)
print(json.dumps(result))
