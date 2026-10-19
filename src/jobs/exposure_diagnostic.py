"""Post-solve participantwise exposure-controlled premium witnesses.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from pathlib import Path
import sys
from fractions import Fraction
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "IMPLEMENTATION/source"))
from jobs.m1_exposure_matched import exact_energy, read, exposure
from jobs.certificates import _debit_guard
from jobs.debit_rule_family import year_case
from tpwrs.persistence import load_physical_case
from tpwrs.debit_rules import weighted_budget_audit, require
from tpwrs.replay import physical_objectives
from tpwrs.storage import atomic_write_json, sha256_file


def main():
    manifest = read(HERE / "experiment_manifest.json")
    require(
        manifest["status"] in ("RUNNING", "COMPLETE", "COMPLETE_VALIDATION_REJECTION"),
        "Accepted campaign state required",
    )
    parent = load_physical_case(
        ROOT / "IMPLEMENTATION/frozen_inputs/artifacts/real_case/physical_case/FINAL_DUTCH_CASE"
    )
    destination = HERE / "final_evidence"
    destination.mkdir(exist_ok=True)
    rows = []
    for folder in sorted(HERE.glob("year_*")):
        year = int(folder.name.split("_")[1])
        case = year_case(parent, year)
        D, U = folder / "CONSERVATIVE_STATE", folder / "UNIFORM_PREMIUM"
        if not (D / "result.json").is_file() or not (U / "result.json").is_file():
            continue
        with np.load(D / "GUARDED_ANNUAL_WITNESS.npz", allow_pickle=False) as z:
            source, q = z["x_mw"], z["q"]
        with np.load(U / "GUARDED_ANNUAL_WITNESS.npz", allow_pickle=False) as z:
            uniform = z["x_mw"]
        reference = exact_energy(uniform, case.time_grid.delta)
        initial = exact_energy(source, case.time_grid.delta)
        fde = exposure(source, case.time_grid.delta, case.capacities_mw)[0]
        factors = np.ones(case.n_participants)
        for n in range(case.n_participants):
            if initial[n] > reference[n]:
                limit = float(reference[n] / Fraction(float(case.capacities_mw[n])))
                target = max(0.0, limit - _debit_guard(limit, case.n_time))
                factors[n] = max(0.0, float(np.nextafter(target / fde[n], 0.0)))
        x = source * factors
        exact = exact_energy(x, case.time_grid.delta)
        excess = [e - r for e, r in zip(exact, reference)]
        require(
            all(e <= 0 for e in excess), "Uniform-exposure witness exceeds saved reference energy"
        )
        H = np.array(read(D / "EXPERIMENT_CASE.json")["experiment"]["allowance"]["values_hours"])
        audit = weighted_budget_audit(case, x, q, H)
        require(audit["strict_pass"], "Exposure-controlled weighted witness infeasible")
        S, E = physical_objectives(case, x)
        uniform_S = read(U / "result.json")["policies"][1]["S"]
        witness_folder = (
            folder
            if (folder / "EXPOSURE_CONTROLLED_ANNUAL_WITNESS.npz").is_file()
            else destination / f"year_{year}"
        )
        witness_folder.mkdir(exist_ok=True)
        out = witness_folder / "EXPOSURE_CONTROLLED_ANNUAL_WITNESS.npz"
        if out.is_file():
            with np.load(out, allow_pickle=False) as z:
                require(np.array_equal(z["x_mw"], x), "Exposure witness regeneration differs")
        else:
            np.savez_compressed(out, x_mw=x, q=q, scaling_factors=factors)
        record = {
            "year": year,
            "object": "POST_SOLVE_UNIFORM_EXPOSURE_CONTROLLED_ANNUAL_WITNESS",
            "S": S,
            "E_MWh": E,
            "FDE_h": float(exposure(x, case.time_grid.delta, case.capacities_mw)[0].sum()),
            "uniform_guarded_S": uniform_S,
            "witness_minus_uniform_S": S - uniform_S,
            "positions_scaled_down": int(np.count_nonzero(factors < 1)),
            "max_energy_excess_vs_uniform_MWh_exact": max(0.0, float(max(excess))),
            "energy_inequalities_passed": len(excess),
            "strict_weighted_physical_box_pass": audit["strict_pass"],
            "source_state_sha256": sha256_file(D / "GUARDED_ANNUAL_WITNESS.npz"),
            "reference_uniform_sha256": sha256_file(U / "GUARDED_ANNUAL_WITNESS.npz"),
            "max_action_change_MW": float(abs(source - x).max()),
        }
        witness_record = {
            **record,
            "audit": audit,
            "exact_position_energy_excess_MWh": [float(e) for e in excess],
            "factors": factors.tolist(),
            "chronological_policy": False,
        }
        record_path = witness_folder / "EXPOSURE_CONTROLLED_ANNUAL_WITNESS.json"
        if record_path.is_file():
            require(read(record_path) == witness_record, "Existing exposure witness record differs")
        else:
            atomic_write_json(record_path, witness_record)
        rows.append(record)
    pd.DataFrame(rows).to_csv(destination / "exposure_controlled_witness_results.csv", index=False)


if __name__ == "__main__":
    main()
