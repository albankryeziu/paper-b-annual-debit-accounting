"""Retrospective position-wise M1-energy-matched rolling M2 extension.

No model or solver code is changed. Run this file with --help for entry points.
The archive SHA256SUMS remains the authority for original files; all new evidence
lives in RESULTS/m1_exposure_matched and has its own manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from jobs.certificates import conservative_m2_witness
from tpwrs.contracts import AllowanceSpec, ExperimentCase, SolverProfile, TimeGrid
from tpwrs.execution import annual_solver_fingerprint
from tpwrs.persistence import load_physical_case, load_policy_solution
from tpwrs.replay import physical_objectives
from tpwrs.rolling import solve_rolling
from tpwrs.storage import atomic_write_json, sha256_array, sha256_file, sha256_json

POLICY = "M2_MATCHED_M1_FDE_ROLLING"
YEARS = (2029, 2030, 2031)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exposure(x, delta, capacities):
    """Accurately sum time-major MW arrays; retain signed source numerics."""
    x, delta, capacities = map(lambda a: np.asarray(a, float), (x, delta, capacities))
    require(x.shape == (len(delta), len(capacities)), "Exposure axes differ")
    require(np.isfinite(x).all() and np.isfinite(delta).all() and
            np.isfinite(capacities).all() and np.all(delta > 0) and
            np.all(capacities > 0), "Nonfinite exposure or invalid duration/capacity")
    energy = np.array([math.fsum((delta*x[:, n]).tolist()) for n in range(len(capacities))])
    fde = np.array([math.fsum((delta*x[:, n]/capacities[n]).tolist()) for n in range(len(capacities))])
    # Each product/division is rounded, followed by correctly rounded summation.
    magnitude = np.array([math.fsum(abs(delta*x[:, n])) for n in range(len(capacities))])
    enclosure = 8*np.finfo(float).eps*magnitude
    require(np.all(abs(energy-capacities*fde) <= enclosure), "Energy/FDE identity failed")
    return fde, energy, enclosure


def exact_energy(x, delta):
    """Exact rational sum of the binary64 coordinates, including products.

    Used for ceilings, not for network evaluation. No decimal display tolerance
    can accept a positive excess. The hourly production grid makes dt=1 exact.
    """
    result = []
    for n in range(x.shape[1]):
        result.append(sum((Fraction(float(v))*Fraction(float(dt))
                           for v, dt in zip(x[:, n], delta) if v != 0), Fraction(0)))
    return result


def ceiling_audit(case, x, limits, reference_energy):
    d, e, _ = exposure(x, case.time_grid.delta, case.capacities_mw)
    exact = exact_energy(x, case.time_grid.delta)
    dr = [v/Fraction(float(p))-Fraction(float(h))
          for v, p, h in zip(exact, case.capacities_mw, limits)]
    er = [v-r for v, r in zip(exact, reference_energy)]
    box = max(float(np.max(-x)), float(np.max(x-case.xbar_mw)), 0.0)
    passed = box == 0 and all(v <= 0 for v in dr+er)
    return {"strict_pass": passed, "max_action_box_violation_MW": box,
            "max_budget_residual_h": float(max(dr)),
            "max_energy_excess_MWh": float(max(er)),
            "maximum_position_budget_violation_h": max(0.0, float(max(dr))),
            "maximum_position_energy_excess_MWh": max(0.0, float(max(er))),
            "budget_violating_positions": sum(v > 0 for v in dr),
            "energy_violating_positions": sum(v > 0 for v in er),
            "arithmetic": "exact rational sums of saved binary64 MW and duration coordinates",
            "exact_budget_residual_h": [float(v) for v in dr],
            "exact_energy_excess_MWh": [float(v) for v in er],
            "FDE_h": d.tolist(), "energy_MWh": e.tolist()}


def guarded_witness(case, raw, limits, reference_energy):
    # Projection onto the common action box is separately recorded; it is needed
    # before invoking the existing nonnegative radial certificate routine.
    boxed = np.clip(raw, 0.0, case.xbar_mw)
    witness = conservative_m2_witness(case, boxed, limits)
    x = witness["witness_x_mw"].copy()
    # Additionally enforce the exact saved-source energy ceiling. Usually the
    # existing conservative guard already supplies much more than this margin.
    for n, e in enumerate(exact_energy(x, case.time_grid.delta)):
        if e > reference_energy[n]:
            require(reference_energy[n] >= 0, "Negative source energy ceiling")
            factor = np.nextafter(float(reference_energy[n]/e), 0.0)
            x[:, n] *= factor
    audit = ceiling_audit(case, x, limits, reference_energy)
    require(audit["strict_pass"], "Guard did not certify strict ceilings")
    return x, {"box_projection_max_change_MW": float(np.max(abs(raw-boxed))),
               "radial_factors": witness["witness_scaling_factors"].tolist(),
               "guard_hours": witness["debit_roundoff_guard_hours"].tolist(),
               "max_total_change_MW": float(np.max(abs(raw-x))), **audit}


def experiment_from_json(path):
    obj = read(path)
    d = obj["experiment"].copy()
    a = d["allowance"].copy()
    a["participant_ids"] = tuple(a["participant_ids"])
    a["values_hours"] = tuple(a["values_hours"])
    d["allowance"] = AllowanceSpec(**a)
    d["solver_profile"] = SolverProfile(**d["solver_profile"])
    exp = ExperimentCase(**d)
    require(exp.hash == obj["experiment_hash"], "Frozen experiment hash mismatch")
    return exp


def validate_identity(case, exp, year, rule="M1"):
    case.verify_experiment_identity(exp.physical_case_id, exp.physical_case_hash, exp.time_grid_id)
    require(tuple(exp.allowance.participant_ids) == case.participant_ids, "Position ordering mismatch")
    require(exp.rule == rule and exp.mode == "rolling", "Wrong source policy")
    require(exp.allowance.nominal_rho == 0.15, "Wrong source allowance")
    require(exp.lookahead_hours == 48 and exp.implementation_hours == 24, "Wrong rolling horizons")
    grid = TimeGrid.uniform_hourly(year)
    require(case.time_grid.timestamps_utc == grid.timestamps_utc and
            np.array_equal(case.time_grid.delta, grid.delta), "Wrong year/UTC chronology")
    nominal = AllowanceSpec.from_nominal(case.participant_ids, case.contract_mask, case.time_grid, 0.15)
    require(exp.allowance.hash == nominal.hash, "Source nominal allowance contract differs")


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.expected = {}
        for line in (self.root/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(None, 1)
            self.expected[name.strip().replace("\\", "/")] = digest
        self.checked = {}

    def verify(self, path):
        path = Path(path).resolve()
        name = path.relative_to(self.root).as_posix()
        require(name in self.expected, f"Original source absent from archive inventory: {name}")
        digest = sha256_file(path)
        require(digest == self.expected[name], f"Archived original was altered: {name}")
        self.checked[name] = digest
        return digest

    def preservation(self):
        require(all(sha256_file(self.root/p) == digest for p, digest in self.checked.items()),
                "An original input/evidence file changed during experiment")
        return {"status": "PASS", "checked_original_files": len(self.checked),
                "files": self.checked}


def epoch_audit(case, exp, folder, x, u=None, archive=None, expected_hashes=None):
    """Independently reconstruct proposals, committed debit and clipping drift."""
    balance = exp.allowance.values.copy()
    exact_debits = np.zeros(case.n_participants)
    records = []
    for i, start in enumerate(range(0, case.n_time, 24)):
        stop = min(start+48, case.n_time)
        end = min(start+24, case.n_time)
        p = folder/"epochs"/f"epoch_{start:06d}"
        if archive:
            archive.verify(p/"solution.json")
            archive.verify(p/"solution_arrays.npz")
        sub = case.slice_time(start, stop)
        a = AllowanceSpec.explicit(case.participant_ids, balance, source="explicit",
                                   nominal_rho=exp.allowance.nominal_rho)
        se = replace(exp, case_id=f"{exp.case_id}__epoch_{start:06d}",
                     physical_case_id=sub.physical_case_id, physical_case_hash=sub.physical_case_hash,
                     time_grid_id=sub.time_grid.grid_id, allowance=a, mode="annual",
                     lookahead_hours=None, implementation_hours=None,
                     result_root_id=f"{exp.result_root_id}/epochs/epoch_{start:06d}")
        sol = load_policy_solution(p, expected_experiment_hash=se.hash,
                                   expected_physical_case_hash=sub.physical_case_hash)
        if expected_hashes is not None:
            require(sol.solution_hash == expected_hashes[i], "Source epoch hash differs from annual registry")
        require(sol.stage1.status == sol.stage2.status == "OPTIMAL" and
                sol.replay_summary.get("passed"), "Epoch numerical acceptance failed")
        k = end-start
        require(np.array_equal(sol.x_mw[:k], x[start:end]), "Implemented arrays differ from proposals")
        if u is not None:
            require(np.array_equal(sol.u[:k], u[start:end]), "Implemented activations differ")
            proposal = np.sum(sub.time_grid.delta[:, None]*sol.u, axis=0)
            committed = np.sum(sub.time_grid.delta[:k, None]*sol.u[:k], axis=0)
        else:
            proposal = exposure(sol.x_mw, sub.time_grid.delta, case.capacities_mw)[0]
            committed = exposure(sol.x_mw[:k], sub.time_grid.delta[:k], case.capacities_mw)[0]
        excess = float(np.max(proposal-balance))
        require(excess <= 10*exp.solver_profile.feasibility_tol,
                f"Prospective ledger exceeds existing numerical contract: {excess}")
        before = balance.copy()
        # Reproduce the existing controller's binary64 reduction and clipping;
        # also retain a separate fsum ledger without clipping.
        original_debit = np.sum(sub.time_grid.delta[:k, None]*(sol.u[:k] if u is not None
                              else sol.x_mw[:k]/case.capacities_mw), axis=0)
        unclipped = balance-original_debit
        balance = np.maximum(unclipped, 0.0)
        exact_debits += committed
        for n, pid in enumerate(case.participant_ids):
            records.append({"epoch": start, "position_id": pid, "balance_before_h": before[n],
                            "lookahead_debit_h": proposal[n], "implemented_debit_h": committed[n],
                            "prospective_residual_h": proposal[n]-before[n],
                            "balance_after_h": balance[n], "clipping_h": max(-unclipped[n], 0.0),
                            "independent_unclipped_balance_h": exp.allowance.values[n]-exact_debits[n]})
    return balance, pd.DataFrame(records)


def source_data(archive, case, year, out):
    folder = archive.root/f"RESULTS/rolling/m1_m2/year_{year}/rho_0.15/rolling/M1"
    for name in ("EXPERIMENT_CASE.json", "rolling_result.json", "rolling_arrays.npz"):
        archive.verify(folder/name)
    exp = experiment_from_json(folder/"EXPERIMENT_CASE.json")
    validate_identity(case, exp, year)
    meta = read(folder/"rolling_result.json")
    require(meta["experiment_hash"] == exp.hash and meta["physical_case_hash"] == case.physical_case_hash
            and meta["allowance_hash"] == exp.allowance.hash and meta["status"] == "PASS"
            and meta["complete_coverage"], "Source rolling evidence identity/acceptance mismatch")
    require(sha256_file(folder/"rolling_arrays.npz") == meta["arrays_sha256"], "Source array hash mismatch")
    with np.load(folder/"rolling_arrays.npz", allow_pickle=False) as z:
        x, u, final = z["x_mw"], z["u"], z["final_balance_hours"]
    require(x.shape == u.shape == case.requests_mw.shape and np.isfinite(x).all()
            and np.isfinite(u).all(), "Source action coordinates invalid")
    require(meta["covered_intervals"] == [[s, min(s+24, case.n_time)] for s in range(0, case.n_time, 24)],
            "Missing or duplicate implemented hours")
    fde, energy, identity_bound = exposure(x, case.time_grid.delta, case.capacities_mw)
    require(np.all(fde >= 0) and np.all(energy >= 0), "Source has negative annual exposure")
    zero = np.all(x == 0, axis=0)
    require(np.all(fde[zero] == 0), "Zero source exposure changed")
    touched = np.array([math.fsum(case.time_grid.delta*u[:, n]) for n in range(case.n_participants)])
    violations = {"nonnegativity_MW": float(max(0, np.max(-x))),
                  "box_MW": float(max(0, np.max(x-case.xbar_mw))),
                  "upper_link_MW": float(max(0, np.max(x-case.xbar_mw*u))),
                  "material_floor_MW": float(max(0, np.max(case.materiality_floor_mw*u-x))),
                  "activation_integrality": float(np.max(abs(u-np.rint(u)))),
                  "activation_range": float(max(0, np.max(-u), np.max(u-1))),
                  "duration_h": float(max(0, np.max(touched-exp.allowance.values)))}
    require(max(violations.values()) <= 10*exp.solver_profile.feasibility_tol,
            f"Source exceeds documented replay contract: {violations}")
    balance, ledger = epoch_audit(case, exp, folder, x, u, archive, meta["epoch_solution_hashes"])
    require(np.array_equal(balance, final), "Source final ledger mismatch")
    s, e = physical_objectives(case, x)
    require(abs(s-meta["S_mw_h_row"]) <= 64*np.finfo(float).eps*max(1, abs(s)),
            "Source full-row residual differs from archive")
    exact = exact_energy(x, case.time_grid.delta)
    reference_residual = [float(v/Fraction(float(p))-Fraction(float(h)))
                          for v, p, h in zip(exact, case.capacities_mw, fde)]
    audit = {"status": "PASS_WITH_DOCUMENTED_SOURCE_NUMERICS", "year": year,
             "units": {"x": "MW", "delta": "h", "capacity": "MW", "FDE": "h"},
             "source_experiment_hash": exp.hash, "source_array_hash": sha256_array(x),
             "physical_case_hash": case.physical_case_hash, "time_grid_hash": case.time_grid.hash,
             "participant_order_hash": sha256_json(case.participant_ids),
             "capacity_hash": sha256_array(case.capacities_mw), "violations": violations,
             "energy_FDE_identity_enclosure_MWh": identity_bound.tolist(),
             "reference_matched_budget_roundoff_residual_h": reference_residual,
             "reference_exact_action_box_feasible": max(violations["nonnegativity_MW"], violations["box_MW"]) == 0,
             "annual_inclusion": "Exact for mathematical feasible M1; saved source only numerically feasible",
             "replayed_S": s, "energy_MWh": e, "total_FDE_h": math.fsum(fde),
             "maximum_source_prospective_residual_h": float(ledger.prospective_residual_h.max()),
             "zero_positions": int(zero.sum())}
    (out/"source_audits").mkdir(parents=True, exist_ok=True)
    atomic_write_json(out/f"source_audits/{year}.json", audit)
    ledger.to_csv(out/f"source_audits/{year}_ledger.csv", index=False)
    return exp, x, fde, energy, exact, audit


def code_identity():
    return {p.relative_to(SOURCE).as_posix(): sha256_file(p)
            for p in sorted(list((SOURCE/"tpwrs").glob("*.py"))+list((SOURCE/"jobs").glob("*.py")))}


def export_tables(root, out, years):
    main = pd.read_csv(root/"RESULTS/final_evidence/MAIN_POLICY_RESULTS.csv")
    rows, participants, differences = [], [], []
    for year in years:
        old = main[(main.year == year) & (main.rho == .15) & (main["mode"] == "rolling")]
        for rule in ("M1", "M2"):
            r = old[old.rule == rule].iloc[0]
            rows.append({"year": year, "policy": "Rolling "+rule, "object": "ARCHIVED_REFERENCE",
                         "replayed_S": r.S_mw_h_row, "total_restricted_energy_MWh": r.E_restricted_mwh,
                         "total_FDE_h": r.physical_fde_sum_h,
                         "energy_ceiling_audit": "Reference" if rule == "M1" else "Not matched"})
        path = out/f"year_{year}/result.json"
        if not path.is_file():
            rows.append({"year": year, "policy": POLICY, "object": "NOT_COMPLETED",
                         "energy_ceiling_audit": "Unavailable"})
            continue
        r = read(path)
        rows.extend(r["policies"])
        participants.extend(r["participants"])
        a = float(old[old.rule == "M1"].iloc[0].S_mw_h_row)
        b = float(old[old.rule == "M2"].iloc[0].S_mw_h_row)
        for p in r["policies"]:
            differences.append({"year": year, "object": p["object"],
                                "ordinary_M2_improvement_percent": 100*(a-b)/a,
                                "matched_improvement_percent": 100*(a-p["replayed_S"])/a,
                                "matched_delta_S": a-p["replayed_S"],
                                "original_improvement_retained_percent": 100*(a-p["replayed_S"])/(a-b),
                                "max_energy_excess_MWh": p["max_energy_excess_MWh"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"policy_results.csv", index=False)
    pd.DataFrame(participants).to_csv(out/"participant_exposure.csv", index=False)
    pd.DataFrame(differences).to_csv(out/"comparison_differences.csv", index=False)
    lines = ["| Year | Policy / object | S (million) | E (TWh) | FDE (thousand h) | Energy ceiling |",
             "|---|---|---:|---:|---:|---|"]
    tex = [r"\begin{tabular}{llrrrl}", r"\toprule Year & Policy & $S/10^6$ & $E/10^6$ & FDE/$10^3$ & Audit \\", r"\midrule"]
    for r in rows:
        v = [r.get("replayed_S", float("nan"))/1e6,
             r.get("total_restricted_energy_MWh", float("nan"))/1e6,
             r.get("total_FDE_h", float("nan"))/1e3]
        label = r["policy"] if r["object"] == "ARCHIVED_REFERENCE" else "Matched M2 / "+r["object"]
        display = [f"{z:.6f}" if math.isfinite(z) else "unavailable" for z in v]
        lines.append(f"| {r['year']} | {label} | "+" | ".join(display)+f" | {r['energy_ceiling_audit']} |")
        tex.append(f"{r['year']} & {label.replace('_', ' ')} & "+" & ".join(display)+f" & {r['energy_ceiling_audit']} "+r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out/"results_table.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    (out/"results_summary.md").write_text("\n".join(lines)+"\n\nRetrospective budgets use completed M1 operation. Raw rolling actions and guarded annual witnesses are distinct. Source M1 arrays have documented numerical action violations; exact annual inclusion applies to mathematical feasible M1 actions. No causal online-policy or annual optimality ordering is claimed.\n", encoding="utf-8")


def run_year(archive, case, year, out, data, implementation_hash, mode):
    _, source_x, limits, source_energy, exact_source, source_audit = data
    folder = out/f"year_{year}"
    folder.mkdir(exist_ok=True)
    old_folder = archive.root/f"RESULTS/rolling/m1_m2/year_{year}/rho_0.15/rolling/M2"
    archive.verify(old_folder/"EXPERIMENT_CASE.json")
    template = experiment_from_json(old_folder/"EXPERIMENT_CASE.json")
    validate_identity(case, template, year, "M2")
    allowance = AllowanceSpec.explicit(case.participant_ids, limits, source="explicit", nominal_rho=.15,
                                        source_solution_hash=sha256_array(source_x))
    exp = replace(template, case_id=f"Y{year}_{POLICY}", allowance=allowance,
                  information_level="RETROSPECTIVE_REALIZED_M1_FDE__48H_SCENARIO_LOOKAHEAD",
                  result_root_id=f"m1_exposure_matched/year_{year}", implementation_hash=implementation_hash)
    target = folder/"EXPERIMENT_CASE.json"
    if target.exists():
        require(read(target)["experiment_hash"] == exp.hash, "Extension experiment changed; use a fresh output root")
    else:
        atomic_write_json(target, {"experiment": asdict(exp), "experiment_hash": exp.hash})
    started = time.perf_counter()
    if mode == "run":
        result = solve_rolling(case, exp, use_stage3=False, checkpoint_root=folder/"epochs")
        require(result.complete_coverage, "Incomplete new rolling chronology")
        raw, final = result.x_mw, result.final_balance_hours
        np.savez_compressed(folder/"RAW_ROLLING_POLICY.npz", x_mw=raw, final_balance_hours=final)
        wall = time.perf_counter()-started
        atomic_write_json(folder/"runtime.json", {"wall_time_seconds": wall})
    else:
        with np.load(folder/"RAW_ROLLING_POLICY.npz", allow_pickle=False) as z:
            raw, final = z["x_mw"], z["final_balance_hours"]
        wall = read(folder/"runtime.json")["wall_time_seconds"]
    balance, ledger = epoch_audit(case, exp, folder, raw)
    require(np.array_equal(balance, final), "New carried ledger mismatch")
    ledger.to_csv(folder/"ledger_audit.csv", index=False)
    raw_audit = ceiling_audit(case, raw, limits, exact_source)
    guarded, guard = guarded_witness(case, raw, limits, exact_source)
    policies, participants = [], []
    objects = [("RAW_ROLLING_POLICY", raw, raw_audit)]
    if not np.array_equal(raw, guarded):
        np.savez_compressed(folder/"GUARDED_FEASIBLE_WITNESS.npz", x_mw=guarded)
        objects.append(("GUARDED_FEASIBLE_WITNESS", guarded, guard))
    for name, x, audit in objects:
        s, _ = physical_objectives(case, x)
        d, e, _ = exposure(x, case.time_grid.delta, case.capacities_mw)
        atomic_write_json(folder/f"{name}_replay.json", {**audit, "replayed_S": s,
                          "full_row_count": case.n_flat_rows, "policy_hash": sha256_array(x),
                          "source_numerics": source_audit["violations"]})
        policies.append({"year": year, "policy": POLICY, "object": name, "source_policy": "rolling M1",
                         "rho_source": .15, "rolling_lookahead_h": 48, "rolling_implemented_h": 24,
                         "replayed_S": s, "total_restricted_energy_MWh": math.fsum(e),
                         "total_FDE_h": math.fsum(d), "number_of_matching_positions": case.n_participants,
                         "number_of_zero_allowance_positions": int(np.sum(limits == 0)),
                         "solver_status": "ALL_EPOCHS_STAGE1_STAGE2_OPTIMAL", "wall_time_seconds": wall,
                         "replay_status": "COMPLETE_8837_ROWS",
                         "energy_ceiling_audit": "Verified" if audit["strict_pass"] else "Failed strict audit",
                         "maximum_position_budget_violation_h": audit["maximum_position_budget_violation_h"],
                         "maximum_position_energy_excess_MWh": audit["maximum_position_energy_excess_MWh"],
                         "max_energy_excess_MWh": audit["max_energy_excess_MWh"],
                         "max_budget_residual_h": audit["max_budget_residual_h"]})
        for n, pid in enumerate(case.participant_ids):
            participants.append({"year": year, "object": name, "position_id": pid,
                                 "contracted_capacity_MW": case.capacities_mw[n],
                                 "original_M1_allowance_h": template.allowance.values[n],
                                 "M1_realized_FDE_h": limits[n], "matched_M2_allowance_h": limits[n],
                                 "M1_restricted_energy_MWh": source_energy[n],
                                 "matched_M2_restricted_energy_MWh": e[n],
                                 "M1_minus_matched_energy_MWh": source_energy[n]-e[n],
                                 "matched_M2_realized_FDE_h": d[n], "matched_budget_residual_h": d[n]-limits[n],
                                 "exact_matched_budget_residual_h": audit["exact_budget_residual_h"][n],
                                 "exact_matched_energy_excess_MWh": audit["exact_energy_excess_MWh"][n]})
    atomic_write_json(folder/"guard.json", guard)
    atomic_write_json(folder/"result.json", {"policies": policies, "participants": participants,
                      "experiment_hash": exp.hash, "maximum_prospective_residual_h": float(ledger.prospective_residual_h.max()),
                      "sum_ledger_clipping_h": float(ledger.clipping_h.sum())})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    require(len(set(args.years)) == len(args.years) and set(args.years) <= set(YEARS), "Only the three evaluation years are permitted")
    archive = Archive(args.root)
    out = (args.output or archive.root/"RESULTS/m1_exposure_matched").resolve()
    require(out != archive.root and not any(out == archive.root/p for p in archive.expected), "Unsafe extension destination")
    require(not (out.is_relative_to(archive.root/"IMPLEMENTATION")), "Results cannot overwrite implementation")
    if out.is_relative_to(archive.root):
        require(out.is_relative_to(archive.root/"RESULTS/m1_exposure_matched"),
                "Within the archive only the dedicated extension output directory is writable")
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        export_tables(archive.root, out, args.years)
        return 0
    inputs = archive.root/"IMPLEMENTATION/frozen_inputs/artifacts/real_case/physical_case/FINAL_DUTCH_CASE"
    for name in ("physical_case.json", "physical_case_arrays.npz"):
        archive.verify(inputs/name)
    for name in ("FINAL_EVIDENCE_MANIFEST.json", "CLAIM_GATE.json", "MAIN_POLICY_RESULTS.csv", "M1_CERTIFICATE_RESULTS.csv", "M3_MATCHED_DECOMPOSITION.csv"):
        archive.verify(archive.root/"RESULTS/final_evidence"/name)
    parent = load_physical_case(inputs)
    require(parent.n_participants == 114 and len(parent.bus_ids) == 45 and parent.n_flat_rows == 8837
            and len(parent.network_states) == 94, "Frozen physical dimensions changed")
    expected_solver = read(archive.root/"ARCHIVE_MANIFEST.json")["annual_solver_fingerprint"]
    require(annual_solver_fingerprint(SOURCE) == expected_solver, "Annual solver implementation differs from qualified archive")
    code = code_identity()
    for relative in code:
        if relative != "jobs/m1_exposure_matched.py":
            archive.verify(SOURCE/relative)
    profile_path = archive.root/"IMPLEMENTATION/config/artifacts/performance/SELECTED_SOLVER_PROFILE.json"
    archive.verify(profile_path)
    profile_record = read(profile_path)
    require(profile_record["status"] == "PASS" and profile_record["annual_solver_fingerprint"] == expected_solver
            and profile_record["parent_physical_case_hash"] == parent.physical_case_hash,
            "Archived solver qualification identity differs")
    implementation_hash = sha256_json(code)
    manifest = {"policy": POLICY, "years": args.years, "mode": "dry-run" if args.dry_run else "verify" if args.verify else "run",
                "status": "RUNNING", "retrospective": True, "implementation_hash": implementation_hash,
                "source_files": code, "annual_solver_fingerprint": expected_solver,
                "qualified_solver_profile": profile_record,
                "original_scientific_fingerprint": read(archive.root/"ARCHIVE_MANIFEST.json")["scientific_fingerprint"],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "parent_physical_case_hash": parent.physical_case_hash}
    if args.run:
        import gurobipy as gp
        manifest["runtime"]["gurobi"] = list(gp.gurobi.version())
        require(gp.gurobi.version() == (12, 0, 3), "Frozen Gurobi version required")
    atomic_write_json(out/"experiment_manifest.json", manifest)
    try:
        for year in args.years:
            print(f"VALIDATING SOURCE {year}", flush=True)
            ids = np.flatnonzero(parent.time_grid.index.year == year)
            case = parent.slice_time(int(ids[0]), int(ids[-1])+1)
            data = source_data(archive, case, year, out)
            (out/"matched_allowances").mkdir(exist_ok=True)
            atomic_write_json(out/f"matched_allowances/{year}.json", {
                "year": year, "position_ids": case.participant_ids, "allowance_h": data[2].tolist(),
                "allowance_hash": sha256_array(data[2]), "source_x_hash": sha256_array(data[1]),
                "physical_case_hash": case.physical_case_hash, "units": "FDE hours"})
            if not args.dry_run:
                print(f"{'VERIFY' if args.verify else 'RUN'} {year} {POLICY}", flush=True)
                run_year(archive, case, year, out, data, implementation_hash, "verify" if args.verify else "run")
        manifest["status"] = "DRY_RUN_PASS" if args.dry_run else "COMPLETE"
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_write_json(out/"original_preservation.json", archive.preservation())
        export_tables(archive.root, out, args.years)
        manifest["outputs"] = {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob("*")
                               if p.is_file() and p.name not in {"experiment_manifest.json", "run.log", "verify.log", "dry_run.log"}}
        atomic_write_json(out/"experiment_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
