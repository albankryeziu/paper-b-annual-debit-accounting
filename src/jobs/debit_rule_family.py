"""Final relief-premium fit, validation gate and frozen held-out campaign.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import gc
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
from jobs.m1_exposure_matched import Archive, read, exposure, exact_energy, experiment_from_json
from tpwrs.cards import rate_card_qp
from tpwrs.contracts import AllowanceSpec, ExperimentCase
from tpwrs.debit_rules import (
    DebitRule,
    require,
    sums,
    hinge_relief_scores,
    weighted_quantile,
    buckets,
    premium_card,
    admissible_probe_checks,
    weighted_budget_audit,
    weighted_witness,
)
from tpwrs.execution import annual_solver_fingerprint
from tpwrs.persistence import load_physical_case, load_policy_solution
from tpwrs.replay import physical_objectives
from tpwrs.rolling import solve_rolling
from tpwrs.storage import atomic_write_json, sha256_file, sha256_array, sha256_json

FIT = (2025, 2026, 2027)
DESIGN = dict(gamma=0.5, xi=0.25, penalty=1.0)
GATE = dict(
    minimum_relative_improvement=0.001,
    minimum_absolute_improvement=1.0,
    require_strict_guarded_feasibility=True,
)


def year_case(parent, year):
    ids = np.flatnonzero(parent.time_grid.index.year == year)
    require(len(ids) in (8760, 8784) and np.all(np.diff(ids) == 1), "Year coverage differs")
    return parent.slice_time(int(ids[0]), int(ids[-1]) + 1)


def code_hashes():
    return {
        str(p.relative_to(SOURCE)).replace("\\", "/"): sha256_file(p)
        for p in (Path(__file__), SOURCE / "tpwrs/debit_rules.py")
    }


def checked_json(archive, path):
    archive.verify(path)
    return read(path)


def teacher_folder(root, year):
    base = root / "RESULTS/provenance/card_fitting_teacher/rho_0.15/teacher_experiments"
    found = list(base.glob(f"*/teacher_solution_{year}/solution.json"))
    require(len(found) == 1, "Teacher source not unique")
    return found[0].parent


def original_m3_audit(archive, parent, out):
    """Rebuild archived bucket aggregates and tiny QPs, never network policies."""
    import joblib

    base = archive.root / "IMPLEMENTATION/frozen_inputs/artifacts"
    cards = {
        m: checked_json(archive, base / f"cards/rho_0.15/CARD_{m}.json")
        for m in ("ridge", "histgb")
    }
    agg = {
        m: {
            d: {
                "A": np.zeros(3),
                "W": np.zeros(3),
                "count": np.zeros(3, int),
                "positive_count": np.zeros(3, int),
            }
            for d in ("import", "export")
        }
        for m in cards
    }
    teacher_checks = []
    for year in FIT:
        print(f"M3 AUDIT {year}", flush=True)
        case = year_case(parent, year)
        path = base / f"cards/rho_0.15/teacher_rows_{year}.parquet"
        archive.verify(path)
        frame = pd.read_parquet(path)
        ti, ni = frame.time_index.to_numpy(int), frame.participant_index.to_numpy(int)
        require(len(frame) == case.n_time * case.n_participants, "Teacher coverage")
        require(
            np.array_equal(frame.participant_id.to_numpy(), np.asarray(case.participant_ids)[ni]),
            "Teacher IDs",
        )
        require(
            np.array_equal(frame.capacity_mw.to_numpy(), case.capacities_mw[ni]),
            "Teacher capacity alignment",
        )
        folder = teacher_folder(archive.root, year)
        for name in ("solution.json", "solution_arrays.npz", "EXPERIMENT_CASE.json"):
            archive.verify(folder / name)
        sol = load_policy_solution(folder, expected_physical_case_hash=case.physical_case_hash)
        require(
            sol.solution_hash == cards["ridge"]["card"]["source_hashes"][f"teacher_{year}"],
            "Teacher identity",
        )
        require(
            np.array_equal(frame.x_m2_mw.to_numpy(), sol.x_mw[ti, ni]), "Teacher action alignment"
        )
        mu = np.array([sol.m2_budget_duals[p] for p in case.participant_ids])
        require(
            np.array_equal(frame.mu_m2_stage2.to_numpy(), mu[ni]), "Teacher multiplier alignment"
        )
        weight = (
            frame.delta_t_hours.to_numpy() * frame.x_m2_mw.to_numpy() / frame.capacity_mw.to_numpy()
        )
        alpha_error = float(np.max(abs(frame.alpha.to_numpy() - mu[ni] * weight)))
        exp = experiment_from_json(folder / "EXPERIMENT_CASE.json")
        fde = exposure(sol.x_mw, case.time_grid.delta, case.capacities_mw)[0]
        complement = mu * (exp.allowance.values - fde)
        teacher_checks.append(
            {
                "year": year,
                "solution_hash": sol.solution_hash,
                "stage1_status": sol.stage1.status,
                "stage2_status": sol.stage2.status,
                "replay_passed": bool(sol.replay_summary.get("passed")),
                "positive_multiplier_positions": int(np.count_nonzero(mu > 0)),
                "multiplier_min": float(mu.min()),
                "multiplier_max": float(mu.max()),
                "multiplier_unique": int(len(np.unique(mu))),
                "alpha_identity_max": alpha_error,
                "max_abs_complementarity": float(np.max(abs(complement))),
                "max_scaled_complementarity": float(
                    np.max(abs(complement) / (1 + abs(mu) * np.maximum(1, exp.allowance.values)))
                ),
                "max_fde_budget_residual": float(np.max(fde - exp.allowance.values)),
                "stage2_dual_certificate": dict(sol.stage2.metadata).get("dual_certificate"),
                "negative_reference_rows": int(np.count_nonzero(weight < 0)),
                "negative_reference_FDE": float(weight[weight < 0].sum()),
            }
        )
        require(
            alpha_error <= 1e-10 * max(1, float(abs(frame.alpha).max())),
            "Alpha attribution mismatch",
        )
        fp = base / f"features/F0_{year}.parquet"
        if not fp.is_file():
            candidates = list((base / "features").glob(f"*{year}*.parquet"))
            require(len(candidates) == 1, "F0 path not unique")
            fp = candidates[0]
        archive.verify(fp)
        features = pd.read_parquet(fp)
        # Frozen F0 indices are parent-grid global; teacher indices are year-local.
        offset = int(features.time_index.iloc[0])
        require(
            np.array_equal(
                features.time_index.to_numpy() - offset,
                np.repeat(np.arange(case.n_time), case.n_participants),
            )
            and np.array_equal(
                features.participant_id.to_numpy(), np.tile(case.participant_ids, case.n_time)
            ),
            "F0 axes",
        )
        require(
            np.array_equal(
                pd.to_datetime(
                    features.timestamp_utc.iloc[:: case.n_participants], utc=True
                ).to_numpy(),
                case.time_grid.index.to_numpy(),
            ),
            "F0 timestamp alignment",
        )
        for model, record in cards.items():
            mp = base / f"cards/rho_0.15/MODEL_{model}.joblib"
            archive.verify(mp)
            bundle = joblib.load(mp)
            pred = bundle.predict(features).reshape(case.requests_mw.shape)[ti, ni]
            for direction, a in agg[model].items():
                hit = frame.direction.eq(direction).to_numpy()
                bi = buckets(pred, record["card"]["thresholds"][direction])
                for b in range(3):
                    mask = hit & (bi == b)
                    a["A"][b] += float(frame.alpha.to_numpy()[mask].sum())
                    a["W"][b] += float(weight[mask].sum())
                    a["count"][b] += int(mask.sum())
                    a["positive_count"][b] += int(np.count_nonzero(mask & (weight > 0)))
            del bundle, pred
        del features, frame, sol, case
        gc.collect()
    records = []
    qp_checks = []
    for model, directions in agg.items():
        card = cards[model]["card"]
        for direction, a in directions.items():
            q, mode = rate_card_qp(
                a["A"], a["W"], q_bounds=tuple(card["q_bounds"]), lambda_q=card["lambda_q"]
            )
            stored = np.asarray(card["rates"][direction])
            require(np.max(abs(q - stored)) <= 1e-7, "Reconstructed original QP differs")
            ratio = np.divide(a["A"], a["W"], out=np.full(3, np.nan), where=a["W"] > 0)
            # Gradient projected to neutrality plane at stored q. A constant
            # A/W makes the entire linear objective constant, not merely flat.
            gradient = a["A"] + card["lambda_q"] * a["W"] * (stored - 1)
            wn = a["W"] / a["W"].sum() if a["W"].sum() > 0 else a["W"]
            projected = gradient - wn * (wn @ gradient) / (wn @ wn) if wn @ wn else gradient
            qp_checks.append(
                {
                    "model": model,
                    "direction": direction,
                    "mode": mode,
                    "raw_recomputed_q": q.tolist(),
                    "stored_q": stored.tolist(),
                    "positive_weight_buckets": int(np.count_nonzero(a["W"] > 0)),
                    "positive_A_over_W_range": float(np.ptp(ratio[a["W"] > 0]))
                    if np.any(a["W"] > 0)
                    else None,
                    "neutrality_residual": float(a["W"] @ (q - 1)),
                    "projected_gradient_norm_at_stored_q": float(np.linalg.norm(projected)),
                    "note": "Projected gradient is not a KKT residual when inequality bounds/order are active.",
                }
            )
            for b in range(3):
                records.append(
                    {
                        "model": model,
                        "direction": direction,
                        "bucket": b + 1,
                        "A": a["A"][b],
                        "W_FDE": a["W"][b],
                        "A_over_W": ratio[b],
                        "rows": a["count"][b],
                        "positive_reference_rows": a["positive_count"][b],
                        "stored_q": stored[b],
                        "recomputed_q": q[b],
                    }
                )
    pd.DataFrame(records).to_csv(out / "m3_bucket_aggregates.csv", index=False)
    status = checked_json(archive, base / "cards/CARD_BUILD_STATUS.json")
    selected = checked_json(archive, base / "cards/RHO15_SELECTED_CARD.json")
    evidence = {
        "teacher_checks": teacher_checks,
        "qp_checks": qp_checks,
        "card_build_status": status,
        "selected": selected,
        "scarcity_gate": "No additional dynamic scarcity gate in frozen primary builder. Sparse-reference fallback is explicit; rho=.03 is a separate allowance experiment.",
    }
    atomic_write_json(out / "m3_diagnostic.json", evidence)
    return evidence


def prepare(archive, parent, out):
    require(
        (out / "DATA_AND_METHOD_FEASIBILITY.md").is_file()
        and (out / "DEBIT_RULE_FAMILY_THEORY.md").is_file(),
        "Audit and theory must precede preparation",
    )
    if not (out / "m3_diagnostic.json").is_file():
        original_m3_audit(archive, parent, out)
    values, weights, checks = [], [], []
    cache = out / "scores"
    cache.mkdir(exist_ok=True)
    for year in FIT:
        print(f"SCORE FIT {year}", flush=True)
        case = year_case(parent, year)
        scores = hinge_relief_scores(case)
        checks.extend([{"year": year, **r} for r in admissible_probe_checks(case, scores)])
        np.savez_compressed(
            cache / f"fit_{year}.npz", scores=scores, physical_case_hash=case.physical_case_hash
        )
        folder = teacher_folder(archive.root, year)
        archive.verify(folder / "solution.json")
        archive.verify(folder / "solution_arrays.npz")
        sol = load_policy_solution(folder, expected_physical_case_hash=case.physical_case_hash)
        w = case.time_grid.delta[:, None] * np.maximum(sol.x_mw, 0) / case.capacities_mw
        positive = w > 0
        values.append(scores[positive])
        weights.append(w[positive])
        del sol, w, case, scores
        gc.collect()
    values, weights = np.concatenate(values), np.concatenate(weights)
    thresholds = weighted_quantile(values, weights, (1 / 3, 2 / 3))
    bi = buckets(values, thresholds)
    W = np.array([math.fsum(weights[bi == b]) for b in range(3)])
    means = np.array(
        [math.fsum(weights[bi == b] * values[bi == b]) / W[b] if W[b] else 0 for b in range(3)]
    )
    scale = max(1.0, float(abs(means[W > 0]).max()))
    q, diagnostic = premium_card(W, means / scale, **DESIGN)
    card = {
        "rule_id": "CONSERVATIVE_STAGE1_RELIEF_V1",
        "fit_years": list(FIT),
        "validation_year": 2028,
        "thresholds": thresholds.tolist(),
        "rates": q.tolist(),
        "uniform_rate": 1 + DESIGN["xi"],
        "reference_W_FDE": W.tolist(),
        "marginal_relief_means_per_FDE": means.tolist(),
        "score_scale": scale,
        "design": DESIGN,
        "gate": GATE,
        "diagnostic": diagnostic,
        "feature_information": "PREDECLARED_CONTROLLED_INPUT; requested-state right derivative, no future optimized actions",
        "reference_weights": "Fitting-only archived annual M2 positive physical FDE; tiny negative solver coordinates excluded only from calibration weights",
        "buckets": "Pooled fitting exposure-weighted tertiles; directions share physical residual/FDE units",
        "code_hashes": code_hashes(),
        "parent_physical_case_hash": parent.physical_case_hash,
        "input_hashes": dict(archive.checked),
    }
    card["card_hash"] = sha256_json(card)
    atomic_write_json(out / "FITTED_CARD.json", card)
    atomic_write_json(
        out / "score_probe_checks.json", {"checks": checks, "status": "PASS", "count": len(checks)}
    )
    atomic_write_json(out / "preparation_preservation.json", archive.preservation())
    print(
        json.dumps(
            {
                k: card[k]
                for k in (
                    "thresholds",
                    "rates",
                    "reference_W_FDE",
                    "marginal_relief_means_per_FDE",
                    "diagnostic",
                )
            }
        ),
        flush=True,
    )


def metrics(case, x, q, H, year, policy, object_id, mode, runtime, audit=None):
    """Common full-network evaluator plus independent participant accounting."""
    S, E = physical_objectives(case, x)
    peak, material_overload_hours = 0.0, 0.0
    for start in range(0, case.n_time, 64):
        stop = min(start + 64, case.n_time)
        flows = (
            case.requested_state_mw[start:stop] + x[start:stop] @ case.A.T
        ) @ case.flat_H.T + case.flat_h
        overload = np.maximum(abs(flows) - case.flat_ratings, 0.0)
        peak = max(peak, float(overload.max()))
        material_overload_hours += float(
            case.time_grid.delta[start:stop][np.any(overload > 1.0, axis=1)].sum()
        )
    fde, energy, identity_bound = exposure(x, case.time_grid.delta, case.capacities_mw)
    debit = DebitRule("WEIGHTED").debit(case, x, q=q)
    material = (x > 0) & (x >= case.materiality_floor_mw) & (case.xbar_mw > 0)
    deep = x / case.capacities_mw > 0.5
    near = (abs(x - case.materiality_floor_mw) <= 1e-7) & (case.materiality_floor_mw > 0)
    touched = sums(case.time_grid.delta[:, None] * material)
    deep_hours = sums(case.time_grid.delta[:, None] * deep)
    up = np.maximum(np.diff(x, axis=0), 0.0)
    participants = []
    for n, pid in enumerate(case.participant_ids):
        participants.append(
            {
                "year": year,
                "rule": policy,
                "object": object_id,
                "mode": mode,
                "position_id": pid,
                "capacity_MW": case.capacities_mw[n],
                "nominal_H_h": H[n],
                "energy_MWh": energy[n],
                "physical_FDE_h": fde[n],
                "rule_debit_h": debit[n],
                "remaining_rule_entitlement_h": H[n] - debit[n],
                "material_hours": touched[n],
                "deep_hours_z_gt_0_5": deep_hours[n],
                "max_depth": float(np.max(x[:, n] / case.capacities_mw[n])),
                "material_boundary_coordinates_1e_7_MW": int(near[:, n].sum()),
                "upward_changes_gt_1e_7_MW": int(np.count_nonzero(up[:, n] > 1e-7)),
                "sum_upward_changes_MW": math.fsum(up[:, n]),
                "energy_FDE_identity_enclosure_MWh": identity_bound[n],
            }
        )
    summary = {
        "year": year,
        "rule": policy,
        "object": object_id,
        "mode": mode,
        "S": S,
        "restricted_E_MWh": E,
        "physical_FDE_h": math.fsum(fde),
        "material_hours": math.fsum(touched),
        "deep_hours_z_gt_0_5": math.fsum(deep_hours),
        "peak_branch_overload_MW": peak,
        "hours_overload_gt_1_MW": material_overload_hours,
        "runtime_s": runtime,
        "physical_model_rows_per_hour": case.n_flat_rows,
        "strict_feasibility": audit["strict_pass"] if audit else "ARCHIVED_QUALIFICATION_RETAINED",
    }
    return summary, participants


def ledger_audit(case, exp, folder, raw, q, epoch_hashes):
    balance = exp.allowance.values.copy()
    cumulative = np.zeros_like(balance)
    records = []
    for i, start in enumerate(range(0, case.n_time, 24)):
        stop, end = min(start + 48, case.n_time), min(start + 24, case.n_time)
        sub = case.slice_time(start, stop)
        se = replace(
            exp,
            case_id=f"{exp.case_id}__epoch_{start:06d}",
            physical_case_id=sub.physical_case_id,
            physical_case_hash=sub.physical_case_hash,
            time_grid_id=sub.time_grid.grid_id,
            allowance=AllowanceSpec.explicit(
                case.participant_ids, balance, source="explicit", nominal_rho=0.15
            ),
            mode="annual",
            lookahead_hours=None,
            implementation_hours=None,
            result_root_id=f"{exp.result_root_id}/epochs/epoch_{start:06d}",
            rate_matrix_hash=sha256_array(q[start:stop]),
        )
        sol = load_policy_solution(
            folder / f"epochs/epoch_{start:06d}",
            expected_experiment_hash=se.hash,
            expected_physical_case_hash=sub.physical_case_hash,
        )
        require(
            sol.solution_hash == epoch_hashes[i]
            and sol.stage1.status == sol.stage2.status == "OPTIMAL"
            and sol.replay_summary.get("passed"),
            "Epoch identity/optimality/replay",
        )
        k = end - start
        require(
            np.array_equal(raw[start:end], sol.x_mw[:k]) and np.array_equal(sol.q, q[start:stop]),
            "Committed action/q mismatch",
        )
        proposal = DebitRule("WEIGHTED").debit(sub, sol.x_mw, q=q[start:stop])
        dt = sub.time_grid.delta[:k, None]
        committed = sums(dt * q[start:end] * raw[start:end] / case.capacities_mw)
        original = np.sum(dt * q[start:end] * raw[start:end] / case.capacities_mw, axis=0)
        require(
            np.max(proposal - balance) <= 10 * exp.solver_profile.feasibility_tol,
            "Prospective balance violated",
        )
        before = balance.copy()
        unclipped = balance - original
        require(
            unclipped.min() >= -10 * exp.solver_profile.feasibility_tol,
            "Negative raw carried balance beyond convention",
        )
        balance = np.maximum(unclipped, 0.0)
        cumulative += committed
        for n, pid in enumerate(case.participant_ids):
            records.append(
                {
                    "epoch": start,
                    "position_id": pid,
                    "balance_before_h": before[n],
                    "prospective_debit_h": proposal[n],
                    "committed_debit_h": committed[n],
                    "balance_after_h": balance[n],
                    "clipping_h": max(0.0, -unclipped[n]),
                    "unclipped_independent_balance_h": exp.allowance.values[n] - cumulative[n],
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(folder / "ledger_audit.csv", index=False)
    return {
        "complete_coverage": True,
        "epochs": len(epoch_hashes),
        "minimum_carried_balance_h": float(frame.balance_after_h.min()),
        "sum_clipping_h": float(frame.clipping_h.sum()),
        "final_balance_h": balance.tolist(),
        "minimum_independent_unclipped_balance_h": float(
            frame.unclipped_independent_balance_h.min()
        ),
    }


def new_policy(archive, case, year, kind, card, freeze, out, verify=False):
    folder = out / f"year_{year}/{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "UNIFORM_PREMIUM":
        q = np.full(case.requests_mw.shape, card["uniform_rate"])
    else:
        scores = hinge_relief_scores(case)
        q = np.asarray(card["rates"])[buckets(scores, card["thresholds"])]
        del scores
    require(np.all(q >= 1) and np.all(q <= 1 + DESIGN["gamma"]), "Conservative bounds failed")
    template_path = (
        archive.root / "RESULTS/rolling/m1_m2/year_2029/rho_0.15/rolling/M2/EXPERIMENT_CASE.json"
    )
    archive.verify(template_path)
    template = experiment_from_json(template_path)
    allowance = AllowanceSpec.from_nominal(
        case.participant_ids, case.contract_mask, case.time_grid, 0.15
    )
    exp = replace(
        template,
        case_id=f"DEBIT_V1_{year}_{kind}",
        physical_case_id=case.physical_case_id,
        physical_case_hash=case.physical_case_hash,
        time_grid_id=case.time_grid.grid_id,
        rule="M3",
        allowance=allowance,
        card_id=kind,
        card_hash=card["card_hash"],
        feature_set_id="REQUESTED_STAGE1_RIGHT_DERIVATIVE_V1",
        information_level="PREDECLARED_CONTROLLED_INPUT",
        rate_matrix_hash=sha256_array(q),
        result_root_id=f"debit_rule_family/year_{year}/{kind}",
        implementation_hash=freeze["implementation_hash"],
    )
    if (folder / "EXPERIMENT_CASE.json").is_file():
        require(
            read(folder / "EXPERIMENT_CASE.json")["experiment_hash"] == exp.hash,
            "Existing new-policy identity differs",
        )
    else:
        require(not verify, "Cannot verify missing new policy")
        atomic_write_json(
            folder / "EXPERIMENT_CASE.json",
            {"experiment": asdict(exp), "experiment_hash": exp.hash},
        )
    if (folder / "RAW_ROLLING_POLICY.npz").is_file():
        registry = read(folder / "rolling_registry.json")
        require(
            registry["experiment_hash"] == exp.hash
            and sha256_file(folder / "RAW_ROLLING_POLICY.npz") == registry["arrays_sha256"],
            "Saved new raw identity",
        )
        with np.load(folder / "RAW_ROLLING_POLICY.npz", allow_pickle=False) as z:
            raw, final = z["x_mw"], z["final_balance_hours"]
        runtime, epoch_hashes = registry["wall_time_s"], registry["epoch_hashes"]
    else:
        require(not verify, "Missing raw output")
        started = time.perf_counter()
        result = solve_rolling(case, exp, q=q, use_stage3=False, checkpoint_root=folder / "epochs")
        runtime = time.perf_counter() - started
        require(
            result.complete_coverage
            and result.covered_intervals
            == tuple((s, min(s + 24, case.n_time)) for s in range(0, case.n_time, 24)),
            "Chronology differs",
        )
        raw, final = result.x_mw, result.final_balance_hours
        epoch_hashes = [s.solution_hash for s in result.epoch_solutions]
        np.savez_compressed(
            folder / "RAW_ROLLING_POLICY.npz", x_mw=raw, final_balance_hours=final, q=q
        )
        atomic_write_json(
            folder / "rolling_registry.json",
            {
                "experiment_hash": exp.hash,
                "wall_time_s": runtime,
                "epoch_hashes": epoch_hashes,
                "arrays_sha256": sha256_file(folder / "RAW_ROLLING_POLICY.npz"),
                "coverage": list(result.covered_intervals),
                "status": "ALL_EPOCHS_STAGE1_STAGE2_OPTIMAL",
            },
        )
        del result
    ledger = ledger_audit(case, exp, folder, raw, q, epoch_hashes)
    require(np.array_equal(final, ledger["final_balance_h"]), "Final ledger mismatch")
    raw_audit = weighted_budget_audit(case, raw, q, allowance.values)
    guarded, guard = weighted_witness(case, raw, q, allowance.values)
    if not (folder / "GUARDED_ANNUAL_WITNESS.npz").is_file():
        np.savez_compressed(folder / "GUARDED_ANNUAL_WITNESS.npz", x_mw=guarded, q=q)
    else:
        with np.load(folder / "GUARDED_ANNUAL_WITNESS.npz", allow_pickle=False) as z:
            require(
                np.array_equal(z["x_mw"], guarded) and np.array_equal(z["q"], q),
                "Witness reproducibility",
            )
    summaries, participants = [], []
    for name, x, audit in [
        ("RAW_ROLLING_POLICY", raw, raw_audit),
        ("GUARDED_ANNUAL_WITNESS", guarded, guard),
    ]:
        summary, rows = metrics(
            case,
            x,
            q,
            allowance.values,
            year,
            kind,
            name,
            "rolling" if name.startswith("RAW") else "post_solve_annual_witness",
            runtime,
            audit,
        )
        summaries.append(summary)
        participants.extend(rows)
    result = {
        "policies": summaries,
        "participants": participants,
        "raw_audit": raw_audit,
        "guard": guard,
        "ledger": ledger,
        "experiment_hash": exp.hash,
        "q_hash": sha256_array(q),
        "guard_S_change": summaries[1]["S"] - summaries[0]["S"],
    }
    atomic_write_json(folder / "result.json", result)
    return result


def run_campaign(archive, parent, out, verify=False):
    require(
        (out / "MINIMAL_EXPERIMENT_PLAN.md").is_file()
        and (out / "M3_FLAT_CARD_DIAGNOSTIC.md").is_file(),
        "Completed audit and frozen plan required before optimization",
    )
    card = read(out / "FITTED_CARD.json")
    payload = dict(card)
    digest = payload.pop("card_hash")
    require(
        sha256_json(payload) == digest and card["design"] == DESIGN and card["gate"] == GATE,
        "Card/parameters changed",
    )
    for name, expected in card["input_hashes"].items():
        require(archive.verify(archive.root / name) == expected, "Fitting source changed")
    for name, expected in read(out / "archive_audit.json")["archive_input_hashes"].items():
        require(archive.verify(archive.root / name) == expected, "Frozen comparator changed")
    require(
        card["diagnostic"]["positive_reference_buckets"] >= 2
        and card["diagnostic"]["rate_dispersion"] >= 0.01,
        "No meaningful differentiated signal: exclude new campaign",
    )
    freeze_path = out / "EXPERIMENT_FREEZE.json"
    identity = {
        "card_hash": digest,
        "implementation_hash": sha256_json(code_hashes()),
        "code_hashes": code_hashes(),
        "plan_hash": sha256_file(out / "MINIMAL_EXPERIMENT_PLAN.md"),
        "theory_hash": sha256_file(out / "DEBIT_RULE_FAMILY_THEORY.md"),
        "data_audit_hash": sha256_file(out / "DATA_AND_METHOD_FEASIBILITY.md"),
        "m3_audit_hash": sha256_file(out / "M3_FLAT_CARD_DIAGNOSTIC.md"),
        "gate": GATE,
        "parent_physical_case_hash": parent.physical_case_hash,
        "maximum_new_trajectories": 8,
    }
    if freeze_path.is_file():
        require(read(freeze_path) == identity, "Frozen experiment changed")
    else:
        require(not verify, "Missing freeze")
        atomic_write_json(freeze_path, identity)
    import gurobipy as gp

    require(gp.gurobi.version() == (12, 0, 3), "Frozen solver version required")
    manifest = {
        "status": "RUNNING",
        "freeze": identity,
        "python": platform.python_version(),
        "gurobi": list(gp.gurobi.version()),
        "annual_solver_fingerprint": annual_solver_fingerprint(SOURCE),
        "completed": [],
        "baseline_reruns": 0,
    }
    atomic_write_json(out / "experiment_manifest.json", manifest)
    try:
        promoted = False
        for year in (2028, 2029, 2030, 2031):
            if year != 2028 and not promoted:
                break
            case = year_case(parent, year)
            pair = {}
            for kind in ("UNIFORM_PREMIUM", "CONSERVATIVE_STATE"):
                print(f"NEW POLICY {year} {kind}", flush=True)
                pair[kind] = new_policy(archive, case, year, kind, card, identity, out, verify)
                manifest["completed"].append({"year": year, "rule": kind})
                atomic_write_json(out / "experiment_manifest.json", manifest)
                export_new_results(out)
                gc.collect()
            if year == 2028:
                U, D = pair["UNIFORM_PREMIUM"], pair["CONSERVATIVE_STATE"]
                gain_raw = U["policies"][0]["S"] - D["policies"][0]["S"]
                gain_guard = U["policies"][1]["S"] - D["policies"][1]["S"]
                threshold = max(
                    GATE["minimum_absolute_improvement"],
                    GATE["minimum_relative_improvement"] * U["policies"][1]["S"],
                )
                promoted = min(gain_raw, gain_guard) >= threshold and all(
                    p["guard"]["strict_pass"] for p in pair.values()
                )
                decision = {
                    "validation_year": 2028,
                    "promoted": promoted,
                    "gain_raw": gain_raw,
                    "gain_guarded": gain_guard,
                    "threshold": threshold,
                    "selection": "single fitting-designed candidate; validation accepts or rejects without retuning",
                    "heldout_years": [2029, 2030] if promoted else [],
                    "confirmation_year": 2031 if promoted else None,
                }
                if (out / "VALIDATION_DECISION.json").is_file():
                    require(
                        read(out / "VALIDATION_DECISION.json") == decision,
                        "Validation decision changed",
                    )
                atomic_write_json(out / "VALIDATION_DECISION.json", decision)
                manifest["validation_decision"] = decision
            del pair, case
        manifest["status"] = "COMPLETE" if promoted else "COMPLETE_VALIDATION_REJECTION"
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        export_new_results(out)
        atomic_write_json(out / "run_preservation.json", archive.preservation())
        manifest["outputs"] = {
            p.relative_to(out).as_posix(): sha256_file(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "experiment_manifest.json" and p.suffix != ".log"
        }
        atomic_write_json(out / "experiment_manifest.json", manifest)


def export_new_results(out):
    results = [read(p) for p in sorted(out.glob("year_*/*/result.json"))]
    if not results:
        return
    pd.DataFrame([r for obj in results for r in obj["policies"]]).to_csv(
        out / "policy_results.csv", index=False
    )
    pd.DataFrame([r for obj in results for r in obj["participants"]]).to_csv(
        out / "participant_metrics.csv", index=False
    )
    atomic_write_json(
        out / "feasibility_audit.json",
        {
            "objects": [
                {
                    "experiment_hash": o["experiment_hash"],
                    "raw": o["raw_audit"],
                    "guarded": o["guard"],
                    "ledger": o["ledger"],
                    "guard_S_change": o["guard_S_change"],
                }
                for o in results
            ],
            "qualification": "Raw chronological policies and distinct annual guarded witnesses; floating point network replay is not an exact-real optimality certificate.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    archive = Archive(args.root)
    out = archive.root / "RESULTS/debit_rule_family"
    out.mkdir(exist_ok=True)
    inputs = (
        archive.root
        / "IMPLEMENTATION/frozen_inputs/artifacts/real_case/physical_case/FINAL_DUTCH_CASE"
    )
    for name in ("physical_case.json", "physical_case_arrays.npz"):
        archive.verify(inputs / name)
    require(
        annual_solver_fingerprint(SOURCE)
        == read(archive.root / "ARCHIVE_MANIFEST.json")["annual_solver_fingerprint"],
        "Frozen annual solver changed",
    )
    parent = load_physical_case(inputs)
    require(
        parent.n_flat_rows == 8837 and parent.n_participants == 114 and len(parent.bus_ids) == 45,
        "Wrong physical evaluator",
    )
    if args.prepare:
        prepare(archive, parent, out)
    else:
        run_campaign(archive, parent, out, verify=args.verify)


if __name__ == "__main__":
    main()
