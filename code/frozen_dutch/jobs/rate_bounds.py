from __future__ import annotations


# ============================================================================
# scripts.run_m3_rate_bound_sensitivity (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from jobs.common import EVALUATION_YEARS, atomic_parquet, frozen_matched_decomposition, load_case, load_frozen_card, make_experiment, q_for_year, run_annual, utc_now, year_case
from tpwrs.execution import CampaignLock
from tpwrs.config import scientific_fingerprint
from tpwrs.contracts import CardSpec
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.persistence import save_card
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.cards import make_card, matched_allowance, q_from_card, verify_m2_matched_contains_m3
from jobs.gamma import _ratio_columns, _ratio_metric
from tpwrs.provenance import require_compatible_status
from jobs.sensitivity_equivalence import (
    exact_equivalence_certificate,
    is_exact_flat_card,
    load_exact_core_bundle,
    matched_stage1_value,
    write_equivalence_certificate,
)


BOUNDS = {
    "tight": (0.7, 1.3),
    "primary": (0.5, 1.5),
    "wide": (0.25, 2.0),
}
FIT_YEARS = (2025, 2026, 2027)


def _completed_case(case_root: Path) -> dict[str, Any] | None:
    status_path = case_root / "STATUS.json"
    summary_path = case_root / "summary.csv"
    if not status_path.is_file() or not summary_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        return None
    expected = status.get("artifacts", {}).get("summary", {}).get("sha256")
    if expected != sha256_file(summary_path):
        return None
    return status


def _commit_case(
    case_root: Path,
    *,
    implementation_hash: str,
    year: int,
    label: str,
    summary_row: dict[str, Any],
    method_artifacts: dict[str, Path],
    equivalence: dict[str, Any] | None = None,
) -> None:
    case_root.mkdir(parents=True, exist_ok=True)
    summary_path = case_root / "summary.csv"
    atomic_write_csv(summary_path, pd.DataFrame([summary_row]))
    artifacts = {
        "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)}
    }
    for name, path in method_artifacts.items():
        if not path.is_file():
            raise RuntimeError(f"Certified method artifact is missing: {path}")
        try:
            recorded_path = path.resolve().relative_to(case_root.resolve()).as_posix()
            path_scope = "CASE_ROOT"
        except ValueError:
            recorded_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
            path_scope = "RELEASE_ROOT"
        artifacts[name] = {
            "path": recorded_path,
            "path_scope": path_scope,
            "sha256": sha256_file(path),
        }
    atomic_write_json(
        case_root / "STATUS.json",
        {
            "schema": "TPWRS_RATE_BOUND_CASE_V1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "scientific_fingerprint": implementation_hash,
            "year": int(year),
            "bound_label": label,
            "summary_row": summary_row,
            "artifacts": artifacts,
            **({} if equivalence is None else equivalence),
        },
    )

def _fit_frame(
    case: Any,
    cards_root: Path,
    features_root: Path,
    selected: Any,
    model_path: Path,
) -> pd.DataFrame:
    teachers = []
    features = []
    for year in FIT_YEARS:
        teacher_path = cards_root / f"rho_0.03/teacher_rows_{year}.parquet"
        feature_path = features_root / f"F0_{year}.parquet"
        teacher = pd.read_parquet(teacher_path)
        feature = pd.read_parquet(feature_path)
        feature = feature.rename(columns={"timestamp_utc": "timestamp"})
        teacher["timestamp"] = pd.to_datetime(
            teacher["timestamp"], utc=True
        ).astype(str)
        feature["timestamp"] = pd.to_datetime(
            feature["timestamp"], utc=True
        ).astype(str)
        teacher["participant_id"] = teacher["participant_id"].astype(str)
        feature["participant_id"] = feature["participant_id"].astype(str)
        teachers.append(teacher)
        features.append(feature)

    fit = pd.concat(teachers, ignore_index=True)
    feature_frame = pd.concat(features, ignore_index=True)
    columns = tuple(
        json.loads(
            (features_root / "F0_SCHEMA.json").read_text(encoding="utf-8")
        )["columns"]
    )
    keys = ["timestamp", "participant_id", "direction"]
    add = [column for column in columns if column not in fit.columns]
    fit = fit.merge(
        feature_frame[keys + add],
        on=keys,
        validate="one_to_one",
    )
    bundle = joblib.load(model_path)
    fit["score_selected"] = bundle.predict(fit)
    return fit


def _cards(
    case: Any,
    cards_root: Path,
    features_root: Path,
) -> tuple[dict[str, Any], Path]:
    selected, _, model_path = load_frozen_card(case, cards_root, 0.03)
    exact_flat_parent = is_exact_flat_card(selected)
    if exact_flat_parent:
        fit = None
    else:
        from jobs.common import verify_fitting_tables
        verify_fitting_tables(selected,cards_root,features_root)
        fit = _fit_frame(case, cards_root, features_root, selected, model_path)
    output = cards_root / "sensitivity_r10/rate_bounds"
    output.mkdir(parents=True, exist_ok=True)

    cards: dict[str, Any] = {"primary": selected}
    for label, bounds in BOUNDS.items():
        if label == "primary":
            continue
        if exact_flat_parent:
            card = CardSpec(
                card_id=f"RHO03_F0_{selected.model_family.upper()}_{label.upper()}_BOUNDS_R10",
                allowance_hash=selected.allowance_hash,
                feature_set_id=selected.feature_set_id,
                model_family=selected.model_family,
                thresholds=dict(selected.thresholds),
                rates={key: (1.0, 1.0, 1.0) for key in selected.rates},
                q_bounds=bounds,
                lambda_q=selected.lambda_q,
                direction_fallbacks=dict(selected.direction_fallbacks),
                source_hashes={
                    **dict(selected.source_hashes),
                    "selected_parent_card": selected.hash,
                    "rate_bound_builder": sha256_file(Path(__file__)),
                    "rate_bounds": sha256_json(bounds),
                    "exact_flat_parent_equivalence": sha256_json({
                        "selected_parent_card": selected.hash,
                        "rates": {key: [1.0, 1.0, 1.0] for key in selected.rates},
                    }),
                },
                fit_years=selected.fit_years,
                validation_year=selected.validation_year,
                heldout_access_audit=selected.heldout_access_audit,
            )
        else:
            card = make_card(
                card_id=f"RHO03_F0_{selected.model_family.upper()}_{label.upper()}_BOUNDS",
                allowance_hash=selected.allowance_hash,
                feature_set_id=selected.feature_set_id,
                model_family=selected.model_family,
                fit_frame=fit,
                score_column="score_selected",
                q_bounds=bounds,
                lambda_q=selected.lambda_q,
                source_hashes={
                    **dict(selected.source_hashes),
                    "selected_parent_card": selected.hash,
                    "rate_bound_builder": sha256_file(Path(__file__)),
                    "rate_bounds": sha256_json(bounds),
                },
            )
        # Thresholds must be the same training-score tertiles.  This sensitivity
        # changes only the allowed rate amplitude.
        for direction in selected.thresholds:
            if not np.allclose(
                np.asarray(card.thresholds[direction], dtype=float),
                np.asarray(selected.thresholds[direction], dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"{label}: rate-bound card changed frozen bucket thresholds."
                )
        save_card(card, output / f"{label.upper()}_CARD.json")
        cards[label] = card

    save_card(selected, output / "PRIMARY_CARD.json")
    if fit is not None:
        atomic_parquet(output / "FITTING_REFERENCE_ROWS.parquet", fit)
    atomic_write_json(
        output / "RATE_BOUND_CARD_FREEZE.json",
        {
            "schema": "TPWRS_RATE_BOUND_CARD_FREEZE_V1",
            "status": "PASS",
            "created_utc": utc_now(),
            "rho": 0.03,
            "selected_parent_card_hash": selected.hash,
            "model_bundle_sha256": sha256_file(model_path),
            "fit_years": FIT_YEARS,
            "bounds": BOUNDS,
            "exact_flat_parent_equivalence": exact_flat_parent,
            "cards": {
                label: {
                    "card_id": card.card_id,
                    "card_hash": card.hash,
                    "rates": dict(card.rates),
                    "thresholds": dict(card.thresholds),
                }
                for label, card in cards.items()
            },
            "claim_boundary": (
                "PREDECLARED RATE-AMPLITUDE SENSITIVITY; SCORE MODEL AND "
                "TRAINING-ONLY BUCKET THRESHOLDS ARE FIXED."
            ),
        },
    )
    return cards, model_path


def _rate_bound_row(
    *,
    year: int,
    label: str,
    card: Any,
    s2: float,
    s3: float,
    sm: float,
    f2: float,
    f3: float,
    e3: float,
) -> dict[str, Any]:
    decomposition = frozen_matched_decomposition(s2, s3, sm)
    return {
        "year": int(year),
        "bound_label": label,
        "q_min": BOUNDS[label][0],
        "q_max": BOUNDS[label][1],
        "card_id": card.card_id,
        "card_hash": card.hash,
        "S_M2": s2,
        "S_M3": s3,
        "S_M2_MATCHED": sm,
        "S_M2_minus_S_M3": s2 - s3,
        "S_M2_minus_S_M2_MATCHED": s2 - sm,
        **_ratio_columns(
            "security_relief_pct_vs_M2",
            _ratio_metric(s2 - s3, s2, denominator_name="M2 residual"),
        ),
        **_ratio_columns(
            "matched_efficiency_pct",
            _ratio_metric(
                s2 - s3,
                s2 - sm,
                denominator_name="M2 minus M2_MATCHED residual",
                comparison_tolerance=decomposition["comparison_tolerance"],
            ),
            claimable_name="matched_efficiency_claimable",
        ),
        "physical_FDE_M2_h": f2,
        "physical_FDE_M3_h": f3,
        **_ratio_columns(
            "physical_FDE_change_pct_vs_M2",
            _ratio_metric(f3 - f2, f2, denominator_name="M2 physical FDE"),
        ),
        "E_restricted_mwh_M3": e3,
        **decomposition,
        "action": "REUSE_EXACT_EQUIVALENT_CORE",
    }


def _rate_bound_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, part in frame.loc[frame["year"].isin((2029, 2030))].groupby("bound_label"):
        s2 = float(part["S_M2"].sum())
        s3 = float(part["S_M3"].sum())
        sm = float(part["S_M2_MATCHED"].sum())
        f2 = float(part["physical_FDE_M2_h"].sum())
        f3 = float(part["physical_FDE_M3_h"].sum())
        decomposition = frozen_matched_decomposition(s2, s3, sm)
        rows.append({
            "period": "2029-2030 held-out",
            "bound_label": label,
            "q_min": float(part["q_min"].iloc[0]),
            "q_max": float(part["q_max"].iloc[0]),
            "S_M2": s2,
            "S_M3": s3,
            "S_M2_MATCHED": sm,
            "S_M2_minus_S_M3": s2 - s3,
            "S_M2_minus_S_M2_MATCHED": s2 - sm,
            **_ratio_columns(
                "security_relief_pct_vs_M2",
                _ratio_metric(s2 - s3, s2, denominator_name="held-out M2 residual"),
            ),
            **_ratio_columns(
                "matched_efficiency_pct",
                _ratio_metric(
                    s2 - s3,
                    s2 - sm,
                    denominator_name="held-out M2 minus M2_MATCHED residual",
                    comparison_tolerance=decomposition["comparison_tolerance"],
                ),
                claimable_name="matched_efficiency_claimable",
            ),
            **_ratio_columns(
                "physical_FDE_change_pct_vs_M2",
                _ratio_metric(f3 - f2, f2, denominator_name="held-out M2 physical FDE"),
            ),
            **decomposition,
        })
    return pd.DataFrame(rows)


def _run_exact_flat_rate_bounds(
    *,
    case: Any,
    selected: Any,
    cards: dict[str, Any],
    core_results: Path,
    results: Path,
    implementation_hash: str,
) -> int:
    if not is_exact_flat_card(selected) or not all(is_exact_flat_card(card) for card in cards.values()):
        raise RuntimeError("Exact rate-bound reuse requires bitwise-flat serialized cards.")
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = {
        "schema": "TPWRS_RATE_BOUND_RUN_R10_V1",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "scientific_fingerprint": implementation_hash,
        "physical_case_hash": case.physical_case_hash,
        "cards": {label: card.hash for label, card in cards.items()},
        "years": list(EVALUATION_YEARS),
        "action": "REUSE_EXACT_EQUIVALENT_CORE",
        "heavy_solve_count": 0,
        "completed_cases": [],
    }
    status_path = results / "RUN_STATUS.json"
    atomic_write_json(status_path, manifest)
    for year in EVALUATION_YEARS:
        bundle = load_exact_core_bundle(core_results, year)
        solutions = bundle["solutions"]
        metrics = bundle["metrics"]
        s2 = float(solutions["M2"].s_value_mw_h_row)
        s3 = float(solutions["M3"].s_value_mw_h_row)
        sm = matched_stage1_value(solutions["M2_MATCHED"])
        f2 = float(metrics["M2"]["physical_fde_sum_h"])
        f3 = float(metrics["M3"]["physical_fde_sum_h"])
        e3 = float(metrics["M3"]["E_restricted_mwh"])
        for label, card in cards.items():
            case_root = results / f"year_{year}/{label}"
            case_root.mkdir(parents=True, exist_ok=True)
            certificate = exact_equivalence_certificate(
                year=year,
                variant_kind="rate_bound",
                variant_value={"label": label, "bounds": list(BOUNDS[label])},
                final_card=card,
                selected_parent_card=selected,
                core_bundle=bundle,
                physical_case_hash=year_case(case, year).physical_case_hash,
                scientific_fingerprint=implementation_hash,
            )
            certificate_path = case_root / "EXACT_EQUIVALENCE_CERTIFICATE.json"
            certificate_file_hash = write_equivalence_certificate(certificate_path, certificate)
            row = _rate_bound_row(
                year=year, label=label, card=card, s2=s2, s3=s3, sm=sm, f2=f2, f3=f3, e3=e3
            )
            _commit_case(
                case_root,
                implementation_hash=implementation_hash,
                year=year,
                label=label,
                summary_row=row,
                method_artifacts={
                    "M3_metrics": bundle["roots"]["M3"] / "metrics.json",
                    "M2_MATCHED_metrics": bundle["roots"]["M2_MATCHED"] / "metrics.json",
                    "exact_equivalence": certificate_path,
                },
                equivalence={
                    "action": "REUSE_EXACT_EQUIVALENT_CORE",
                    "source_m3_solution_hash": solutions["M3"].solution_hash,
                    "source_matched_solution_hash": solutions["M2_MATCHED"].solution_hash,
                    "exact_equivalence_certificate_hash": certificate["certificate_hash"],
                    "exact_equivalence_certificate_file_sha256": certificate_file_hash,
                    "final_card_hash": card.hash,
                },
            )
            rows.append(row)
            manifest["completed_cases"].append(f"{year}/{label}")
            manifest["heartbeat_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
    frame = pd.DataFrame(rows).sort_values(["year", "q_min", "q_max"])
    held = _rate_bound_heldout(frame)
    atomic_write_csv(results / "RATE_BOUND_YEARLY.csv", frame)
    atomic_write_csv(results / "RATE_BOUND_HELDOUT.csv", held)
    atomic_write_json(
        results / "RATE_BOUND_STATUS.json",
        {
            "schema": "TPWRS_RATE_BOUND_STATUS_R10_V1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "scientific_fingerprint": implementation_hash,
            "rows": len(frame),
            "expected_rows": len(BOUNDS) * len(EVALUATION_YEARS),
            "bounds": BOUNDS,
            "all_cases_action": "REUSE_EXACT_EQUIVALENT_CORE",
            "heavy_solve_count": 0,
            "exact_flat_parent_equivalence": True,
            "outputs": {
                "yearly": {"path": "RATE_BOUND_YEARLY.csv", "sha256": sha256_file(results / "RATE_BOUND_YEARLY.csv")},
                "heldout": {"path": "RATE_BOUND_HELDOUT.csv", "sha256": sha256_file(results / "RATE_BOUND_HELDOUT.csv")},
            },
        },
    )
    manifest.update(status="COMPLETE", completed_utc=utc_now())
    atomic_write_json(status_path, manifest)
    print("M3_RATE_BOUND_SENSITIVITY=COMPLETE EXACT_FLAT_REUSE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exact corrected V3 M3 rate-bound sensitivity at rho=0.03."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=ROOT / "artifacts/real_case/physical_case/FINAL_DUTCH_CASE",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=ROOT / "artifacts/cards",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/features",
    )
    parser.add_argument(
        "--core-results",
        type=Path,
        default=ROOT / "results_m3_final",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results_m3_sensitivity_r10/rate_bounds",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case = load_case(args.case)
    core_status, _ = require_compatible_status(
        args.core_results / "M3_CORE_STATUS.json",
        allowed=("COMPLETE",),
        use="UPSTREAM_STAGE05_CORE_NUMERICAL_EVIDENCE",
    )
    implementation_hash = scientific_fingerprint()
    if core_status.get("status") != "COMPLETE":
        raise RuntimeError("Corrected M3 core campaign must complete first.")

    cards, _ = _cards(case, args.cards, args.features)
    exact_flat_parent = all(is_exact_flat_card(card) for card in cards.values())
    args.results.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema": "TPWRS_RATE_BOUND_PLAN_V1",
        "status": "PASS",
        "created_utc": utc_now(),
        "rho": 0.03,
        "bounds": BOUNDS,
        "evaluation_years": EVALUATION_YEARS,
        "primary_bound_reused_from_core": True,
        "new_heavy_case_count": 0 if exact_flat_parent else 12,
        "exact_flat_parent_equivalence": exact_flat_parent,
        "no_scientific_optimization": bool(args.dry_run),
    }
    atomic_write_json(args.results / "DRY_RUN_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if exact_flat_parent:
        return _run_exact_flat_rate_bounds(
            case=case,
            selected=cards["primary"],
            cards=cards,
            core_results=args.core_results,
            results=args.results,
            implementation_hash=implementation_hash,
        )

    identity = {
        "physical_case_hash": case.physical_case_hash,
        "cards": {label: card.hash for label, card in cards.items()},
        "core_status_hash": sha256_file(args.core_results / "M3_CORE_STATUS.json"),
        "implementation_hash": implementation_hash,
        "years": EVALUATION_YEARS,
    }
    campaign_hash = sha256_json(identity)
    status_path = args.results / "RUN_STATUS.json"

    with CampaignLock(args.results / ".campaign.lock", campaign_hash):
        if status_path.is_file():
            manifest = json.loads(status_path.read_text(encoding="utf-8"))
            if manifest.get("campaign_hash") != campaign_hash:
                raise RuntimeError("Rate-bound result root belongs to another campaign.")
            manifest["status"] = "RUNNING"
            manifest["resumed_utc"] = utc_now()
            manifest["invocation_count"] = int(
                manifest.get("invocation_count", 1)
            ) + 1
        else:
            manifest = {
                "schema": "TPWRS_RATE_BOUND_RUN_V1",
                "status": "RUNNING",
                "started_utc": utc_now(),
                "invocation_count": 1,
                "campaign_hash": campaign_hash,
                **identity,
                "jobs": {},
            }
        atomic_write_json(status_path, manifest)

        rows: list[dict[str, Any]] = []
        try:
            for year in EVALUATION_YEARS:
                current = year_case(case, year)
                feature_q, score = q_for_year(
                    current,
                    year,
                    cards["primary"],
                    args.cards,
                    args.features,
                )
                del feature_q
                baseline = json.loads(
                    (
                        args.core_results
                        / f"year_{year}/rho_0.03/annual/M2/metrics.json"
                    ).read_text(encoding="utf-8")
                )
                s2 = float(baseline["S_mw_h_row"])
                f2 = float(baseline["physical_fde_sum_h"])

                for label, card in cards.items():
                    case_root = args.results / f"year_{year}/{label}"
                    completed = _completed_case(case_root)
                    if completed is not None:
                        rows.append(dict(completed["summary_row"]))
                        continue
                    if label == "primary":
                        m3_metrics_path = (
                            args.core_results
                            / f"year_{year}/rho_0.03/annual/M3/metrics.json"
                        )
                        matched_metrics_path = (
                            args.core_results
                            / f"year_{year}/rho_0.03/annual/M2_MATCHED/metrics.json"
                        )
                        m3_metrics = json.loads(m3_metrics_path.read_text(encoding="utf-8"))
                        matched_metrics = json.loads(
                            matched_metrics_path.read_text(encoding="utf-8")
                        )
                        s3 = float(m3_metrics["S_mw_h_row"])
                        sm = float(matched_metrics["S_mw_h_row"])
                        f3 = float(m3_metrics["physical_fde_sum_h"])
                        E3 = float(m3_metrics["E_restricted_mwh"])
                        action = "REUSE_CORE"
                    else:
                        q = q_from_card(
                            card,
                            current.participant_directions,
                            score,
                        )
                        relative = Path(
                            f"year_{year}/{label}/M3"
                        )
                        experiment = make_experiment(
                            current,
                            case_id=f"Y{year}_RHO03_M3_{label.upper()}_BOUNDS",
                            rule="M3",
                            rho=0.03,
                            mode="annual",
                            result_root_id=str(relative).replace("\\", "/"),
                            implementation_hash=implementation_hash,
                            card=card,
                            q=q,
                            information_level=(
                                f"FROZEN_F0_RATE_BOUND_{label.upper()}"
                            ),
                        )
                        m3_solution, _ = run_annual(
                            current,
                            experiment,
                            q,
                            args.results / relative,
                        )
                        matched = matched_allowance(current, m3_solution)
                        if not verify_m2_matched_contains_m3(
                            current,
                            m3_solution,
                            matched,
                        ):
                            raise RuntimeError(
                                f"{year}/{label}: matched allowance does not contain M3."
                            )
                        matched_relative = Path(
                            f"year_{year}/{label}/M2_MATCHED"
                        )
                        matched_experiment = make_experiment(
                            current,
                            case_id=(
                                f"Y{year}_RHO03_M2_MATCHED_{label.upper()}_BOUNDS"
                            ),
                            rule="M2_MATCHED",
                            rho=0.03,
                            mode="annual",
                            result_root_id=str(matched_relative).replace("\\", "/"),
                            implementation_hash=implementation_hash,
                            card=card,
                            allowance=matched,
                            information_level=(
                                f"POST_M3_MATCHED_RATE_BOUND_{label.upper()}"
                            ),
                        )
                        matched_solution, _ = run_annual(
                            current,
                            matched_experiment,
                            None,
                            args.results / matched_relative,
                        )
                        s3 = float(m3_solution.s_value_mw_h_row)
                        sm = float(matched_solution.stage1.objective_value)
                        f3 = float(
                            np.sum(
                                current.time_grid.delta[:, None]
                                * m3_solution.x_mw
                                / current.capacities_mw[None, :]
                            )
                        )
                        E3 = float(m3_solution.e_value_mwh)
                        action = "RUN_OR_REUSE"
                        m3_metrics_path = args.results / relative / "metrics.json"
                        matched_metrics_path = (
                            args.results / matched_relative / "metrics.json"
                        )

                    decomposition = frozen_matched_decomposition(s2, s3, sm)
                    row = {
                            "year": year,
                            "bound_label": label,
                            "q_min": BOUNDS[label][0],
                            "q_max": BOUNDS[label][1],
                            "card_id": card.card_id,
                            "card_hash": card.hash,
                            "S_M2": s2,
                            "S_M3": s3,
                            "S_M2_MATCHED": sm,
                            "S_M2_minus_S_M3": s2 - s3,
                            "S_M2_minus_S_M2_MATCHED": s2 - sm,
                            **_ratio_columns(
                                "security_relief_pct_vs_M2",
                                _ratio_metric(
                                    s2 - s3,
                                    s2,
                                    denominator_name="M2 residual",
                                ),
                            ),
                            **_ratio_columns(
                                "matched_efficiency_pct",
                                _ratio_metric(
                                    s2 - s3,
                                    s2 - sm,
                                    denominator_name="M2 minus M2_MATCHED residual",
                                    comparison_tolerance=decomposition[
                                        "comparison_tolerance"
                                    ],
                                ),
                                claimable_name="matched_efficiency_claimable",
                            ),
                            "physical_FDE_M2_h": f2,
                            "physical_FDE_M3_h": f3,
                            **_ratio_columns(
                                "physical_FDE_change_pct_vs_M2",
                                _ratio_metric(
                                    f3 - f2,
                                    f2,
                                    denominator_name="M2 physical FDE",
                                ),
                            ),
                            "E_restricted_mwh_M3": E3,
                            **decomposition,
                            "action": action,
                        }
                    _commit_case(
                        case_root,
                        implementation_hash=implementation_hash,
                        year=year,
                        label=label,
                        summary_row=row,
                        method_artifacts={
                            "M3_metrics": m3_metrics_path,
                            "M2_MATCHED_metrics": matched_metrics_path,
                        },
                    )
                    rows.append(row)

            frame = pd.DataFrame(rows).sort_values(
                ["year", "q_min", "q_max"]
            )
            held_rows = []
            for label, part in frame.loc[
                frame["year"].isin((2029, 2030))
            ].groupby("bound_label"):
                s2 = float(part["S_M2"].sum())
                s3 = float(part["S_M3"].sum())
                sm = float(part["S_M2_MATCHED"].sum())
                f2 = float(part["physical_FDE_M2_h"].sum())
                f3 = float(part["physical_FDE_M3_h"].sum())
                decomposition = frozen_matched_decomposition(s2, s3, sm)
                held_rows.append(
                    {
                        "period": "2029-2030 held-out",
                        "bound_label": label,
                        "q_min": float(part["q_min"].iloc[0]),
                        "q_max": float(part["q_max"].iloc[0]),
                        "S_M2": s2,
                        "S_M3": s3,
                        "S_M2_MATCHED": sm,
                        "S_M2_minus_S_M3": s2 - s3,
                        "S_M2_minus_S_M2_MATCHED": s2 - sm,
                        **_ratio_columns(
                            "security_relief_pct_vs_M2",
                            _ratio_metric(
                                s2 - s3,
                                s2,
                                denominator_name="held-out M2 residual",
                            ),
                        ),
                        **_ratio_columns(
                            "matched_efficiency_pct",
                            _ratio_metric(
                                s2 - s3,
                                s2 - sm,
                                denominator_name=(
                                    "held-out M2 minus M2_MATCHED residual"
                                ),
                                comparison_tolerance=decomposition[
                                    "comparison_tolerance"
                                ],
                            ),
                            claimable_name="matched_efficiency_claimable",
                        ),
                        **_ratio_columns(
                            "physical_FDE_change_pct_vs_M2",
                            _ratio_metric(
                                f3 - f2,
                                f2,
                                denominator_name="held-out M2 physical FDE",
                            ),
                        ),
                        **decomposition,
                    }
                )
            held = pd.DataFrame(held_rows)
            atomic_write_csv(args.results / "RATE_BOUND_YEARLY.csv", frame)
            atomic_write_csv(args.results / "RATE_BOUND_HELDOUT.csv", held)
            atomic_write_json(
                args.results / "RATE_BOUND_STATUS.json",
                {
                    "schema": "TPWRS_RATE_BOUND_STATUS_V1",
                    "status": "COMPLETE",
                    "created_utc": utc_now(),
                    "scientific_fingerprint": implementation_hash,
                    "rows": len(frame),
                    "bounds": BOUNDS,
                    "claim_boundary": (
                        "SECURITY RELIEF IS REPORTED JOINTLY WITH PHYSICAL "
                        "FDE AND M2-MATCHED SHAPE PENALTY."
                    ),
                },
            )
            manifest["status"] = "COMPLETE"
            manifest["completed_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
            print("M3_RATE_BOUND_SENSITIVITY=COMPLETE", flush=True)
            return 0
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["failed_utc"] = utc_now()
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(status_path, manifest)
            raise
