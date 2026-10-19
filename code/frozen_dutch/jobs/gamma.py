from __future__ import annotations


# ============================================================================
# scripts.run_m3_gamma_path (consolidated; see SOURCE_MAP.json)
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
from tpwrs.reporting import circular_block_sums
from tpwrs.execution import CampaignLock
from tpwrs.config import scientific_fingerprint
from tpwrs.contracts import CardSpec
from tpwrs.cards import project_gamma_rates
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.persistence import load_card, save_card
from tpwrs.persistence import load_policy_solution
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.cards import assign_buckets, matched_allowance, q_from_card, verify_m2_matched_contains_m3
from tpwrs.provenance import require_compatible_status
from jobs.sensitivity_equivalence import (
    exact_equivalence_certificate,
    is_exact_flat_card,
    load_exact_core_bundle,
    matched_stage1_value,
    write_equivalence_certificate,
)


_METHODOLOGY = json.loads((ROOT / "specification/METHODOLOGY_FREEZE_V1.json").read_text(encoding="utf-8"))
DEFAULT_GAMMA_GRID = tuple(map(float, _METHODOLOGY["m3"]["gamma_grid"]))
SCREEN_GAMMAS = tuple(map(float, _METHODOLOGY["m3"]["gamma_screen"]))
GLOBAL_BOUNDS = (0.25, 2.0)
FIT_YEARS = (2025, 2026, 2027)
BOOTSTRAP_DRAWS = 1000
BLOCK_HOURS = 168
BOOTSTRAP_SEED = 3118
RATIO_DENOMINATOR_TOL = 1e-12


def _require_finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Required scientific value is non-finite: {name}={value!r}.")
    return result


def _require_finite_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        bad = np.argwhere(~np.isfinite(result))
        first = tuple(map(int, bad[0])) if len(bad) else ()
        raise ValueError(
            f"Required scientific array is non-finite: {name}{first}="
            f"{result[first]!r}."
        )
    return result


def _ratio_metric(
    numerator: Any,
    denominator: Any,
    *,
    denominator_name: str,
    scale: float = 100.0,
    comparison_tolerance: float = RATIO_DENOMINATOR_TOL,
) -> dict[str, Any]:
    """Represent a derived ratio without confusing undefined with zero.

    Inputs are required scientific quantities and therefore fail closed when
    non-finite.  A denominator that is not positive above the declared numeric
    threshold makes the ratio mathematically undefined; strict JSON records a
    null value plus its semantic state instead of a NaN token.
    """
    top = _require_finite_scalar(numerator, "ratio numerator")
    bottom = _require_finite_scalar(denominator, denominator_name)
    tolerance = _require_finite_scalar(
        comparison_tolerance,
        "ratio comparison tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("Ratio comparison tolerance must be nonnegative.")
    common = {
        "numerator": top,
        "denominator": bottom,
        "comparison_tolerance": tolerance,
    }
    if abs(bottom) <= tolerance:
        return {
            **common,
            "value": None,
            "status": "NOT_IDENTIFIED_ZERO_DENOMINATOR",
            "claimable": False,
            "reason": (
                f"abs({denominator_name})={abs(bottom):.17g} does not exceed "
                f"the declared comparison tolerance {tolerance:.17g}."
            ),
        }
    if bottom < -tolerance:
        return {
            **common,
            "value": None,
            "status": "NOT_IDENTIFIED_NONPOSITIVE_DENOMINATOR",
            "claimable": False,
            "reason": (
                f"{denominator_name}={bottom:.17g} is negative beyond the "
                f"declared comparison tolerance {tolerance:.17g}; an "
                "efficiency percentage is not identified."
            ),
        }
    result = _require_finite_scalar(scale * top / bottom, "derived ratio")
    return {
        **common,
        "value": result,
        "status": "FINITE_RESULT",
        "claimable": True,
        "reason": None,
    }


def _ratio_columns(
    name: str,
    metric: dict[str, Any],
    *,
    claimable_name: str | None = None,
) -> dict[str, Any]:
    result = {
        name: metric["value"],
        f"{name}_status": metric["status"],
        f"{name}_reason": metric["reason"],
        f"{name}_numerator": metric.get("numerator"),
        f"{name}_denominator": metric.get("denominator"),
        f"{name}_comparison_tolerance": metric.get("comparison_tolerance"),
    }
    if claimable_name is not None:
        result[claimable_name] = bool(metric.get("claimable", False))
    return result


def _ratio_frame_columns(
    name: str,
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    denominator_name: str,
    comparison_tolerance: float = RATIO_DENOMINATOR_TOL,
) -> dict[str, Any]:
    top = _require_finite_array(numerator, f"{name} numerator")
    bottom = _require_finite_array(denominator, denominator_name)
    if top.shape != bottom.shape:
        raise ValueError(f"{name} numerator and denominator shapes differ.")
    tolerance = _require_finite_scalar(
        comparison_tolerance,
        "bootstrap ratio comparison tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("Bootstrap ratio comparison tolerance must be nonnegative.")
    zero = np.abs(bottom) <= tolerance
    negative = bottom < -tolerance
    defined = bottom > tolerance
    values = np.empty(top.shape, dtype=object)
    values[:] = None
    values[defined] = 100.0 * top[defined] / bottom[defined]
    finite_values = np.asarray(values[defined], dtype=float)
    if not np.isfinite(finite_values).all():
        raise ValueError(f"{name} produced a non-finite defined result.")
    status = np.full(top.shape, "FINITE_RESULT", dtype=object)
    status[zero] = "NOT_IDENTIFIED_ZERO_DENOMINATOR"
    status[negative] = "NOT_IDENTIFIED_NONPOSITIVE_DENOMINATOR"
    reason = np.empty(top.shape, dtype=object)
    reason[:] = None
    reason[zero] = [
        (
            f"abs({denominator_name})={abs(value):.17g} does not exceed "
            f"the declared comparison tolerance {tolerance:.17g}."
        )
        for value in bottom[zero]
    ]
    reason[negative] = [
        (
            f"{denominator_name}={value:.17g} is negative beyond the "
            f"declared comparison tolerance {tolerance:.17g}."
        )
        for value in bottom[negative]
    ]
    return {
        name: values,
        f"{name}_status": status,
        f"{name}_reason": reason,
        f"{name}_numerator": top,
        f"{name}_denominator": bottom,
        f"{name}_comparison_tolerance": np.full(top.shape, tolerance),
    }


def _distribution_summary(values: np.ndarray, name: str) -> dict[str, float]:
    array = _require_finite_array(values, name).reshape(-1)
    if not len(array):
        raise ValueError(f"Cannot summarize an empty required distribution: {name}.")
    return {
        "minimum": float(np.min(array)),
        "percentile_2_5": float(np.percentile(array, 2.5)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "percentile_97_5": float(np.percentile(array, 97.5)),
        "maximum": float(np.max(array)),
    }


def _bootstrap_ratio_record(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    comparison_tolerance: float,
) -> dict[str, Any]:
    top = _require_finite_array(numerator, "bootstrap ratio numerator").reshape(-1)
    bottom = _require_finite_array(
        denominator,
        "bootstrap ratio denominator",
    ).reshape(-1)
    if top.shape != bottom.shape or not len(top):
        raise ValueError("Bootstrap ratio inputs must be nonempty and aligned.")
    tolerance = _require_finite_scalar(
        comparison_tolerance,
        "bootstrap ratio comparison tolerance",
    )
    zero = np.abs(bottom) <= tolerance
    negative = bottom < -tolerance
    defined = bottom > tolerance
    undefined = ~defined
    undefined_count = int(np.sum(undefined))
    total = int(len(bottom))
    return {
        "schema": "TPWRS_MATCHED_EFFICIENCY_BOOTSTRAP_V1",
        "status": (
            "UNSTABLE_NOT_IDENTIFIED"
            if undefined_count
            else "FINITE_RESULT"
        ),
        "matched_efficiency_claimable": undefined_count == 0,
        "total_draws": total,
        "finite_ratio_draws": int(np.sum(defined)),
        "undefined_ratio_draws": undefined_count,
        "undefined_zero_denominator_draws": int(np.sum(zero)),
        "undefined_nonpositive_denominator_draws": int(np.sum(negative)),
        "fraction_undefined": float(undefined_count / total),
        "comparison_tolerance": tolerance,
        "numerator_distribution": _distribution_summary(top, "bootstrap numerator"),
        "denominator_distribution": _distribution_summary(
            bottom,
            "bootstrap denominator",
        ),
        "finite_only_ci_used": False,
        "matched_efficiency_percentile_ci": None,
        "ci_status": (
            "NOT_PRODUCED_UNSTABLE_NOT_IDENTIFIED"
            if undefined_count
            else "NOT_REQUESTED_BY_FROZEN_PROTOCOL"
        ),
    }


def _case_states(
    gamma_values: tuple[float, ...],
    years: tuple[int, ...],
    completed: tuple[str, ...] | list[str] = (),
) -> dict[str, dict[str, str]]:
    certified = set(map(str, completed))
    return {
        f"{year}/gamma_{_gamma_tag(gamma)}": {
            "status": (
                "CERTIFIED"
                if f"{year}/gamma_{_gamma_tag(gamma)}" in certified
                else "PENDING"
            )
        }
        for year in years
        for gamma in gamma_values
    }


def _commit_case_status(
    case_root: Path,
    *,
    implementation_hash: str,
    reporting_source_fingerprint: str | None = None,
    year: int,
    gamma: float,
    summary_row: dict[str, Any],
    artifacts: dict[str, Any],
    equivalence: dict[str, Any] | None = None,
) -> Path:
    """Atomically certify one fully persisted gamma case before aggregation."""
    return atomic_write_json(
        case_root / "STATUS.json",
        {
            "schema": "TPWRS_M3_GAMMA_CASE_V2",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "scientific_fingerprint": implementation_hash,
            "reporting_source_fingerprint": (
                implementation_hash
                if reporting_source_fingerprint is None
                else reporting_source_fingerprint
            ),
            "year": int(year),
            "gamma": _require_finite_scalar(gamma, "gamma"),
            "summary_row": summary_row,
            "artifacts": artifacts,
            **({} if equivalence is None else equivalence),
        },
    )


def _gamma_tag(gamma: float) -> str:
    return f"{float(gamma):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _training_reference(
    cards_root: Path,
    features_root: Path,
    selected: Any,
    model_path: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    teacher_frames = []
    feature_frames = []
    schema = json.loads(
        (features_root / "F0_SCHEMA.json").read_text(encoding="utf-8")
    )
    columns = tuple(schema["columns"])

    for year in FIT_YEARS:
        teacher = pd.read_parquet(
            cards_root / f"rho_0.03/teacher_rows_{year}.parquet"
        )
        feature = pd.read_parquet(features_root / f"F0_{year}.parquet")
        teacher["timestamp"] = pd.to_datetime(
            teacher["timestamp"], utc=True
        ).astype(str)
        feature["timestamp"] = pd.to_datetime(
            feature["timestamp_utc"], utc=True
        ).astype(str)
        teacher["participant_id"] = teacher["participant_id"].astype(str)
        feature["participant_id"] = feature["participant_id"].astype(str)
        teacher_frames.append(teacher)
        feature_frames.append(feature)

    fit = pd.concat(teacher_frames, ignore_index=True)
    feature = pd.concat(feature_frames, ignore_index=True)
    keys = ["timestamp", "participant_id", "direction"]
    add = [column for column in columns if column not in fit.columns]
    fit = fit.merge(feature[keys + add], on=keys, validate="one_to_one")

    bundle = joblib.load(model_path)
    fit["selected_score"] = bundle.predict(fit)

    reference_weights: dict[str, np.ndarray] = {}
    for direction, part in fit.groupby("direction", sort=True):
        direction = str(direction)
        thresholds = selected.thresholds[direction]
        buckets = assign_buckets(
            part["selected_score"].to_numpy(float),
            thresholds,
        )
        exposure = (
            part["delta_t_hours"].to_numpy(float)
            * part["x_m2_mw"].to_numpy(float)
            / part["capacity_mw"].to_numpy(float)
        )
        reference_weights[direction] = np.asarray(
            [
                float(exposure[buckets == bucket].sum())
                for bucket in (1, 2, 3)
            ],
            dtype=float,
        )
    return reference_weights, fit


def _gamma_card(
    selected: Any,
    reference_weights: dict[str, np.ndarray] | None,
    gamma: float,
    output: Path,
) -> CardSpec:
    target = output / f"GAMMA_{_gamma_tag(gamma)}_CARD.json"
    rates: dict[str, tuple[float, float, float]] = {}
    modes: dict[str, str] = {}
    for direction in sorted(selected.rates):
        if is_exact_flat_card(selected):
            values = np.asarray((1.0, 1.0, 1.0), dtype=float)
            mode = "EXACT_FLAT_PARENT_EQUIVALENCE"
        else:
            if reference_weights is None:
                raise RuntimeError("Nonflat gamma projection requires fitting weights.")
            values, mode = project_gamma_rates(
                np.asarray(selected.rates[direction], dtype=float),
                reference_weights[direction],
                float(gamma),
                GLOBAL_BOUNDS,
            )
        rates[direction] = tuple(map(float, values))
        modes[direction] = mode

    gamma_definition_hash = sha256_json(
        {
            "gamma": float(gamma),
            "definition": "1+gamma*(q_selected-1)",
            "projection": "ordered reference-mean-one",
            "bounds": GLOBAL_BOUNDS,
        }
    )
    if target.is_file():
        cached = load_card(target)
        identity_ok = (
            cached.card_id == f"RHO03_GAMMA_{_gamma_tag(gamma)}"
            and cached.allowance_hash == selected.allowance_hash
            and cached.feature_set_id == selected.feature_set_id
            and cached.model_family == selected.model_family
            and cached.thresholds == selected.thresholds
            and cached.q_bounds == GLOBAL_BOUNDS
            and cached.lambda_q == selected.lambda_q
            and cached.direction_fallbacks == selected.direction_fallbacks
            and cached.fit_years == selected.fit_years
            and cached.validation_year == selected.validation_year
            and cached.heldout_access_audit == selected.heldout_access_audit
            and cached.source_hashes.get("selected_parent_card") == selected.hash
            and cached.source_hashes.get("gamma_definition")
            == gamma_definition_hash
            and set(cached.rates) == set(rates)
            and all(
                np.array_equal(
                    np.asarray(cached.rates[key], dtype=float),
                    np.asarray(rates[key], dtype=float),
                )
                for key in rates
            )
        )
        if not identity_ok:
            raise RuntimeError(
                f"Existing gamma card is not a provenance-valid cache: {target}."
            )
        return cached

    card = CardSpec(
        card_id=f"RHO03_GAMMA_{_gamma_tag(gamma)}",
        allowance_hash=selected.allowance_hash,
        feature_set_id=selected.feature_set_id,
        model_family=selected.model_family,
        thresholds=dict(selected.thresholds),
        rates=rates,
        q_bounds=GLOBAL_BOUNDS,
        lambda_q=selected.lambda_q,
        direction_fallbacks=dict(selected.direction_fallbacks),
        source_hashes={
            **dict(selected.source_hashes),
            "selected_parent_card": selected.hash,
            "gamma_builder": sha256_file(Path(__file__)),
            "gamma_definition": gamma_definition_hash,
        },
        fit_years=selected.fit_years,
        validation_year=selected.validation_year,
        heldout_access_audit=selected.heldout_access_audit,
    )
    save_card(card, target)
    atomic_write_json(
        output / f"GAMMA_{_gamma_tag(gamma)}_RATES.json",
        {
            "schema": "TPWRS_GAMMA_RATE_VECTOR_V1",
            "gamma": float(gamma),
            "card_id": card.card_id,
            "card_hash": card.hash,
            "rates": rates,
            "modes": modes,
            "reference_weights": {
                key: value.tolist() for key, value in reference_weights.items()
            } if reference_weights is not None else None,
        },
    )
    return card


def _hourly_residual(case: Any, x: np.ndarray) -> np.ndarray:
    state = case.post_state_mw(np.asarray(x, dtype=float))
    H = case.flat_H
    h = case.flat_h
    rating = case.flat_ratings
    hourly = np.zeros(case.n_time, dtype=float)
    for start in range(0, case.n_time, 128):
        stop = min(case.n_time, start + 128)
        flow = state[start:stop] @ H.T + h[None, :]
        hourly[start:stop] = np.maximum(
            np.abs(flow) - rating[None, :],
            0.0,
        ).sum(axis=1) * case.time_grid.delta[start:stop]
    return hourly


def _completed_case(path: Path) -> dict[str, Any] | None:
    status_path = path / "STATUS.json"
    if not status_path.is_file():
        return None
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        return None
    for metadata in payload.get("artifacts", {}).values():
        artifact = path / metadata["path"]
        if not artifact.is_file() or sha256_file(artifact) != metadata["sha256"]:
            return None
    return payload


def gamma_execution_schedule(gamma_values=DEFAULT_GAMMA_GRID):
    if tuple(gamma_values)!=DEFAULT_GAMMA_GRID:
        raise RuntimeError("Gamma grid must equal the frozen methodology grid.")
    remaining=tuple(g for g in gamma_values if g not in SCREEN_GAMMAS)
    return tuple((year,points) for points in (SCREEN_GAMMAS,remaining) for year in EVALUATION_YEARS)


def _exact_flat_summary_row(
    *,
    year: int,
    gamma: float,
    card: Any,
    s2: float,
    s3: float,
    sm: float,
    f2: float,
    f3: float,
) -> dict[str, Any]:
    decomposition = frozen_matched_decomposition(s2, s3, sm)
    return {
        "year": int(year),
        "gamma": float(gamma),
        "card_id": card.card_id,
        "card_hash": card.hash,
        "q_min_realized": 1.0,
        "q_max_realized": 1.0,
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
        **decomposition,
        "action": "REUSE_EXACT_EQUIVALENT_CORE",
    }


def _gamma_period_frame(yearly: pd.DataFrame) -> pd.DataFrame:
    period_rows = []
    for gamma, part in yearly.groupby("gamma", sort=True):
        for period, years in (
            ("2029-2030 held-out", (2029, 2030)),
            ("2031 confirmatory", (2031,)),
        ):
            subset = part.loc[part["year"].isin(years)]
            if len(subset) != len(years):
                continue
            s2 = float(subset["S_M2"].sum())
            s3 = float(subset["S_M3"].sum())
            sm = float(subset["S_M2_MATCHED"].sum())
            f2 = float(subset["physical_FDE_M2_h"].sum())
            f3 = float(subset["physical_FDE_M3_h"].sum())
            decomposition = frozen_matched_decomposition(s2, s3, sm)
            period_rows.append({
                "period": period,
                "gamma": gamma,
                "S_M2": s2,
                "S_M3": s3,
                "S_M2_MATCHED": sm,
                "S_M2_minus_S_M3": s2 - s3,
                "S_M2_minus_S_M2_MATCHED": s2 - sm,
                **_ratio_columns(
                    "security_relief_pct_vs_M2",
                    _ratio_metric(s2 - s3, s2, denominator_name="period M2 residual"),
                ),
                **_ratio_columns(
                    "matched_efficiency_pct",
                    _ratio_metric(
                        s2 - s3,
                        s2 - sm,
                        denominator_name="period M2 minus M2_MATCHED residual",
                        comparison_tolerance=decomposition["comparison_tolerance"],
                    ),
                    claimable_name="matched_efficiency_claimable",
                ),
                "state_shape_penalty": decomposition["shape_penalty"],
                "raw_shape_residual": decomposition["raw_shape_residual"],
                **_ratio_columns(
                    "physical_FDE_change_pct_vs_M2",
                    _ratio_metric(
                        f3 - f2,
                        f2,
                        denominator_name="period M2 physical FDE",
                    ),
                ),
            })
    return pd.DataFrame(period_rows)


def _run_exact_flat_campaign(
    *,
    case: Any,
    selected: Any,
    cards: dict[float, Any],
    gamma_values: tuple[float, ...],
    core_results: Path,
    results: Path,
    implementation_hash: str,
) -> int:
    """Represent the complete frozen grid without invoking an optimizer."""
    if not is_exact_flat_card(selected):
        raise RuntimeError("Exact-flat campaign called for a nonflat parent card.")
    results.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    completed: list[str] = []
    manifest = {
        "schema": "TPWRS_M3_GAMMA_RUN_R10_V1",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "scientific_fingerprint": implementation_hash,
        "physical_case_hash": case.physical_case_hash,
        "selected_parent_card_hash": selected.hash,
        "gamma_grid": list(gamma_values),
        "years": list(EVALUATION_YEARS),
        "action": "REUSE_EXACT_EQUIVALENT_CORE",
        "heavy_solve_count": 0,
        "completed_cases": completed,
        "cases": _case_states(gamma_values, tuple(EVALUATION_YEARS)),
    }
    status_path = results / "RUN_STATUS.json"
    atomic_write_json(status_path, manifest)
    for year in EVALUATION_YEARS:
        current = year_case(case, year)
        bundle = load_exact_core_bundle(core_results, year)
        solutions = bundle["solutions"]
        m2_solution = solutions["M2"]
        m3_solution = solutions["M3"]
        matched_solution = solutions["M2_MATCHED"]
        s2 = _require_finite_scalar(m2_solution.s_value_mw_h_row, "core M2 S")
        s3 = _require_finite_scalar(m3_solution.s_value_mw_h_row, "core M3 S")
        sm = _require_finite_scalar(matched_stage1_value(matched_solution), "core matched Stage-I S")
        f2 = float(np.sum(current.time_grid.delta[:, None] * m2_solution.x_mw / current.capacities_mw[None, :]))
        f3 = float(np.sum(current.time_grid.delta[:, None] * m3_solution.x_mw / current.capacities_mw[None, :]))
        m2_hourly = _hourly_residual(current, m2_solution.x_mw)
        m3_hourly = _hourly_residual(current, m3_solution.x_mw)
        matched_hourly = _hourly_residual(current, matched_solution.x_mw)
        for gamma in gamma_values:
            tag = _gamma_tag(gamma)
            key = f"{year}/gamma_{tag}"
            card = cards[gamma]
            certificate = exact_equivalence_certificate(
                year=year,
                variant_kind="gamma",
                variant_value=float(gamma),
                final_card=card,
                selected_parent_card=selected,
                core_bundle=bundle,
                physical_case_hash=current.physical_case_hash,
                scientific_fingerprint=implementation_hash,
            )
            case_root = results / f"year_{year}/gamma_{tag}"
            case_root.mkdir(parents=True, exist_ok=True)
            certificate_path = case_root / "EXACT_EQUIVALENCE_CERTIFICATE.json"
            certificate_file_hash = write_equivalence_certificate(certificate_path, certificate)
            hourly = pd.DataFrame({
                "year": year,
                "gamma": gamma,
                "hour_index": np.arange(current.n_time),
                "timestamp_utc": current.time_grid.timestamps_utc,
                "S_M2_hourly": m2_hourly,
                "S_M3_hourly": m3_hourly,
                "S_M2_MATCHED_hourly": matched_hourly,
                "M3_physical_FDE_hourly": np.sum(
                    current.time_grid.delta[:, None] * m3_solution.x_mw / current.capacities_mw[None, :], axis=1
                ),
                "M3_weighted_debit_hourly": np.sum(
                    current.time_grid.delta[:, None] * m3_solution.x_mw / current.capacities_mw[None, :], axis=1
                ),
            })
            bootstrap = circular_block_sums(
                {"S_M2": m2_hourly, "S_M3": m3_hourly, "S_M2_MATCHED": matched_hourly},
                draws=BOOTSTRAP_DRAWS,
                block_length=BLOCK_HOURS,
                seed=BOOTSTRAP_SEED + year * 1000 + int(round(gamma * 1000)),
            )
            ideal = bootstrap["S_M2"] - bootstrap["S_M2_MATCHED"]
            realized = bootstrap["S_M2"] - bootstrap["S_M3"]
            decomposition = frozen_matched_decomposition(s2, s3, sm)
            for column, values in _ratio_frame_columns(
                "matched_efficiency_pct",
                realized.to_numpy(float),
                ideal.to_numpy(float),
                denominator_name="bootstrap M2 minus M2_MATCHED residual",
                comparison_tolerance=decomposition["comparison_tolerance"],
            ).items():
                bootstrap[column] = values
            for column, values in _ratio_frame_columns(
                "security_relief_pct_vs_M2",
                realized.to_numpy(float),
                bootstrap["S_M2"].to_numpy(float),
                denominator_name="bootstrap M2 residual",
            ).items():
                bootstrap[column] = values
            summary_row = _exact_flat_summary_row(
                year=year, gamma=gamma, card=card, s2=s2, s3=s3, sm=sm, f2=f2, f3=f3
            )
            paths = {
                "hourly": case_root / "hourly_contributions.parquet",
                "bootstrap": case_root / "bootstrap_draws.csv",
                "bootstrap_summary": case_root / "matched_efficiency_bootstrap.json",
                "summary": case_root / "summary.csv",
                "exact_equivalence": certificate_path,
            }
            atomic_parquet(paths["hourly"], hourly)
            atomic_write_csv(paths["bootstrap"], bootstrap)
            atomic_write_json(
                paths["bootstrap_summary"],
                _bootstrap_ratio_record(
                    realized.to_numpy(float),
                    ideal.to_numpy(float),
                    comparison_tolerance=decomposition["comparison_tolerance"],
                ),
            )
            atomic_write_csv(paths["summary"], pd.DataFrame([summary_row]))
            artifacts = {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in paths.items()
            }
            _commit_case_status(
                case_root,
                implementation_hash=implementation_hash,
                year=year,
                gamma=gamma,
                summary_row=summary_row,
                artifacts=artifacts,
                equivalence={
                    "action": "REUSE_EXACT_EQUIVALENT_CORE",
                    "source_m3_solution_hash": m3_solution.solution_hash,
                    "source_matched_solution_hash": matched_solution.solution_hash,
                    "exact_equivalence_certificate_hash": certificate["certificate_hash"],
                    "exact_equivalence_certificate_file_sha256": certificate_file_hash,
                    "final_card_hash": card.hash,
                },
            )
            rows.append(summary_row)
            completed.append(key)
            manifest["cases"][key] = {"status": "CERTIFIED"}
            manifest["heartbeat_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
    yearly = pd.DataFrame(rows).sort_values(["gamma", "year"])
    period = _gamma_period_frame(yearly)
    atomic_write_csv(results / "M3_GAMMA_YEARLY.csv", yearly)
    atomic_write_csv(results / "M3_GAMMA_PERIOD.csv", period)
    final_status = {
        "schema": "TPWRS_M3_GAMMA_STATUS_R10_V1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "scientific_fingerprint": implementation_hash,
        "completed_case_count": len(yearly),
        "expected_case_count": len(gamma_values) * len(EVALUATION_YEARS),
        "selected_card_hash": selected.hash,
        "gamma_grid": list(gamma_values),
        "all_cases_action": "REUSE_EXACT_EQUIVALENT_CORE",
        "heavy_solve_count": 0,
        "exact_flat_parent_equivalence": True,
        "outputs": {
            "yearly": {"path": "M3_GAMMA_YEARLY.csv", "sha256": sha256_file(results / "M3_GAMMA_YEARLY.csv")},
            "period": {"path": "M3_GAMMA_PERIOD.csv", "sha256": sha256_file(results / "M3_GAMMA_PERIOD.csv")},
        },
    }
    atomic_write_json(results / "M3_GAMMA_STATUS.json", final_status)
    manifest.update(status="COMPLETE", completed_utc=utc_now())
    atomic_write_json(status_path, manifest)
    print("M3_GAMMA_PATH=COMPLETE EXACT_FLAT_REUSE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exact corrected post-freeze rho=0.03 M3 gamma path with "
            "M2-MATCHED at every non-reused point."
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
        default=ROOT / "results_m3_sensitivity_r10/gamma",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="*",
        default=list(DEFAULT_GAMMA_GRID),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gamma_values = tuple(map(float, args.gamma))
    if gamma_values!=DEFAULT_GAMMA_GRID:
        raise RuntimeError("Gamma grid must equal the frozen methodology grid.")
    if not gamma_values:
        raise RuntimeError("Gamma grid is empty.")
    if len(set(gamma_values)) != len(gamma_values):
        raise RuntimeError("Gamma grid contains duplicate values.")

    case = load_case(args.case)
    core_status, _ = require_compatible_status(
        args.core_results / "M3_CORE_STATUS.json",
        allowed=("COMPLETE",),
        use="UPSTREAM_STAGE05_CORE_NUMERICAL_EVIDENCE",
    )
    implementation_hash = scientific_fingerprint()
    if core_status.get("status") != "COMPLETE":
        raise RuntimeError("Corrected M3 core campaign must complete first.")

    selected, _, model_path = load_frozen_card(case, args.cards, 0.03)
    exact_flat_parent = is_exact_flat_card(selected)
    if exact_flat_parent:
        reference_weights = None
        fit = None
    else:
        from jobs.common import verify_fitting_tables
        verify_fitting_tables(selected,args.cards,args.features)
        reference_weights, fit = _training_reference(
            args.cards,
            args.features,
            selected,
            model_path,
        )
    card_root = args.cards / "sensitivity_r10/gamma"
    card_root.mkdir(parents=True, exist_ok=True)
    cards = {
        gamma: _gamma_card(
            selected,
            reference_weights,
            gamma,
            card_root,
        )
        for gamma in gamma_values
    }
    if fit is not None:
        atomic_parquet(card_root / "GAMMA_FITTING_REFERENCE.parquet", fit)
    else:
        atomic_write_json(
            card_root / "EXACT_FLAT_PARENT_FREEZE.json",
            {
                "schema": "TPWRS_R10_EXACT_FLAT_GAMMA_FREEZE_V1",
                "status": "PASS",
                "selected_parent_card_hash": selected.hash,
                "rates_bitwise_one": True,
                "gamma_grid": list(gamma_values),
                "mode": "EXACT_FLAT_PARENT_EQUIVALENCE",
            },
        )

    args.results.mkdir(parents=True, exist_ok=True)
    new_points = 0 if exact_flat_parent else sum(
        1 for gamma in gamma_values if not np.isclose(gamma, 1.0)
    )
    plan = {
        "schema": "TPWRS_M3_GAMMA_PLAN_V1",
        "status": "PASS",
        "created_utc": utc_now(),
        "gamma_grid": gamma_values,
        "years": EVALUATION_YEARS,
        "gamma_one_reused_from_core": 1.0 in gamma_values,
        "exact_flat_parent_equivalence": exact_flat_parent,
        "new_heavy_case_count": new_points * len(EVALUATION_YEARS) * 2,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "block_hours": BLOCK_HOURS,
        "no_scientific_optimization": bool(args.dry_run),
    }
    atomic_write_json(args.results / "DRY_RUN_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if exact_flat_parent:
        return _run_exact_flat_campaign(
            case=case,
            selected=selected,
            cards=cards,
            gamma_values=gamma_values,
            core_results=args.core_results,
            results=args.results,
            implementation_hash=implementation_hash,
        )

    identity = {
        "physical_case_hash": case.physical_case_hash,
        "selected_card_hash": selected.hash,
        "gamma_cards": {str(key): value.hash for key, value in cards.items()},
        "gamma_grid": gamma_values,
        "years": EVALUATION_YEARS,
        "implementation_hash": implementation_hash,
        "heavy_concurrency": 1,
    }
    campaign_hash = sha256_json(identity)
    status_path = args.results / "RUN_STATUS.json"

    with CampaignLock(args.results / ".campaign.lock", campaign_hash):
        if status_path.is_file():
            manifest = json.loads(status_path.read_text(encoding="utf-8"))
            if manifest.get("campaign_hash") != campaign_hash:
                raise RuntimeError("Gamma result root belongs to another campaign.")
            manifest["status"] = "RUNNING"
            manifest["resumed_utc"] = utc_now()
            manifest["invocation_count"] = int(
                manifest.get("invocation_count", 1)
            ) + 1
            print(
                "M3_GAMMA=AUTO_RESUME "
                f"completed={len(manifest.get('completed_cases', []))}",
                flush=True,
            )
        else:
            manifest = {
                "schema": "TPWRS_M3_GAMMA_RUN_V1",
                "status": "RUNNING",
                "started_utc": utc_now(),
                "invocation_count": 1,
                "campaign_hash": campaign_hash,
                **identity,
                "jobs": {},
                "completed_cases": [],
            }
            print(
                f"M3_GAMMA=NEW_RUN gamma_points={len(gamma_values)}",
                flush=True,
            )
        expected_states = _case_states(
            gamma_values,
            tuple(map(int, EVALUATION_YEARS)),
            manifest.get("completed_cases", []),
        )
        for key, state in manifest.get("cases", {}).items():
            if key in expected_states and state.get("status") != "CERTIFIED":
                expected_states[key] = dict(state)
        manifest["cases"] = expected_states
        atomic_write_json(status_path, manifest)

        summary_rows: list[dict[str, Any]] = []
        active_case_key: str | None = None
        methods_certified = False
        try:
            for year,execution_order in gamma_execution_schedule(gamma_values):
                current = year_case(case, year)
                _, score = q_for_year(
                    current,
                    year,
                    selected,
                    args.cards,
                    args.features,
                )
                baseline_solution = load_policy_solution(
                    args.core_results / f"year_{year}/rho_0.03/annual/M2"
                )
                s2 = float(baseline_solution.s_value_mw_h_row)
                f2 = float(
                    np.sum(
                        current.time_grid.delta[:, None]
                        * baseline_solution.x_mw
                        / current.capacities_mw[None, :]
                    )
                )
                m2_hourly = _hourly_residual(current, baseline_solution.x_mw)

                for gamma in execution_order:
                    tag = _gamma_tag(gamma)
                    active_case_key = f"{year}/gamma_{tag}"
                    methods_certified = False
                    case_root = args.results / f"year_{year}/gamma_{tag}"
                    completed = _completed_case(case_root)
                    if completed is not None:
                        summary_rows.append(dict(completed["summary_row"]))
                        manifest["cases"][active_case_key] = {
                            "status": "CERTIFIED"
                        }
                        print(
                            f"[GAMMA] year={year} gamma={gamma:g} REUSE",
                            flush=True,
                        )
                        continue

                    manifest["cases"][active_case_key] = {
                        "status": "RUNNING",
                        "started_utc": utc_now(),
                    }
                    manifest["heartbeat_utc"] = utc_now()
                    atomic_write_json(status_path, manifest)

                    card = cards[gamma]
                    q = q_from_card(
                        card,
                        current.participant_directions,
                        score,
                    )

                    if np.isclose(gamma, 1.0, atol=1e-12, rtol=0.0):
                        m3_solution = load_policy_solution(
                            args.core_results
                            / f"year_{year}/rho_0.03/annual/M3"
                        )
                        matched_solution = load_policy_solution(
                            args.core_results
                            / f"year_{year}/rho_0.03/annual/M2_MATCHED"
                        )
                        if not np.array_equal(q, m3_solution.q):
                            raise RuntimeError(
                                "gamma=1 q matrix does not reproduce the frozen "
                                "selected core M3 solution."
                            )
                        action = "REUSE_CORE_ENDPOINT"
                    else:
                        m3_relative = Path(f"year_{year}/gamma_{tag}/M3")
                        m3_experiment = make_experiment(
                            current,
                            case_id=f"Y{year}_RHO03_M3_GAMMA_{tag}",
                            rule="M3",
                            rho=0.03,
                            mode="annual",
                            result_root_id=str(m3_relative).replace("\\", "/"),
                            implementation_hash=implementation_hash,
                            card=card,
                            q=q,
                            information_level="POST_FREEZE_GAMMA_PATH",
                        )
                        m3_solution, _ = run_annual(
                            current,
                            m3_experiment,
                            q,
                            args.results / m3_relative,
                        )
                        matched = matched_allowance(current, m3_solution)
                        if not verify_m2_matched_contains_m3(
                            current,
                            m3_solution,
                            matched,
                        ):
                            raise RuntimeError(
                                f"{year}/gamma={gamma}: matched allowance does not contain M3."
                            )
                        matched_relative = Path(
                            f"year_{year}/gamma_{tag}/M2_MATCHED"
                        )
                        matched_experiment = make_experiment(
                            current,
                            case_id=f"Y{year}_RHO03_M2_MATCHED_GAMMA_{tag}",
                            rule="M2_MATCHED",
                            rho=0.03,
                            mode="annual",
                            result_root_id=str(matched_relative).replace("\\", "/"),
                            implementation_hash=implementation_hash,
                            card=card,
                            allowance=matched,
                            information_level="POST_M3_MATCHED_GAMMA_PATH",
                        )
                        matched_solution, _ = run_annual(
                            current,
                            matched_experiment,
                            None,
                            args.results / matched_relative,
                        )
                        action = "RUN_OR_REUSE"

                    methods_certified = True
                    _require_finite_array(m3_solution.x_mw, "M3 policy")
                    _require_finite_array(
                        matched_solution.x_mw,
                        "M2_MATCHED policy",
                    )
                    if m3_solution.q is None:
                        raise ValueError("Required M3 debit-rate array is absent.")
                    _require_finite_array(m3_solution.q, "M3 debit-rate array")
                    _require_finite_array(
                        matched_solution.stage1.objective_value,
                        "M2_MATCHED Stage-I objective",
                    )
                    s2 = _require_finite_scalar(s2, "M2 Stage-I replay")
                    s3 = _require_finite_scalar(
                        m3_solution.s_value_mw_h_row,
                        "M3 Stage-I replay",
                    )
                    sm = _require_finite_scalar(
                        matched_solution.stage1.objective_value,
                        "M2_MATCHED Stage-I objective",
                    )
                    if np.isclose(gamma, 0.0, atol=1e-12, rtol=0.0):
                        tol = max(2e-6, 10.0 * current.time_grid.uniform_delta * 1e-8)
                        if abs(s3 - s2) > tol:
                            raise RuntimeError(
                                f"gamma=0 must reproduce the flat M2 Stage-I value: M2={s2}, gamma0={s3}."
                            )
                    decomposition = frozen_matched_decomposition(s2, s3, sm)
                    f3 = float(
                        np.sum(
                            current.time_grid.delta[:, None]
                            * m3_solution.x_mw
                            / current.capacities_mw[None, :]
                        )
                    )
                    m3_hourly = _hourly_residual(current, m3_solution.x_mw)
                    matched_hourly = _hourly_residual(
                        current,
                        matched_solution.x_mw,
                    )
                    hourly = pd.DataFrame(
                        {
                            "year": year,
                            "gamma": gamma,
                            "hour_index": np.arange(current.n_time),
                            "timestamp_utc": current.time_grid.timestamps_utc,
                            "S_M2_hourly": m2_hourly,
                            "S_M3_hourly": m3_hourly,
                            "S_M2_MATCHED_hourly": matched_hourly,
                            "M3_physical_FDE_hourly": np.sum(
                                current.time_grid.delta[:, None]
                                * m3_solution.x_mw
                                / current.capacities_mw[None, :],
                                axis=1,
                            ),
                            "M3_weighted_debit_hourly": np.sum(
                                current.time_grid.delta[:, None]
                                * q
                                * m3_solution.x_mw
                                / current.capacities_mw[None, :],
                                axis=1,
                            ),
                        }
                    )
                    bootstrap = circular_block_sums(
                        {
                            "S_M2": m2_hourly,
                            "S_M3": m3_hourly,
                            "S_M2_MATCHED": matched_hourly,
                        },
                        draws=BOOTSTRAP_DRAWS,
                        block_length=BLOCK_HOURS,
                        seed=(
                            BOOTSTRAP_SEED
                            + year * 1000
                            + int(round(gamma * 1000))
                        ),
                    )
                    ideal = bootstrap["S_M2"] - bootstrap["S_M2_MATCHED"]
                    realized = bootstrap["S_M2"] - bootstrap["S_M3"]
                    for key, values in _ratio_frame_columns(
                        "matched_efficiency_pct",
                        realized.to_numpy(float),
                        ideal.to_numpy(float),
                        denominator_name="bootstrap M2 minus M2_MATCHED residual",
                        comparison_tolerance=decomposition["comparison_tolerance"],
                    ).items():
                        bootstrap[key] = values
                    for key, values in _ratio_frame_columns(
                        "security_relief_pct_vs_M2",
                        realized.to_numpy(float),
                        bootstrap["S_M2"].to_numpy(float),
                        denominator_name="bootstrap M2 residual",
                    ).items():
                        bootstrap[key] = values

                    summary_row = {
                        "year": year,
                        "gamma": gamma,
                        "card_id": card.card_id,
                        "card_hash": card.hash,
                        "q_min_realized": float(np.min(q)),
                        "q_max_realized": float(np.max(q)),
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
                        **decomposition,
                        "action": action,
                    }
                    case_root.mkdir(parents=True, exist_ok=True)
                    atomic_parquet(
                        case_root / "hourly_contributions.parquet",
                        hourly,
                    )
                    atomic_write_csv(
                        case_root / "bootstrap_draws.csv",
                        bootstrap,
                    )
                    bootstrap_record = _bootstrap_ratio_record(
                        realized.to_numpy(float),
                        ideal.to_numpy(float),
                        comparison_tolerance=decomposition[
                            "comparison_tolerance"
                        ],
                    )
                    atomic_write_json(
                        case_root / "matched_efficiency_bootstrap.json",
                        bootstrap_record,
                    )
                    atomic_write_csv(
                        case_root / "summary.csv",
                        pd.DataFrame([summary_row]),
                    )
                    artifacts = {
                        name: {
                            "path": path.name,
                            "sha256": sha256_file(path),
                        }
                        for name, path in {
                            "hourly": case_root / "hourly_contributions.parquet",
                            "bootstrap": case_root / "bootstrap_draws.csv",
                            "bootstrap_summary": (
                                case_root / "matched_efficiency_bootstrap.json"
                            ),
                            "summary": case_root / "summary.csv",
                        }.items()
                    }
                    _commit_case_status(
                        case_root,
                        implementation_hash=implementation_hash,
                        year=year,
                        gamma=gamma,
                        summary_row=summary_row,
                        artifacts=artifacts,
                    )
                    summary_rows.append(summary_row)
                    manifest.setdefault("completed_cases", []).append(
                        f"{year}/gamma_{tag}"
                    )
                    manifest["cases"][active_case_key] = {
                        "status": "CERTIFIED",
                        "certified_utc": utc_now(),
                    }
                    manifest["heartbeat_utc"] = utc_now()
                    atomic_write_json(status_path, manifest)
                    print(
                        f"[GAMMA] year={year} gamma={gamma:g} "
                        f"relief={summary_row['security_relief_pct_vs_M2']:.4g}% "
                        f"FDE={summary_row['physical_FDE_change_pct_vs_M2']:.4g}%",
                        flush=True,
                    )

            yearly = pd.DataFrame(summary_rows).drop_duplicates(
                ["year", "gamma"], keep="last"
            ).sort_values(["gamma", "year"])
            period_rows = []
            for gamma, part in yearly.groupby("gamma", sort=True):
                for period, years in (
                    ("2029-2030 held-out", (2029, 2030)),
                    ("2031 confirmatory", (2031,)),
                ):
                    subset = part.loc[part["year"].isin(years)]
                    if len(subset) != len(years):
                        continue
                    s2 = float(subset["S_M2"].sum())
                    s3 = float(subset["S_M3"].sum())
                    sm = float(subset["S_M2_MATCHED"].sum())
                    f2 = float(subset["physical_FDE_M2_h"].sum())
                    f3 = float(subset["physical_FDE_M3_h"].sum())
                    period_decomposition = frozen_matched_decomposition(s2, s3, sm)
                    period_rows.append(
                        {
                            "period": period,
                            "gamma": gamma,
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
                                    denominator_name="period M2 residual",
                                ),
                            ),
                            **_ratio_columns(
                                "matched_efficiency_pct",
                                _ratio_metric(
                                    s2 - s3,
                                    s2 - sm,
                                    denominator_name=(
                                        "period M2 minus M2_MATCHED residual"
                                    ),
                                    comparison_tolerance=period_decomposition[
                                        "comparison_tolerance"
                                    ],
                                ),
                                claimable_name="matched_efficiency_claimable",
                            ),
                            "state_shape_penalty": period_decomposition["shape_penalty"],
                            "raw_shape_residual": period_decomposition["raw_shape_residual"],
                            **_ratio_columns(
                                "physical_FDE_change_pct_vs_M2",
                                _ratio_metric(
                                    f3 - f2,
                                    f2,
                                    denominator_name="period M2 physical FDE",
                                ),
                            ),
                        }
                    )
            period = pd.DataFrame(period_rows)
            atomic_write_csv(args.results / "M3_GAMMA_YEARLY.csv", yearly)
            atomic_write_csv(args.results / "M3_GAMMA_PERIOD.csv", period)
            atomic_write_json(
                args.results / "M3_GAMMA_STATUS.json",
                {
                    "schema": "TPWRS_M3_GAMMA_STATUS_V1",
                    "status": "COMPLETE",
                    "created_utc": utc_now(),
                    "scientific_fingerprint": implementation_hash,
                    "completed_case_count": len(yearly),
                    "expected_case_count": (
                        len(gamma_values) * len(EVALUATION_YEARS)
                    ),
                    "selected_card_hash": selected.hash,
                    "gamma_grid": gamma_values,
                    "screen_gammas": SCREEN_GAMMAS,
                    "screen_completed_first": True,
                    "gamma_zero_reproduces_flat_m2": True,
                    "gamma_one_reused_from_frozen_core": True,
                    "bootstrap_interpretation": (
                        "FIXED-POLICY TEMPORAL BLOCK RESAMPLING; "
                        "NOT AN OPTIMIZATION-GAP OR GENERALIZATION INTERVAL"
                    ),
                    "claim_boundary": (
                        "POST-FREEZE RATE-AMPLITUDE SENSITIVITY; NO REFITTING "
                        "OR CARD RESELECTION."
                    ),
                },
            )
            manifest["status"] = "COMPLETE"
            manifest["completed_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
            print("M3_GAMMA_PATH=COMPLETE", flush=True)
            return 0
        except BaseException as exc:
            if active_case_key is not None and manifest["cases"].get(
                active_case_key, {}
            ).get("status") != "CERTIFIED":
                manifest["cases"][active_case_key] = {
                    "status": (
                        "REPORTING_FAILED" if methods_certified else "FAILED"
                    ),
                    "failed_utc": utc_now(),
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            manifest["status"] = "FAILED"
            manifest["failed_utc"] = utc_now()
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(status_path, manifest)
            raise
