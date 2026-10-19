from __future__ import annotations


# ============================================================================
# scripts.run_final_m3_core (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from jobs.common import EVALUATION_YEARS, exact_profile, frozen_matched_decomposition, load_case, load_frozen_card, make_experiment, q_for_year, require_m1_m2_freeze, run_annual, run_rolling_case, utc_now, verify_card_allowance, year_case
from tpwrs.execution import CampaignLock
from tpwrs.config import scientific_fingerprint
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.cards import matched_allowance, verify_m2_matched_contains_m3



def _job_update(
    manifest: dict[str, Any],
    status_path: Path,
    job_id: str,
    status: str,
    **extra: Any,
) -> None:
    jobs = manifest.setdefault("jobs", {})
    current = dict(jobs.get(job_id, {}))
    current.update({"status": status, "updated_utc": utc_now(), **extra})
    jobs[job_id] = current
    manifest["heartbeat_utc"] = utc_now()
    manifest["completed_cases"] = sorted(
        key for key, value in jobs.items() if value.get("status") == "PASS"
    )
    manifest["failed_cases"] = sorted(
        key for key, value in jobs.items() if value.get("status") == "FAILED"
    )
    atomic_write_json(status_path, manifest)


def _baseline_metric(
    root: Path,
    year: int,
    *,
    mode: str,
    rule: str,
) -> dict[str, Any]:
    path = (
        root
        / f"year_{year}"
        / "rho_0.15"
        / mode
        / rule
        / ("metrics.json" if mode == "annual" else "rolling_result.json")
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("replay_passed") is not True:
        raise RuntimeError(f"Baseline {year}/{mode}/{rule} did not pass replay.")
    return payload


def _annual_case(
    *,
    case: Any,
    year: int,
    rho: float,
    rule: str,
    card: Any | None,
    q: np.ndarray | None,
    implementation_hash: str,
    results: Path,
    allowance: Any | None = None,
) -> tuple[Any, str, Path]:
    relative = Path(f"year_{year}/rho_{rho:.2f}/annual/{rule}")
    experiment = make_experiment(
        case,
        case_id=f"Y{year}_RHO{int(round(rho * 100)):02d}_{rule}",
        rule=rule,
        rho=rho,
        mode="annual",
        result_root_id=str(relative).replace("\\", "/"),
        implementation_hash=implementation_hash,
        card=card if rule in {"M3", "M2_MATCHED"} else None,
        q=q if rule == "M3" else None,
        allowance=allowance,
        information_level=(
            "FROZEN_F0"
            if rule == "M3"
            else (
                "POST_M3_MATCHED_EXPOSURE"
                if rule == "M2_MATCHED"
                else "FROZEN_INPUT"
            )
        ),
    )
    solution, action = run_annual(
        case,
        experiment,
        q if rule == "M3" else None,
        results / relative,
    )
    return solution, action, relative


def _rolling_case(
    *,
    case: Any,
    year: int,
    rho: float,
    rule: str,
    card: Any | None,
    q: np.ndarray | None,
    implementation_hash: str,
    results: Path,
) -> tuple[dict[str, Any], str, Path]:
    relative = Path(f"year_{year}/rho_{rho:.2f}/rolling/{rule}")
    experiment = make_experiment(
        case,
        case_id=f"Y{year}_RHO{int(round(rho * 100)):02d}_ROLL_{rule}",
        rule=rule,
        rho=rho,
        mode="rolling",
        result_root_id=str(relative).replace("\\", "/"),
        implementation_hash=implementation_hash,
        card=card if rule == "M3" else None,
        q=q if rule == "M3" else None,
        information_level="FROZEN_F0" if rule == "M3" else "FROZEN_INPUT",
    )
    metrics, action = run_rolling_case(
        case,
        experiment,
        q if rule == "M3" else None,
        results / relative,
    )
    return metrics, action, relative


def _compile_summary(
    baseline_root: Path,
    results: Path,
    implementation_hash: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for year in EVALUATION_YEARS:
        for rho in (0.15, 0.03):
            if rho == 0.15:
                m2_annual = _baseline_metric(
                    baseline_root, year, mode="annual", rule="M2"
                )
                m2_rolling = _baseline_metric(
                    baseline_root, year, mode="rolling", rule="M2"
                )
            else:
                m2_annual = json.loads(
                    (
                        results
                        / f"year_{year}/rho_0.03/annual/M2/metrics.json"
                    ).read_text(encoding="utf-8")
                )
                m2_rolling = json.loads(
                    (
                        results
                        / f"year_{year}/rho_0.03/rolling/M2/rolling_result.json"
                    ).read_text(encoding="utf-8")
                )

            m3_annual = json.loads(
                (
                    results
                    / f"year_{year}/rho_{rho:.2f}/annual/M3/metrics.json"
                ).read_text(encoding="utf-8")
            )
            m3_rolling = json.loads(
                (
                    results
                    / f"year_{year}/rho_{rho:.2f}/rolling/M3/rolling_result.json"
                ).read_text(encoding="utf-8")
            )

            for mode, m2, m3 in (
                ("annual", m2_annual, m3_annual),
                ("rolling", m2_rolling, m3_rolling),
            ):
                s2 = float(m2["S_mw_h_row"])
                s3 = float(m3["S_mw_h_row"])
                f2 = float(m2["physical_fde_sum_h"])
                f3 = float(m3["physical_fde_sum_h"])
                rows.append(
                    {
                        "year": year,
                        "rho": rho,
                        "mode": mode,
                        "S_M2": s2,
                        "S_M3": s3,
                        "security_relief_pct_vs_M2": (
                            100.0 * (s2 - s3) / s2 if s2 > 1e-12 else np.nan
                        ),
                        "physical_FDE_M2_h": f2,
                        "physical_FDE_M3_h": f3,
                        "physical_FDE_change_pct_vs_M2": (
                            100.0 * (f3 - f2) / f2 if f2 > 1e-12 else np.nan
                        ),
                        "restricted_MWh_M2": float(m2["E_restricted_mwh"]),
                        "restricted_MWh_M3": float(m3["E_restricted_mwh"]),
                    }
                )

            if rho == 0.03:
                matched = json.loads(
                    (
                        results
                        / f"year_{year}/rho_0.03/annual/M2_MATCHED/metrics.json"
                    ).read_text(encoding="utf-8")
                )
                rows[-2]["S_M2_MATCHED"] = float(matched["S_mw_h_row"])
                rows[-2].update(matched.get("matched_decomposition", {}))

    frame = pd.DataFrame(rows)
    atomic_write_csv(results / "M3_CORE_SUMMARY.csv", frame)
    atomic_write_json(
        results / "M3_CORE_STATUS.json",
        {
            "schema": "TPWRS_M3_CORE_STATUS_V1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "scientific_fingerprint": implementation_hash,
            "rows": len(frame),
            "outputs": {
                "summary": {
                    "path": "M3_CORE_SUMMARY.csv",
                    "sha256": sha256_file(results / "M3_CORE_SUMMARY.csv"),
                }
            },
            "claim_boundary": (
                "M3 SECURITY RESULTS MUST BE INTERPRETED JOINTLY WITH PHYSICAL "
                "FDE AND, AT RHO=0.03, THE M2-MATCHED DECOMPOSITION."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Corrected V3 M3 core campaign. Reuses the separately frozen "
            "rho=0.15 M1/M2 evidence and never overwrites it."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=ROOT / "artifacts/real_case/physical_case/FINAL_DUTCH_CASE",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "results_m1_m2_final",
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
        "--results",
        type=Path,
        default=ROOT / "results_m3_final",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case = load_case(args.case)
    baseline_freeze = require_m1_m2_freeze(args.baseline, case)
    cards: dict[float, Any] = {}
    freezes: dict[float, dict[str, Any]] = {}
    for rho in (0.15, 0.03):
        cards[rho], freezes[rho], _ = load_frozen_card(case, args.cards, rho)

    implementation_hash = scientific_fingerprint()
    plan = {
        "schema": "TPWRS_M3_CORE_PLAN_V1",
        "status": "PASS",
        "created_utc": utc_now(),
        "physical_case_hash": case.physical_case_hash,
        "m1_m2_baseline_freeze_sha256": sha256_file(
            args.baseline / "M1_M2_EVIDENCE_FREEZE.json"
        ),
        "jobs": [
            *[
                {
                    "year": year,
                    "rho": 0.15,
                    "annual": ["M3"],
                    "rolling": ["M3"],
                }
                for year in EVALUATION_YEARS
            ],
            *[
                {
                    "year": year,
                    "rho": 0.03,
                    "annual": ["M2", "M3", "M2_MATCHED"],
                    "rolling": ["M2", "M3"],
                }
                for year in EVALUATION_YEARS
            ],
        ],
        "new_heavy_case_count": 21,
        "rolling_stage3_used": False,
        "annual_stage3_used": False,
        "controls_in_separate_campaign": True,
        "no_scientific_optimization": bool(args.dry_run),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.results / "DRY_RUN_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    campaign_identity = {
        "physical_case_hash": case.physical_case_hash,
        "baseline_freeze_hash": sha256_file(
            args.baseline / "M1_M2_EVIDENCE_FREEZE.json"
        ),
        "cards": {str(rho): cards[rho].hash for rho in cards},
        "implementation_hash": implementation_hash,
        "years": EVALUATION_YEARS,
        "heavy_concurrency": 1,
        "annual_stage3": False,
        "rolling_stage3": False,
    }
    campaign_hash = sha256_json(campaign_identity)
    status_path = args.results / "RUN_STATUS.json"

    with CampaignLock(args.results / ".campaign.lock", campaign_hash):
        if status_path.is_file():
            manifest = json.loads(status_path.read_text(encoding="utf-8"))
            if manifest.get("campaign_hash") != campaign_hash:
                raise RuntimeError(
                    "Existing M3 result root belongs to another immutable campaign."
                )
            manifest["status"] = "RUNNING"
            manifest["resumed_utc"] = utc_now()
            manifest["invocation_count"] = int(
                manifest.get("invocation_count", 1)
            ) + 1
            print(
                "M3_CORE=AUTO_RESUME "
                f"completed={len(manifest.get('completed_cases', []))}/21",
                flush=True,
            )
        else:
            manifest = {
                "schema": "TPWRS_M3_CORE_RUN_V1",
                "status": "RUNNING",
                "started_utc": utc_now(),
                "invocation_count": 1,
                "campaign_hash": campaign_hash,
                **campaign_identity,
                "physical_case_id": case.physical_case_id,
                "baseline_run_campaign_hash": baseline_freeze.get(
                    "run_campaign_hash"
                ),
                "jobs": {},
                "completed_cases": [],
                "failed_cases": [],
            }
            print("M3_CORE=NEW_RUN total_new_heavy_cases=21", flush=True)
        atomic_write_json(status_path, manifest)

        try:
            for year in EVALUATION_YEARS:
                current = year_case(case, year)
                q_by_rho: dict[float, np.ndarray] = {}
                for rho in (0.15, 0.03):
                    q, _ = q_for_year(
                        current,
                        year,
                        cards[rho],
                        args.cards,
                        args.features,
                    )
                    q_by_rho[rho] = q
                    allowance_probe = make_experiment(
                        current,
                        case_id=f"Y{year}_RHO{int(round(rho*100)):02d}_ALLOWANCE_PROBE",
                        rule="M2",
                        rho=rho,
                        mode="annual",
                        result_root_id="ALLOWANCE_PROBE",
                        implementation_hash=implementation_hash,
                    )
                    verify_card_allowance(
                        allowance_probe,
                        cards[rho],
                        freezes[rho],
                        year,
                    )

                # rho=0.15: only M3 is new; M2 comes from the frozen baseline.
                for mode in ("annual", "rolling"):
                    job_id = f"Y{year}_RHO15_{mode.upper()}_M3"
                    _job_update(manifest, status_path, job_id, "RUNNING")
                    if mode == "annual":
                        solution, action, relative = _annual_case(
                            case=current,
                            year=year,
                            rho=0.15,
                            rule="M3",
                            card=cards[0.15],
                            q=q_by_rho[0.15],
                            implementation_hash=implementation_hash,
                            results=args.results,
                        )
                        solution_hash = solution.solution_hash
                    else:
                        metrics, action, relative = _rolling_case(
                            case=current,
                            year=year,
                            rho=0.15,
                            rule="M3",
                            card=cards[0.15],
                            q=q_by_rho[0.15],
                            implementation_hash=implementation_hash,
                            results=args.results,
                        )
                        solution_hash = None
                    _job_update(
                        manifest,
                        status_path,
                        job_id,
                        "PASS",
                        action=action,
                        output=str(relative),
                        solution_hash=solution_hash,
                    )

                # rho=0.03: M2, M3, M2-MATCHED and rolling M2/M3.
                annual: dict[str, Any] = {}
                for rule in ("M2", "M3"):
                    job_id = f"Y{year}_RHO03_ANNUAL_{rule}"
                    _job_update(manifest, status_path, job_id, "RUNNING")
                    solution, action, relative = _annual_case(
                        case=current,
                        year=year,
                        rho=0.03,
                        rule=rule,
                        card=cards[0.03] if rule == "M3" else None,
                        q=q_by_rho[0.03] if rule == "M3" else None,
                        implementation_hash=implementation_hash,
                        results=args.results,
                    )
                    annual[rule] = solution
                    _job_update(
                        manifest,
                        status_path,
                        job_id,
                        "PASS",
                        action=action,
                        output=str(relative),
                        solution_hash=solution.solution_hash,
                    )

                matched = matched_allowance(current, annual["M3"])
                if not verify_m2_matched_contains_m3(
                    current,
                    annual["M3"],
                    matched,
                ):
                    raise RuntimeError(
                        f"{year}: M2-MATCHED allowance does not contain M3."
                    )
                job_id = f"Y{year}_RHO03_ANNUAL_M2_MATCHED"
                _job_update(manifest, status_path, job_id, "RUNNING")
                matched_solution, action, relative = _annual_case(
                    case=current,
                    year=year,
                    rho=0.03,
                    rule="M2_MATCHED",
                    card=cards[0.03],
                    q=None,
                    implementation_hash=implementation_hash,
                    results=args.results,
                    allowance=matched,
                )
                decomposition = frozen_matched_decomposition(
                    annual["M2"].s_value_mw_h_row,
                    annual["M3"].s_value_mw_h_row,
                    matched_solution.stage1.objective_value,
                )
                metrics_path = args.results / relative / "metrics.json"
                matched_metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                matched_metrics["matched_decomposition"] = {
                    "S_M2": annual["M2"].s_value_mw_h_row,
                    "S_M3": annual["M3"].s_value_mw_h_row,
                    "S_M2_MATCHED": matched_solution.stage1.objective_value,
                    **decomposition,
                }
                atomic_write_json(metrics_path, matched_metrics)
                _job_update(
                    manifest,
                    status_path,
                    job_id,
                    "PASS",
                    action=action,
                    output=str(relative),
                    solution_hash=matched_solution.solution_hash,
                )

                for rule in ("M2", "M3"):
                    job_id = f"Y{year}_RHO03_ROLLING_{rule}"
                    _job_update(manifest, status_path, job_id, "RUNNING")
                    metrics, action, relative = _rolling_case(
                        case=current,
                        year=year,
                        rho=0.03,
                        rule=rule,
                        card=cards[0.03] if rule == "M3" else None,
                        q=q_by_rho[0.03] if rule == "M3" else None,
                        implementation_hash=implementation_hash,
                        results=args.results,
                    )
                    _job_update(
                        manifest,
                        status_path,
                        job_id,
                        "PASS",
                        action=action,
                        output=str(relative),
                    )

            _compile_summary(args.baseline, args.results, implementation_hash)
            manifest["status"] = "COMPLETE"
            manifest["completed_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
            print("M3_CORE=COMPLETE", flush=True)
            return 0
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["failed_utc"] = utc_now()
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(status_path, manifest)
            raise
