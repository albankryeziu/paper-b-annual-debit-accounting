from __future__ import annotations


# ============================================================================
# scripts.run_m3_controls (consolidated; see SOURCE_MAP.json)
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

from jobs.common import EVALUATION_YEARS, load_case, load_frozen_card, make_experiment, q_for_year, run_annual, utc_now, year_case
from tpwrs.execution import CampaignLock
from tpwrs.config import scientific_fingerprint
from tpwrs.cards import frozen_shuffle_scores, inverse_order, ptdf_net_relief_scores
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.cards import validate_preheldout_audit
from tpwrs.persistence import load_card
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.cards import q_from_card


SHUFFLE_SEEDS = tuple(range(118001, 118033))
CONTROL_KEYS = (
    "PTDF",
    "INVERSE",
    *(f"SHUFFLE_{seed}" for seed in SHUFFLE_SEEDS),
)

def _load_controls(
    cards_root: Path,
    selected_card: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = cards_root / "RHO03_CONTROL_CARD_FREEZE.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "TPWRS_RHO03_CONTROL_CARD_FREEZE_V2":
        raise RuntimeError("Control-card freeze schema is not the current contract.")
    if payload.get("status") != "PASS":
        raise RuntimeError("Control-card freeze is not PASS.")
    if payload.get("selected_card_hash") != selected_card.hash:
        raise RuntimeError("Control-card freeze selected-card hash mismatch.")
    audit_value = payload.get("heldout_access_audit_path")
    audit_hash = payload.get("heldout_access_audit_sha256")
    if not audit_value or not audit_hash:
        raise RuntimeError("Control-card freeze lacks a bound held-out audit artifact.")
    audit_path = Path(audit_value)
    if not audit_path.is_absolute():
        audit_path = cards_root / audit_path
    try:
        audit_payload, validated_hash = validate_preheldout_audit(audit_path)
    except Exception as exc:
        raise RuntimeError("Control-card held-out audit artifact is invalid.") from exc
    if validated_hash != audit_hash or payload.get("heldout_access_audit_schema") != audit_payload.get("schema"):
        raise RuntimeError("Control-card held-out audit binding mismatch.")
    if payload.get("scientific_fingerprint") != scientific_fingerprint():
        raise RuntimeError("Control-card freeze belongs to different scientific source code.")
    if tuple(map(int, payload.get("shuffle_seeds", ()))) != SHUFFLE_SEEDS:
        raise RuntimeError("Frozen shuffle seed registry differs from protocol.")

    expected = {"PTDF", "INVERSE", *CONTROL_KEYS[2:]}
    records = payload.get("records", {})
    if set(records) != expected:
        raise RuntimeError("Control-card registry is incomplete.")

    cards: dict[str, Any] = {}
    for key, record in records.items():
        card = load_card(cards_root / record["path"])
        if card.hash != record.get("card_hash"):
            raise RuntimeError(f"{key}: control-card hash mismatch.")
        cards[key] = card
    return cards, payload


def _control_score(
    case: Any,
    *,
    key: str,
    selected_score: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if key == "PTDF":
        gross, induced, score = ptdf_net_relief_scores(case)
        return score, {
            "gross_relief_sum": float(gross.sum()),
            "induced_or_worsened_sum": float(induced.sum()),
            "net_relief_sum": float(score.sum()),
            "score_sha256": sha256_array(score),
        }
    if key == "INVERSE":
        score = inverse_order(selected_score)
        return score, {"score_sha256": sha256_array(score)}
    if key.startswith("SHUFFLE_"):
        seed = int(key.split("_", 1)[1])
        score = frozen_shuffle_scores(
            selected_score,
            case.participant_directions,
            (seed,),
        )[seed]
        return score, {
            "seed": seed,
            "score_sha256": sha256_array(score),
        }
    raise RuntimeError(f"Unsupported control {key}.")


def _update(
    manifest: dict[str, Any],
    status_path: Path,
    job_id: str,
    status: str,
    **extra: Any,
) -> None:
    jobs = manifest.setdefault("jobs", {})
    record = dict(jobs.get(job_id, {}))
    record.update({"status": status, "updated_utc": utc_now(), **extra})
    jobs[job_id] = record
    manifest["completed_cases"] = sorted(
        key for key, value in jobs.items() if value.get("status") == "PASS"
    )
    manifest["failed_cases"] = sorted(
        key for key, value in jobs.items() if value.get("status") == "FAILED"
    )
    manifest["heartbeat_utc"] = utc_now()
    atomic_write_json(status_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predeclared rho=0.03 M3 control campaign: PTDF, inverse, "
            "and 32 frozen within-direction shuffles."
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
        default=ROOT / "results_m3_controls",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    case = load_case(args.case)
    core_status = json.loads(
        (args.core_results / "M3_CORE_STATUS.json").read_text(encoding="utf-8")
    )
    implementation_hash = scientific_fingerprint()
    if core_status.get("status") != "COMPLETE":
        raise RuntimeError("Corrected M3 core campaign must complete first.")
    if core_status.get("scientific_fingerprint") != implementation_hash:
        raise RuntimeError("M3 core campaign belongs to different scientific source code.")

    selected, _, _ = load_frozen_card(case, args.cards, 0.03)
    control_cards, control_freeze = _load_controls(args.cards, selected)

    plan = {
        "schema": "TPWRS_M3_CONTROL_PLAN_V1",
        "status": "PASS",
        "created_utc": utc_now(),
        "years": EVALUATION_YEARS,
        "control_keys": CONTROL_KEYS,
        "new_heavy_case_count": len(EVALUATION_YEARS) * len(CONTROL_KEYS),
        "selected_card_hash": selected.hash,
        "control_freeze_sha256": sha256_file(
            args.cards / "RHO03_CONTROL_CARD_FREEZE.json"
        ),
        "no_scientific_optimization": bool(args.dry_run),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.results / "DRY_RUN_PLAN.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    identity = {
        "physical_case_hash": case.physical_case_hash,
        "selected_card_hash": selected.hash,
        "control_freeze_hash": sha256_file(
            args.cards / "RHO03_CONTROL_CARD_FREEZE.json"
        ),
        "years": EVALUATION_YEARS,
        "controls": CONTROL_KEYS,
        "implementation_hash": implementation_hash,
        "heavy_concurrency": 1,
    }
    campaign_hash = sha256_json(identity)
    status_path = args.results / "RUN_STATUS.json"

    with CampaignLock(args.results / ".campaign.lock", campaign_hash):
        if status_path.is_file():
            manifest = json.loads(status_path.read_text(encoding="utf-8"))
            if manifest.get("campaign_hash") != campaign_hash:
                raise RuntimeError("Control result root belongs to another campaign.")
            manifest["status"] = "RUNNING"
            manifest["resumed_utc"] = utc_now()
            manifest["invocation_count"] = int(
                manifest.get("invocation_count", 1)
            ) + 1
            print(
                "M3_CONTROLS=AUTO_RESUME "
                f"completed={len(manifest.get('completed_cases', []))}/"
                f"{len(EVALUATION_YEARS) * len(CONTROL_KEYS)}",
                flush=True,
            )
        else:
            manifest = {
                "schema": "TPWRS_M3_CONTROL_RUN_V1",
                "status": "RUNNING",
                "started_utc": utc_now(),
                "invocation_count": 1,
                "campaign_hash": campaign_hash,
                **identity,
                "jobs": {},
                "completed_cases": [],
                "failed_cases": [],
            }
            print(
                "M3_CONTROLS=NEW_RUN "
                f"total={len(EVALUATION_YEARS) * len(CONTROL_KEYS)}",
                flush=True,
            )
        atomic_write_json(status_path, manifest)

        rows: list[dict[str, Any]] = []
        try:
            for year in EVALUATION_YEARS:
                current = year_case(case, year)
                _, selected_score = q_for_year(
                    current,
                    year,
                    selected,
                    args.cards,
                    args.features,
                )
                baseline = json.loads(
                    (
                        args.core_results
                        / f"year_{year}/rho_0.03/annual/M2/metrics.json"
                    ).read_text(encoding="utf-8")
                )
                selected_metrics = json.loads(
                    (
                        args.core_results
                        / f"year_{year}/rho_0.03/annual/M3/metrics.json"
                    ).read_text(encoding="utf-8")
                )
                s2 = float(baseline["S_mw_h_row"])
                s_selected = float(selected_metrics["S_mw_h_row"])

                for key in CONTROL_KEYS:
                    job_id = f"Y{year}_RHO03_CONTROL_{key}"
                    _update(manifest, status_path, job_id, "RUNNING")
                    score, score_audit = _control_score(
                        current,
                        key=key,
                        selected_score=selected_score,
                    )
                    card = control_cards[key]
                    q = q_from_card(
                        card,
                        current.participant_directions,
                        score,
                    )
                    relative = Path(f"year_{year}/{key}")
                    experiment = make_experiment(
                        current,
                        case_id=job_id,
                        rule="M3",
                        rho=0.03,
                        mode="annual",
                        result_root_id=str(relative).replace("\\", "/"),
                        implementation_hash=implementation_hash,
                        card=card,
                        q=q,
                        information_level=f"FROZEN_CONTROL_{key}",
                    )
                    solution, action = run_annual(
                        current,
                        experiment,
                        q,
                        args.results / relative,
                    )
                    S = float(solution.s_value_mw_h_row)
                    rows.append(
                        {
                            "year": year,
                            "control": key,
                            "card_id": card.card_id,
                            "card_hash": card.hash,
                            "solution_hash": solution.solution_hash,
                            "S_mw_h_row": S,
                            "security_relief_pct_vs_M2": (
                                100.0 * (s2 - S) / s2 if s2 > 1e-12 else np.nan
                            ),
                            "delta_S_pct_vs_selected_M3": (
                                100.0 * (S - s_selected) / s_selected
                                if s_selected > 1e-12
                                else np.nan
                            ),
                            "E_restricted_mwh": float(solution.e_value_mwh),
                            "physical_fde_sum_h": float(
                                np.sum(
                                    current.time_grid.delta[:, None]
                                    * solution.x_mw
                                    / current.capacities_mw[None, :]
                                )
                            ),
                            "score_audit": json.dumps(
                                score_audit,
                                sort_keys=True,
                            ),
                        }
                    )
                    _update(
                        manifest,
                        status_path,
                        job_id,
                        "PASS",
                        action=action,
                        solution_hash=solution.solution_hash,
                    )

            frame = pd.DataFrame(rows).drop_duplicates(
                ["year", "control"], keep="last"
            )
            atomic_write_csv(args.results / "M3_CONTROL_SUMMARY.csv", frame)
            atomic_write_json(
                args.results / "M3_CONTROL_STATUS.json",
                {
                    "schema": "TPWRS_M3_CONTROL_STATUS_V1",
                    "status": "COMPLETE",
                    "created_utc": utc_now(),
                    "scientific_fingerprint": implementation_hash,
                    "expected_case_count": (
                        len(EVALUATION_YEARS) * len(CONTROL_KEYS)
                    ),
                    "completed_case_count": len(frame),
                    "selected_card_hash": selected.hash,
                    "control_freeze_sha256": sha256_file(
                        args.cards / "RHO03_CONTROL_CARD_FREEZE.json"
                    ),
                    "outputs": {
                        "summary": {
                            "path": "M3_CONTROL_SUMMARY.csv",
                            "sha256": sha256_file(
                                args.results / "M3_CONTROL_SUMMARY.csv"
                            ),
                        }
                    },
                },
            )
            manifest["status"] = "COMPLETE"
            manifest["completed_utc"] = utc_now()
            atomic_write_json(status_path, manifest)
            print("M3_CONTROLS=COMPLETE", flush=True)
            return 0
        except BaseException as exc:
            manifest["status"] = "FAILED"
            manifest["failed_utc"] = utc_now()
            manifest["failure"] = f"{type(exc).__name__}: {exc}"
            atomic_write_json(status_path, manifest)
            raise
