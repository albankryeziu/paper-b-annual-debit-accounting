from __future__ import annotations


# ============================================================================
# tpwrs_v3.campaign_runtime (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Iterable

from tpwrs.storage import ContractError, ReplayFailure
from tpwrs.contracts import ExperimentCase
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.contracts import PhysicalCase
from tpwrs.replay import replay_solution
from tpwrs.contracts import PolicySolution
from tpwrs.replay import certified_face_numeric_tolerance


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def implementation_fingerprint(root: str | Path, relative_paths: Iterable[str]) -> str:
    base = Path(root)
    names=set(relative_paths)
    for directory in ('tpwrs','jobs'):
        names.update(p.relative_to(base).as_posix() for p in (base/directory).rglob('*.py'))
    records={str(rel).replace("\\","/"):sha256_file(base/rel) for rel in sorted(names)}
    return sha256_json(records)


def validate_policy_solution(
    case: PhysicalCase,
    experiment: ExperimentCase,
    solution: PolicySolution,
    *,
    require_stage3: bool,
) -> dict[str, object]:
    if solution.stage1.status != "OPTIMAL" or solution.stage2.status != "OPTIMAL":
        raise ReplayFailure(
            "Accepted policy requires proved Stage-I and Stage-II optima."
        )

    if require_stage3 and (
        solution.stage3 is None or solution.stage3.status != "OPTIMAL"
    ):
        raise ReplayFailure(
            "Accepted final policy requires a proved Stage-III selector optimum."
        )

    if not require_stage3 and solution.stage3 is not None:
        raise ReplayFailure(
            "Stage-II-only policy unexpectedly contains a Stage-III selector."
        )

    report = replay_solution(
        case,
        experiment,
        solution,
        raise_on_failure=True,
    )

    s_cap = (
        solution.stage1.objective_value
        + experiment.stage1_tolerance(solution.stage1.objective_value)
    )
    e_cap = (
        solution.stage2.objective_value
        + experiment.stage2_tolerance(solution.stage2.objective_value)
    )
    numeric = certified_face_numeric_tolerance(solution, experiment.solver_profile)

    if report.s_value_mw_h_row > s_cap + numeric:
        excess = report.s_value_mw_h_row - (s_cap + numeric)
        raise ReplayFailure(
            "Selected policy lies outside the certified Stage-I face: "
            f"replayed_S={report.s_value_mw_h_row:.17g}, "
            f"scientific_cap={s_cap:.17g}, numeric_certificate={numeric:.17g}, "
            f"excess={excess:.3e}."
        )
    if report.e_value_mwh > e_cap + numeric:
        excess = report.e_value_mwh - (e_cap + numeric)
        raise ReplayFailure(
            "Selected policy lies outside the certified Stage-II face: "
            f"replayed_E={report.e_value_mwh:.17g}, "
            f"scientific_cap={e_cap:.17g}, numeric_certificate={numeric:.17g}, "
            f"excess={excess:.3e}."
        )
    if not solution.replay_summary.get("passed"):
        raise ReplayFailure("Stored policy replay marker is not PASS.")

    return {
        "passed": True,
        "stage1_status": solution.stage1.status,
        "stage2_status": solution.stage2.status,
        "stage3_status": (
            None if solution.stage3 is None else solution.stage3.status
        ),
        "replayed_S": report.s_value_mw_h_row,
        "replayed_E": report.e_value_mwh,
        "maximum_budget_violation_h": report.max_budget_violation_h,
        "maximum_balance_residual_mw": report.max_balance_residual_mw,
        "maximum_x_bound_violation_mw": report.max_x_bound_violation_mw,
    }


def _pid_is_active(pid: int) -> bool:
    """No signal is ever sent. Unknown/access-denied is treated as active."""
    if not isinstance(pid,int) or pid <= 0:
        return True
    import psutil
    try:
        process=psutil.Process(pid)
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return True


@dataclass
class CampaignLock:
    path: Path
    campaign_hash: str
    acquired: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        payload = {
            "schema": "TPWRS_CAMPAIGN_LOCK_V2",
            "campaign_hash": self.campaign_hash,
            "pid": os.getpid(),
            "host": host,
            "created_utc": utc_now(),
        }

        try:
            fd = os.open(
                str(self.path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            try:
                current = json.loads(
                    self.path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise ContractError(
                    f"Campaign lock exists but is unreadable: {self.path}"
                ) from exc

            lock_host = str(current.get("host", ""))
            try:
                lock_pid = int(current.get("pid", -1))
            except Exception:
                lock_pid = -1

            if lock_host != host:
                raise ContractError(
                    "Campaign lock belongs to another host and cannot be "
                    f"declared stale locally (host={lock_host!r}, "
                    f"pid={lock_pid}, lock={self.path})."
                )

            if _pid_is_active(lock_pid):
                raise ContractError(
                    "Duplicate campaign process is active "
                    f"(pid={lock_pid}, lock={self.path})."
                )

            self.path.unlink()
            try:
                fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError as exc:
                raise ContractError(
                    "Campaign lock changed while stale-lock recovery was "
                    f"in progress: {self.path}"
                ) from exc

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return

        try:
            if not self.path.is_file():
                return

            current = json.loads(
                self.path.read_text(encoding="utf-8")
            )
            if (
                int(current.get("pid", -1)) == os.getpid()
                and current.get("host") == socket.gethostname()
                and current.get("campaign_hash") == self.campaign_hash
            ):
                self.path.unlink()
        finally:
            self.acquired = False

    def __enter__(self) -> "CampaignLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def active_campaign_lock(path: str | Path) -> tuple[bool, str]:
    lock = Path(path)

    if not lock.is_file():
        return False, "absent"

    try:
        payload = json.loads(
            lock.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return True, f"unreadable: {type(exc).__name__}: {exc}"

    lock_host = str(payload.get("host", ""))
    current_host = socket.gethostname()

    if lock_host != current_host:
        return True, "lock belongs to another host"

    try:
        pid = int(payload.get("pid", -1))
    except Exception:
        return True, "local lock has invalid pid"

    active = _pid_is_active(pid)
    return active, ("active" if active else "stale")


# ============================================================================
# tpwrs_v3.runtime_support (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Execution diagnostics and stopping budgets; never part of a scientific objective."""

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Callable

from tpwrs.storage import SolverFailure
from tpwrs.storage import atomic_write_json


def memory_snapshot() -> dict[str, float]:
    try:
        import psutil
        p = psutil.Process()
        m = p.memory_info()
        v = psutil.virtual_memory()
        out = {"working_gb": m.rss / 1024**3,
               "available_ram_gb": v.available / 1024**3,
               "total_ram_gb": v.total / 1024**3,
               "cpu_seconds": sum(p.cpu_times()[:2])}
        # Windows private commit is not interchangeable with Linux VMS.
        if hasattr(m, "private"): out["private_gb"] = m.private / 1024**3
        if hasattr(m, "peak_wset"): out["peak_working_gb"] = m.peak_wset / 1024**3
        return out
    except (ImportError, OSError):
        return {}


def budget_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionBudget:
    """Controlled stop, not approximate acceptance. SoftMemLimit uses decimal GB.

    Resource limits are recorded separately from the physical/allowance hashes.
    An incomplete stage cannot advance or become a final solution.
    """
    def __init__(self, root: Path | None = None, callback: Callable | None = None):
        self.root = None if root is None else Path(root)
        self.started = time.monotonic()
        self.reason = None
        self.max_case_s = float(os.environ.get("TPWRS_MAX_CASE_SECONDS", "14400"))
        self.max_stage_s = float(os.environ.get("TPWRS_MAX_STAGE_SECONDS", "3600"))
        if min(self.max_case_s,self.max_stage_s)<=0:
            raise ValueError("Execution budgets must be positive; unlimited blind runs are not a default.")
        self._end = threading.Event()
        self._callback = callback
        self._stage = "BUILD"
        self._latest = {}
        self._peak = 0.0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()
        else: self._thread = None

    def _monitor(self):
        while not self._end.wait(30):
            m = memory_snapshot()
            self._peak = max(self._peak, m.get("private_gb",m.get("working_gb",0.0)))
            record = {"status":"RUNNING", "created_utc":budget_utc_now(), "pid":os.getpid(),
                      "stage":self._stage,"elapsed_s":time.monotonic()-self.started,
                      "peak_process_memory_gb":self._peak, **m}
            try: atomic_write_json(self.root / "PROCESS_HEARTBEAT.json", record)
            except OSError: pass  # output error is diagnosed by the main writer

    def set_stage(self, stage: str):
        self._stage=stage

    def configure(self, model, *, model_size=None):
        import psutil
        import numpy as np
        # A bounded micro-model must not inherit the annual OS/Python reserve.
        # Callers declare dimensions BEFORE allocation. Missing dimensions are
        # conservative production scope, never an implicit exemption.
        # The 168-hour A/B qualification matrices are deliberately bounded
        # reference problems.  They must remain independent of the annual
        # production headroom guard as well as the very small unit cases.
        # Full-year matrices exceed at least the coordinate cap below.
        small = (model_size is not None and len(model_size) == 3
                 and all(int(v) >= 0 for v in model_size)
                 and all(int(v) <= cap for v, cap in
                         zip(model_size, (100000, 250000, 2000000))))
        self.memory_scope = "BOUNDED_SMALL_MODEL" if small else "PRODUCTION"
        if small:
            model.Params.TimeLimit = min(float(model.Params.TimeLimit), self.max_stage_s)
            # Do not apply an ambient production cap to qualification microcases.
            return
        virtual = psutil.virtual_memory()
        available = int(virtual.available)
        total = int(getattr(virtual, "total", 16*1024**3))
        # Leave a proportional OS reserve, with a smaller floor on hosts that
        # have at most 8 GiB installed.  A fixed 512 MiB floor consumed most of
        # the usable solver headroom on such hosts despite a healthy page file.
        reserve_floor = 256*1024**2 if total <= 8*1024**3 else 512*1024**2
        reserve = min(2*1024**3, max(reserve_floor,0.25*available))
        remaining = available-reserve
        if remaining < 128*1024**2:
            raise SolverFailure("INSUFFICIENT_MEMORY_HEADROOM: less than 128 MiB remains after "
                                "the Python/OS reserve; no optimization was started.")
        # Gurobi's cap counts existing allocations in its environment; available
        # system RAM counts only additional headroom. Do not confuse the two.
        try:
            allocated=float(model.MemUsed)
        except (AttributeError, RuntimeError):
            allocated=0.0
        derived=allocated+0.8*remaining/1e9
        limit=float(os.environ.get("TPWRS_GUROBI_SOFTMEM_GB",str(derived)))
        if not np.isfinite(limit) or limit<=0:
            raise ValueError("Soft memory limit must be positive and finite.")
        if limit>allocated+remaining/1e9:
            raise SolverFailure("MEMORY_OVERRIDE_EXCEEDS_HEADROOM: fixed Gurobi cap exceeds "
                                "the measured additional RAM budget. No optimization started.")
        model.Params.SoftMemLimit=limit
        configured=float(model.Params.TimeLimit)
        model.Params.TimeLimit=min(configured,self.max_stage_s)
        if self.root is not None:
            atomic_write_json(self.root / "EXECUTION_BUDGET.json",{
                "created_utc":budget_utc_now(),"pid":os.getpid(),"max_case_seconds":self.max_case_s,
                "max_stage_seconds":self.max_stage_s,"gurobi_softmem_decimal_gb":limit,
                "scope":"EXECUTION_GUARD_NOT_OPTIMALITY_ACCEPTANCE",**memory_snapshot()})

    def check_callback(self, model):
        if time.monotonic()-self.started>self.max_case_s:
            self.reason="CASE_TIME_BUDGET"
        stop=os.environ.get("TPWRS_STOP_FILE")
        if stop and Path(stop).is_file(): self.reason="USER_STOP_REQUEST"
        if self.root is not None and (self.root / "STOP_REQUESTED").is_file():
            self.reason="USER_STOP_REQUEST"
        if self.reason: model.terminate()

    def check_status(self, model):
        from gurobipy import GRB
        if model.Status == GRB.OPTIMAL: return
        names={GRB.TIME_LIMIT:"STAGE_TIME_BUDGET",GRB.MEM_LIMIT:"SOFT_MEMORY_LIMIT",
               GRB.INTERRUPTED:self.reason or "INTERRUPTED"}
        reason=names.get(model.Status,f"GUROBI_STATUS_{model.Status}")
        if self.root is not None:
            atomic_write_json(self.root / "INCOMPLETE_STAGE.json",{
                "status":"INCOMPLETE","reason":reason,"created_utc":budget_utc_now(),
                "stage":self._stage,"elapsed_s":time.monotonic()-self.started,
                "accepted_as_scientific_result":False,**memory_snapshot()})
        raise SolverFailure(f"{reason}: stage is incomplete, not a certified result.")

    def close(self):
        self._end.set()
        if self._thread is not None: self._thread.join(timeout=2)


# ============================================================================
# tpwrs_v3.stage_workspace (consolidated; see SOURCE_MAP.json)
# ============================================================================

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

from tpwrs.storage import sha256_file, sha256_json
from tpwrs.storage import atomic_write_json


# Only derived output roots belong here. Frozen source/input roots are never
# archived by this mechanism.
STAGE_OUTPUT_ROOTS: dict[str, tuple[str, ...]] = {
    "03": ("artifacts/cards", "artifacts/features"),
    "04": ("results_m1_m2_final",),
    "05": ("results_m3_final",),
    "05B": ("results_m3_controls",),
    "05C": ("results_m3_gamma", "results_m3_rate_bounds"),
    "Q": ("results_engineering_validation/selected_registry", "artifacts/ac_inputs"),
    "06": ("results_engineering_validation/independent_dc", "results_engineering_validation/selected_ac"),
    "07": ("results_final_evidence",),
}

STAGE_IDENTITY_INPUTS: dict[str, tuple[str, ...]] = {
    "Q": ("artifacts/cards/CARD_BUILD_STATUS.json", "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json", "results_m3_final/M3_CORE_STATUS.json", "results_m3_gamma/M3_GAMMA_STATUS.json", "results_m3_rate_bounds/RATE_BOUND_STATUS.json", "local/AC_INPUTS.json"),
    "03": (
        "artifacts/FINAL_PHYSICAL_READINESS.json",
        "artifacts/real_case/SOURCE_REGISTRY.json",
        "artifacts/performance/SELECTED_SOLVER_PROFILE.json",
        "results_2028_performance_regression/STATUS.json",
    ),
    "04": (
        "artifacts/FINAL_PHYSICAL_READINESS.json",
        "artifacts/performance/SELECTED_SOLVER_PROFILE.json",
        "results_2028_performance_regression/STATUS.json",
        "artifacts/cards/CARD_BUILD_STATUS.json",
    ),
    "05": (
        "artifacts/performance/SELECTED_SOLVER_PROFILE.json",
        "results_2028_performance_regression/STATUS.json",
        "artifacts/FINAL_PHYSICAL_READINESS.json",
        "artifacts/cards/CARD_BUILD_STATUS.json",
        "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json",
    ),
    "05B": (
        "artifacts/performance/SELECTED_SOLVER_PROFILE.json",
        "results_2028_performance_regression/STATUS.json",
        "artifacts/FINAL_PHYSICAL_READINESS.json",
        "artifacts/cards/CARD_BUILD_STATUS.json",
        "artifacts/cards/RHO03_CONTROL_CARD_FREEZE.json",
        "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json",
        "results_m3_final/M3_CORE_STATUS.json",
    ),
    "06": (
        "artifacts/FINAL_PHYSICAL_READINESS.json",
        "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json",
        "results_m3_final/M3_CORE_STATUS.json",
        "artifacts/ac_inputs/AC_INPUT_FREEZE.json",
        "results_engineering_validation/selected_registry/SELECTED_STATE_REGISTRY_FREEZE.json",
    ),
    "05C": (
        "artifacts/performance/SELECTED_SOLVER_PROFILE.json",
        "results_2028_performance_regression/STATUS.json",
        "artifacts/cards/CARD_BUILD_STATUS.json",
        "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json",
        "results_m3_final/M3_CORE_STATUS.json",
    ),
    "07": (
        "results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json",
        "results_m3_final/M3_CORE_STATUS.json",
        "results_m3_controls/M3_CONTROL_STATUS.json",
        "results_m3_gamma/M3_GAMMA_STATUS.json",
        "results_m3_rate_bounds/RATE_BOUND_STATUS.json",
        "results_engineering_validation/independent_dc/STATUS.json",
        "results_engineering_validation/selected_ac/STATUS.json",
        "artifacts/prices/PRICE_FREEZE.json",
    ),
}

MARKER = ".TPWRS_STAGE_WORKSPACE.json"


def _input_hashes(root: Path, relatives: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relatives:
        path = root / relative
        if path.is_file():
            if relative == "artifacts/FINAL_PHYSICAL_READINESS.json":
                payload=json.loads(path.read_text(encoding="utf-8"))
                stable={k:v for k,v in payload.items() if k not in {"created_utc","free_disk_gib"}}
                result[relative]=sha256_json(stable)
            else:
                result[relative] = sha256_file(path)
        else:
            result[relative] = "MISSING"
    return result


def stage_workspace_identity(root: str | Path, stage: str, scientific_fingerprint: str) -> str:
    base = Path(root)
    return sha256_json({
        "schema": "TPWRS_STAGE_WORKSPACE_IDENTITY_V1",
        "stage": stage,
        "scientific_fingerprint": scientific_fingerprint,
        "inputs": _input_hashes(base, STAGE_IDENTITY_INPUTS.get(stage, ())),
    })


def _nonempty(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _read_marker(path: Path) -> dict[str, object] | None:
    marker = path / MARKER
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _history_destination(root: Path, stage: str, output: Path, old_identity: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = output.relative_to(root).as_posix().replace("/", "__")
    base = root / "artifacts" / "stage_history" / f"RUN_{stage}"
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{stamp}_{old_identity[:12]}_{safe_name}"
    candidate = base / stem
    counter = 1
    while candidate.exists():
        candidate = base / f"{stem}_{counter:02d}"
        counter += 1
    return candidate


def prepare_stage_outputs(root: str | Path, stage: str, scientific_fingerprint: str) -> dict[str, object]:
    """Prepare derived stage roots transactionally.

    Same stage identity resumes in place. A different or legacy identity is
    preserved by an atomic same-volume directory rename into stage_history and
    a clean workspace is created. Numerical solutions are never rebound or
    migrated between identities.
    """
    base = Path(root)
    identity = stage_workspace_identity(base, stage, scientific_fingerprint)
    actions: list[dict[str, object]] = []

    for relative in STAGE_OUTPUT_ROOTS.get(stage, ()):
        output = base / relative
        marker = _read_marker(output) if output.is_dir() else None
        current_identity = None if marker is None else marker.get("workspace_identity")

        if output.is_dir() and current_identity == identity:
            actions.append({"root": relative, "action": "RESUME", "workspace_identity": identity})
            continue

        if output.exists() and (not output.is_dir() or _nonempty(output)):
            old_identity = str(current_identity or "LEGACY_UNVERSIONED")
            destination = _history_destination(base, stage, output, old_identity)
            output.rename(destination)
            actions.append({
                "root": relative,
                "action": "ARCHIVE_AND_RESET",
                "old_workspace_identity": old_identity,
                "archive": destination.relative_to(base).as_posix(),
                "workspace_identity": identity,
            })
        elif output.exists() and output.is_dir():
            # Empty stale directory: no scientific artifact exists to preserve.
            try:
                output.rmdir()
            except OSError:
                pass

        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / MARKER, {
            "schema": "TPWRS_STAGE_WORKSPACE_V1",
            "stage": stage,
            "workspace_identity": identity,
            "scientific_fingerprint": scientific_fingerprint,
            "identity_inputs": _input_hashes(base, STAGE_IDENTITY_INPUTS.get(stage, ())),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        if not actions or actions[-1].get("root") != relative:
            actions.append({"root": relative, "action": "CREATE", "workspace_identity": identity})

    return {
        "schema": "TPWRS_STAGE_WORKSPACE_PREP_V1",
        "stage": stage,
        "workspace_identity": identity,
        "actions": actions,
    }


# ============================================================================
# tpwrs_v3.readiness (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from typing import Mapping

from tpwrs.storage import atomic_write_json


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str
    mandatory: bool = True


def environment_gate(
    *,
    require_gurobi: bool = True,
    require_pandapower: bool = True,
) -> list[Gate]:
    gates: list[Gate] = []
    for name, required in (
        ("gurobipy", require_gurobi),
        ("pandapower", require_pandapower),
    ):
        found = importlib.util.find_spec(name) is not None
        gates.append(
            Gate(
                f"dependency::{name}",
                found if required else True,
                "available"
                if found
                else "NOT EXECUTED IN CHAT ENVIRONMENT / missing locally",
                required,
            )
        )
    return gates


def build_readiness(
    output_path: str | Path,
    *,
    tests_passed: bool,
    compile_passed: bool,
    parent_freeze_passed: bool,
    physical_case_frozen: bool,
    cards_frozen: bool,
    heldout_leakage_passed: bool,
    smoke_tests_passed: bool,
    source_paths_resolved: bool,
    output_root_writable: bool,
    disk_space_ok: bool,
    resume_verified: bool,
    dry_run_passed: bool,
    no_active_campaign_lock: bool,
    lightweight_environment_consistent: bool,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    gates = [
        Gate("compile", compile_passed, "python compile gate"),
        Gate("tests", tests_passed, "lightweight/unit regression gate"),
        Gate(
            "lightweight_environment",
            lightweight_environment_consistent,
            "same tested interpreter; supported Python; Gurobi license and pandapower gate",
        ),
        Gate("parent_freeze", parent_freeze_passed, "frozen parent integrity"),
        Gate("source_paths", source_paths_resolved, "real Dutch source/data paths"),
        Gate("physical_case", physical_case_frozen, "immutable physical case"),
        Gate("cards", cards_frozen, "allowance-specific M3 cards"),
        Gate("heldout_isolation", heldout_leakage_passed, "no selection leakage"),
        Gate("smoke_tests", smoke_tests_passed, "real small smoke tests"),
        Gate("output_writable", output_root_writable, "result root"),
        Gate("disk_space", disk_space_ok, "free space"),
        Gate("resume", resume_verified, "restart-safe artifacts"),
        Gate("dry_run", dry_run_passed, "complete no-solve campaign enumeration"),
        Gate("duplicate_lock", no_active_campaign_lock, "no active duplicate campaign process"),
    ]
    gates.extend(environment_gate())

    ready = all(g.passed for g in gates if g.mandatory)
    payload: dict[str, object] = {
        "schema": "TPWRS_V3_OVERNIGHT_READINESS_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if ready else "NOT_READY",
        "gates": [asdict(g) for g in gates],
        "extra": dict(extra or {}),
    }
    atomic_write_json(output_path, payload)
    return payload


# ============================================================================
# tpwrs_v3.annual_qualification (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tpwrs.contracts import SolverProfile
from tpwrs.storage import sha256_file, sha256_json

# Scientific identity of the annual continuous solver and its independent replay.
# Deliberately excludes downstream M3 feature/card scripts, reporting code, AC code,
# and execution-only wrappers.  A change here requires a fresh RUN_02; a change to
# an unrelated downstream module does not.
ANNUAL_SOLVER_SOURCE_PATHS: tuple[str, ...] = (
    "tpwrs/annual.py", "tpwrs/model.py", "tpwrs/linear.py",
    "tpwrs/replay.py", "tpwrs/contracts.py", "tpwrs/execution.py",
    "tpwrs/persistence.py", "tpwrs/storage.py", "tpwrs/reporting.py",
    "tpwrs/compression.py",
)

QUALIFICATION_PROFILE_NAME = "validation_support_oa_dual_simplex_v3"
QUALIFICATION_CONTRACT = {
    "schema": "TPWRS_ANNUAL_QUALIFICATION_CONTRACT_V1",
    "solver_profile_name": QUALIFICATION_PROFILE_NAME,
    "backend": "gurobi",
    "presolve": False,
    "threads": 1,
    "lp_method": "dual_simplex",
    "output_flag": True,
    "feasibility_tol": 1e-8,
    "mip_gap": 1e-8,
    "week_seconds": 600,
    "full_year_seconds": 3600,
    "rules": ["M1_LP", "M2"],
    "qualification_year": 2028,
    "qualification_window_hours": 168,
    "full_year_hours": 8784,
}


def annual_solver_fingerprint(root: str | Path) -> str:
    base = Path(root)
    records = {
        rel: sha256_file(base / rel)
        for rel in ANNUAL_SOLVER_SOURCE_PATHS
    }
    return sha256_json({
        "contract": QUALIFICATION_CONTRACT,
        "sources": records,
    })


def validation_solver_profile(seconds: int = 3600) -> SolverProfile:
    return SolverProfile(
        name=QUALIFICATION_PROFILE_NAME,
        backend="gurobi",
        time_limit_s=seconds,
        presolve=False,
        threads=1,
        lp_method="dual_simplex",
        output_flag=True,
        feasibility_tol=1e-8,
        mip_gap=1e-8,
    )


def qualification_contract_record(root: str | Path) -> dict[str, object]:
    return {
        **QUALIFICATION_CONTRACT,
        "annual_solver_fingerprint": annual_solver_fingerprint(root),
        "profile": asdict(validation_solver_profile()),
    }
