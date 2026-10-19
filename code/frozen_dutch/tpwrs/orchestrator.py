"""One entry point, explicit stages, fail-closed prerequisites and UTF-8 logs."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .config import ROOT, settings, scientific_fingerprint, verify_release_manifest
from .execution import CampaignLock, utc_now, prepare_stage_outputs
from .storage import atomic_write_json
from .provenance import require_compatible_status

# key -> ordinary Python module and function; there are no generated patch scripts.
JOBS = {
    "check": ("jobs.gate", "main"),
    "inputs": ("jobs.input_audit", "main"),
    "features": ("jobs.cards", "features_main"),
    "cards": ("jobs.cards", "main"),
    "qualify": ("jobs.qualification", "main"),
    "smoke": ("jobs.qualification", "smoke_main"),
    "reproduce": ("jobs.qualification", "reproduce_main"),
    "primary": ("jobs.primary", "main"),
    "primary_freeze": ("jobs.primary_freeze", "main"),
    "m3": ("jobs.m3", "main"),
    "controls": ("jobs.controls", "main"),
    "selection": ("jobs.selection", "main"),
    "dc": ("jobs.dc", "main"),
    "ac": ("jobs.ac", "main"),
    "ac_freeze": ("jobs.ac", "freeze_main"),
    "evidence": ("jobs.report", "main"),
    "prices": ("jobs.input_audit", "price_main"),
    "gamma": ("jobs.gamma", "main"),
    "bounds": ("jobs.rate_bounds", "main"),
    "rebuild_inputs": ("jobs.input_build", "main"),
}
# 07 exports evidence only. No experimental prose/manuscript is written here.
STAGES = {
    "00": ("Compile, tests, imports, native micro-probes", [("check", [])], False),
    "01": ("Import/audit immutable physical inputs", [("inputs", [])], False),
    "REPRO": ("Read-only R9 ledger reconstruction; solve ONLY failing 48-hour epoch", [("reproduce", [])], False),
    "SMOKE": ("Small real-case numerical smoke test", [("smoke", [])], False),
    "02": ("2028 qualification: 168-hour A/B then annual", [("qualify", [])], True),
    "03": ("Fit and freeze F0 cards", [("features", []), ("cards", [])], True),
    "04": ("Primary M1/M2 and evidence freeze", [("features", ["--deployment"]), ("primary", []), ("primary_freeze", [])], True),
    "05": ("M3/M2-MATCHED", [("m3", [])], True),
    "05B": ("Predeclared M3 controls", [("controls", [])], True),
    "05C": ("Post-freeze M3 gamma and rate-bound sensitivities", [("gamma", []), ("bounds", [])], True),
    "Q": ("Freeze selected engineering states and AC inputs", [("selection", []), ("ac_freeze", [])], False),
    "06": ("Selected states: independent DC and AC", [("dc", []), ("ac", [])], True),
    "07": ("Export result evidence (no manuscript prose)", [("evidence", [])], False),
    "PRICE": ("Freeze optional ex-post price", [("prices", [])], False),
    "DATA": ("Explicit source reconstruction", [("rebuild_inputs", ["--phase", "all", "--output", str(ROOT/"artifacts/reconstructed_real_case")])], True),
}

def require_status(path,allowed=('PASS',)):
    if not path.is_file(): raise RuntimeError(f"Prerequisite missing: {path}")
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('status') not in allowed: raise RuntimeError(f"Prerequisite did not pass: {path}: {obj.get('status')}")
    return obj

def prerequisites(stage):
    if stage in {'00','PRICE'}: return
    cheap=require_status(ROOT/'artifacts/LIGHTWEIGHT_TEST_STATUS.json')
    integrity=verify_release_manifest()
    current_fp=scientific_fingerprint()
    if cheap.get('scientific_fingerprint')!=current_fp:
        raise RuntimeError('Code changed after tests. Rerun RUN_00 before any optimization.')
    if cheap.get('release_id')!=integrity.get('release_id') or cheap.get('release_manifest_sha256')!=integrity.get('manifest_sha256'):
        raise RuntimeError('RUN_00 belongs to a different extracted release. Rerun RUN_00 before any optimization.')
    if not cheap.get('native_environment_ready'):
        raise RuntimeError('Native Gurobi/pandapower/Parquet tests must pass locally. No expensive job started.')
    if stage=='01': return
    if stage=='DATA':
        if not settings().get('permit_heavy_source_rebuild'):
            raise RuntimeError('Source rebuild deliberately disabled by default. Reuse the frozen case with RUN_01. For an intentional rebuild set permit_heavy_source_rebuild=true in LOCAL_SETTINGS.json.')
        return
    physical,_=require_compatible_status(
        ROOT/'artifacts/FINAL_PHYSICAL_READINESS.json',
        use='IMMUTABLE_PHYSICAL_CASE_INPUT',
    )
    if stage not in {'02','SMOKE'}:
        from .config import qualified_solver_profile
        qualified_solver_profile()
    if stage in {'04','05','05B','05C','Q','06','07','REPRO'}:
        cards,_=require_compatible_status(
            ROOT/'artifacts/cards/CARD_BUILD_STATUS.json',
            use='FROZEN_CARD_LOADING',
        )
        if cards.get('schema')!='TPWRS_ALLOWANCE_SPECIFIC_CARD_BUILD_V3':
            raise RuntimeError('Frozen card-build status schema is not current. Rerun RUN_03.')
    if stage in {'05','05B','05C','Q','06','07'}:
        baseline,_=require_compatible_status(
            ROOT/'results_m1_m2_final/M1_M2_EVIDENCE_FREEZE.json',
            use='UPSTREAM_STAGE04_NUMERICAL_EVIDENCE',
            fingerprint_fields=('implementation_hash_recorded_by_run',),
        )
    if stage in {'05B','05C','Q','06','07'}:
        core,_=require_compatible_status(
            ROOT/'results_m3_final/M3_CORE_STATUS.json',
            allowed=('COMPLETE',),
            use='UPSTREAM_STAGE05_CORE_NUMERICAL_EVIDENCE',
        )
    if stage=='05B':
        control_freeze,_=require_compatible_status(
            ROOT/'artifacts/cards/RHO03_CONTROL_CARD_FREEZE.json',
            use='UPSTREAM_STAGE05B_CONTROL_CARD_FREEZE',
        )
        if control_freeze.get('schema')!='TPWRS_RHO03_CONTROL_CARD_FREEZE_V2':
            raise RuntimeError('Control-card freeze is stale or from a different source identity. Rerun RUN_03.')
    if stage=='04':
        reproduction,_=require_compatible_status(
            ROOT/'results_numerical_reproduction/STATUS.json',
            use='UPSTREAM_REPRODUCTION_EVIDENCE',
        )
    if stage in {'05C','Q','06','07'}:
        controls,_=require_compatible_status(
            ROOT/'results_m3_controls/M3_CONTROL_STATUS.json',
            allowed=('COMPLETE','PASS'),
            use='UPSTREAM_STAGE05B_CONTROL_EVIDENCE',
        )
    if stage in {'Q','06','07'}:
        for relative,use in (
            ('results_m3_sensitivity_r10/gamma/M3_GAMMA_STATUS.json','UPSTREAM_STAGE05C_GAMMA_EVIDENCE'),
            ('results_m3_sensitivity_r10/rate_bounds/RATE_BOUND_STATUS.json','UPSTREAM_STAGE05C_RATE_BOUND_EVIDENCE'),
        ):
            require_compatible_status(ROOT/relative,allowed=('COMPLETE',),use=use)
    if stage=='Q':
        from jobs.ac import require_ac_input_preflight
        require_ac_input_preflight(ROOT/'local/AC_INPUTS.json')
    if stage=='06':
        require_status(ROOT/'artifacts/ac_inputs/AC_INPUT_FREEZE.json')
        selected=require_status(ROOT/'results_engineering_validation/selected_registry/SELECTED_STATE_REGISTRY_FREEZE.json')
        if selected.get('scientific_fingerprint')!=current_fp:
            raise RuntimeError('Run Q to freeze the current selection registry.')
    if stage=='07':
        require_compatible_status(
            ROOT/'results_m3_sensitivity_r10/gamma/M3_GAMMA_STATUS.json',
            allowed=('COMPLETE',),use='UPSTREAM_STAGE05C_GAMMA_EVIDENCE',
        )
        require_compatible_status(
            ROOT/'results_m3_sensitivity_r10/rate_bounds/RATE_BOUND_STATUS.json',
            allowed=('COMPLETE',),use='UPSTREAM_STAGE05C_RATE_BOUND_EVIDENCE',
        )
        controls,_=require_compatible_status(
            ROOT/'results_m3_controls/M3_CONTROL_STATUS.json',
            allowed=('COMPLETE','PASS'),
            use='UPSTREAM_STAGE05B_CONTROL_EVIDENCE',
        )
        require_compatible_status(
            ROOT/'results_engineering_validation/independent_dc/STATUS.json',
            use='UPSTREAM_STAGE06_DC_EVIDENCE',
        )
        require_compatible_status(
            ROOT/'results_engineering_validation/selected_ac/STATUS.json',
            use='UPSTREAM_STAGE06_AC_EVIDENCE',
        )

def _assert_release_stable(expected_fp: str, expected_release_id: object, expected_manifest_sha256: object) -> dict[str, object]:
    """Fail if the source tree changes during a numbered stage.

    This is checked before and after every child command.  A source edit can
    therefore never authorize the next command or a PASS marker in the same
    stage.  The next run will receive a different workspace identity.
    """
    integrity=verify_release_manifest()
    current=scientific_fingerprint()
    if current!=expected_fp:
        raise RuntimeError('Scientific source changed while the stage was running. The stage is invalid and must be rerun from the new identity.')
    if integrity.get('release_id')!=expected_release_id or integrity.get('manifest_sha256')!=expected_manifest_sha256:
        raise RuntimeError('Release manifest changed while the stage was running. The stage is invalid.')
    return integrity


def safe_console_write(text, stream=None):
    """Display errors must never kill a subprocess pipe reader."""
    output = sys.stdout if stream is None else stream
    try:
        output.write(text)
        output.flush()
    except UnicodeEncodeError:
        encoding = getattr(output, "encoding", None) or "ascii"
        output.write(text.encode(encoding, "backslashreplace").decode(encoding))
        output.flush()


def relay_output(pipe, log, errors, stream=None):
    """Persist UTF-8 output before display; keep draining on a display failure."""
    display_failed = False
    for line in iter(pipe.readline, ""):
        try:
            log.write(line)
            log.flush()
        except Exception as exc:
            errors.append("Log write failed: " + repr(exc))
        if not display_failed:
            try:
                safe_console_write(line, stream)
            except Exception as exc:
                # The file still receives complete output. A missing console is
                # not a reason to deadlock the numerical child.
                display_failed = True
                try:
                    log.write("\nCONSOLE_UNAVAILABLE: " + repr(exc) + "\n")
                    log.flush()
                except Exception:
                    pass


def terminate_owned_child(proc):
    """Terminate only this live child and its descendants, never an old PID."""
    if proc is None or proc.poll() is not None:
        return
    descendants = []
    try:
        import psutil
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        pass
    for child in reversed(descendants):
        try:
            child.terminate()
        except Exception:
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    for child in reversed(descendants):
        try:
            if child.is_running():
                child.kill()
        except Exception:
            pass


def run_child(job, params, env, log, stop, max_seconds, heartbeat=None):
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT/"RUN.py"), "--_job", job, *params],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="backslashreplace", bufsize=1)
    errors = []
    reader = threading.Thread(target=relay_output, args=(proc.stdout,log,errors), daemon=True)
    reader.start()
    began = time.monotonic()
    signalled = None
    try:
        while proc.poll() is None:
            elapsed = time.monotonic()-began
            if errors or elapsed > max_seconds or stop.exists():
                stop.touch(exist_ok=True)
                if signalled is None:
                    signalled = time.monotonic()
                if errors or time.monotonic()-signalled > 30:
                    terminate_owned_child(proc)
            if heartbeat is not None:
                heartbeat(proc.pid, elapsed)
            time.sleep(1)
        reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError("Output relay did not finish; no stage PASS is permitted.")
        if errors:
            raise RuntimeError(errors[0])
        if signalled is not None:
            raise RuntimeError("Run stopped by stop marker, elapsed budget, or output failure.")
        return int(proc.returncode)
    except BaseException:
        stop.touch(exist_ok=True)
        terminate_owned_child(proc)
        reader.join(timeout=10)
        raise
    finally:
        if not reader.is_alive():
            proc.stdout.close()


def build_child_environment(cfg, stop, runid):
    """Build a deterministic child environment from LOCAL_SETTINGS.

    LOCAL_SETTINGS is authoritative for release-scoped execution controls.
    In particular, an old shell/PyCharm Gurobi memory override must not leak
    into a new release when gurobi_softmem_gb is null.
    """
    env=os.environ.copy()
    # Erase stale release-scoped controls, including paths and stop/job state.
    for key in tuple(env):
        if key.startswith("TPWRS_"):
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1",PYTHONUTF8="1",PYTHONIOENCODING="utf-8:backslashreplace",
        TPWRS_STOP_FILE=str(stop),TPWRS_RUN_ID=runid,
        TPWRS_MAX_STAGE_SECONDS=str(cfg["max_stage_seconds"]),
        TPWRS_MAX_CASE_SECONDS=str(cfg["max_case_seconds"]))
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        env[key]="1"
    if cfg.get("gurobi_softmem_gb") is not None:
        env["TPWRS_GUROBI_SOFTMEM_GB"]=str(cfg["gurobi_softmem_gb"])
    else:
        env.pop("TPWRS_GUROBI_SOFTMEM_GB", None)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description="One TPWRS runner. Stage is read from LOCAL_SETTINGS.json; default 00 is cheap.")
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--portable", action="store_true", help="Only for stage 00; never authorizes production.")
    parser.add_argument("--_job", choices=sorted(JOBS), help=argparse.SUPPRESS)
    args, job_arguments = parser.parse_known_args(argv)
    if args._job:
        verify_release_manifest()
        module_name, function_name = JOBS[args._job]
        sys.argv = [str(ROOT/"RUN.py")] + job_arguments + (["--portable"] if args.portable else [])
        function = getattr(importlib.import_module(module_name),function_name)
        return int(function() or 0)
    if job_arguments:
        parser.error("Unrecognized arguments: " + " ".join(job_arguments))
    stage = args.stage or str(settings().get("stage", "00")).upper()
    if stage.isdigit(): stage=stage.zfill(2)
    if stage not in STAGES:
        parser.error("LOCAL_SETTINGS.json stage is not a supported stage.")
    if args.portable and stage != "00":
        parser.error("--portable only applies to stage 00.")
    if args.describe:
        print(json.dumps({"stage":stage,"description":STAGES[stage][0],"jobs":STAGES[stage][1],"heavy":STAGES[stage][2]},indent=2))
        return 0
    statuspath = None
    status = {"stage":stage,"status":"FAIL"}
    try:
        integrity = verify_release_manifest()
        folder = ROOT/"artifacts/run_control"
        folder.mkdir(parents=True,exist_ok=True)
        stop = folder/"STOP_REQUESTED"
        if stop.exists():
            raise RuntimeError("Stop marker remains: " + str(stop) + ". No process was started. Remove only after confirming no other campaign is active.")
        runid = time.strftime("%Y%m%dT%H%M%S") + "_" + str(os.getpid())
        statuspath = folder/("RUN_"+stage+"_STATUS.json")
        status.update(status="RUNNING",run_id=runid,started_utc=utc_now(),
            scientific_fingerprint=scientific_fingerprint(),release_id=integrity["release_id"],
            release_manifest_sha256=integrity["manifest_sha256"],commands_completed=[])
        prerequisites(stage)
        _assert_release_stable(status["scientific_fingerprint"],status["release_id"],status["release_manifest_sha256"])
        with CampaignLock(folder/".campaign.lock",status["scientific_fingerprint"]):
            status["workspace"] = prepare_stage_outputs(ROOT,stage,status["scientific_fingerprint"])
            if stage == "03":
                from .cards import freeze_preheldout_inventory
                freeze_preheldout_inventory(ROOT,ROOT/"artifacts/cards")
            cfg=settings()
            env=build_child_environment(cfg,stop,runid)
            def heartbeat(pid,elapsed):
                status.update(child_pid=pid,heartbeat_utc=utc_now(),elapsed_s=elapsed)
                atomic_write_json(statuspath,status)
            total_started=time.monotonic()
            with (folder/("RUN_"+stage+"_"+runid+".log")).open("w",encoding="utf-8") as log:
                for key,params in STAGES[stage][1]:
                    _assert_release_stable(status["scientific_fingerprint"],status["release_id"],status["release_manifest_sha256"])
                    safe_console_write("\nRUN "+stage+": "+key+"\n")
                    status["current_job"]=key
                    atomic_write_json(statuspath,status)
                    params=list(params)+(["--portable"] if args.portable and key=="check" else [])
                    remaining=float(cfg["max_runner_seconds"])-(time.monotonic()-total_started)
                    if remaining<=0:
                        raise RuntimeError("Runner time budget exhausted before next job.")
                    code=run_child(key,params,env,log,stop,remaining,heartbeat)
                    if code != 0:
                        raise RuntimeError(key+" returned "+str(code)+"; no result readiness is inferred.")
                    _assert_release_stable(status["scientific_fingerprint"],status["release_id"],status["release_manifest_sha256"])
                    status["commands_completed"].append(key)
            _assert_release_stable(status["scientific_fingerprint"],status["release_id"],status["release_manifest_sha256"])
            status.update(status="PASS_PORTABLE_ONLY" if args.portable else "PASS",completed_utc=utc_now())
            atomic_write_json(statuspath,status)
        safe_console_write("RUN_"+stage+"="+status["status"]+"\n")
        return 0
    except BaseException as exc:
        status.update(status="INCOMPLETE" if isinstance(exc,KeyboardInterrupt) else "FAIL",error=repr(exc),ended_utc=utc_now())
        if statuspath is not None:
            atomic_write_json(statuspath,status)
        safe_console_write("RUN_"+stage+"="+status["status"]+": "+str(exc)+"\n")
        return 1
