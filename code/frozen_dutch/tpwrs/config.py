from __future__ import annotations


# ============================================================================
# tpwrs_v3.release_config (consolidated; see SOURCE_MAP.json)
# ============================================================================
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE=Path(r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\Final Paper Script\PAPER B\IEEE_TPWRS\TPWRS_M1_M2_M3_IMPLEMENTATION_FINAL_20260825\revision_v3")

def settings():
    p=ROOT/"LOCAL_SETTINGS.json"
    data=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    data.setdefault("existing_workspace",str(DEFAULT_WORKSPACE))
    data.setdefault("stage", "00")
    data.setdefault("max_stage_seconds",3600)
    data.setdefault("max_case_seconds",14400)
    data.setdefault("max_runner_seconds",86400)
    import math
    for key in ("max_stage_seconds", "max_case_seconds", "max_runner_seconds"):
        if isinstance(data[key], bool) or not math.isfinite(float(data[key])) or float(data[key]) <= 0:
            raise ValueError(key + " must be finite and positive.")
    cap=data.get("gurobi_softmem_gb")
    if cap is not None and (isinstance(cap,bool) or not math.isfinite(float(cap)) or float(cap)<=0):
        raise ValueError("gurobi_softmem_gb must be null or finite and positive.")
    return data

def existing_workspace():
    return Path(settings()["existing_workspace"])

def source_records():
    candidates=(ROOT/"local/SOURCE_PATHS.json", ROOT/"artifacts/real_case/LOCAL_SOURCE_CONFIG.json",
                existing_workspace()/"artifacts/real_case/LOCAL_SOURCE_CONFIG.json")
    for p in candidates:
        if p.is_file():
            d=json.loads(p.read_text(encoding="utf-8")); return d.get("records",d)
    raise FileNotFoundError("No source registry. Run RUN.py with stage 01 to import the existing frozen case, or fill local/SOURCE_PATHS.json from its example.")

def source_path(name,verify=True):
    from tpwrs.storage import sha256_file
    records=source_records()
    if name not in records: raise FileNotFoundError(f"Required source key absent: {name}")
    row=records[name]; p=Path(row if isinstance(row,str) else row['path'])
    if not p.is_absolute(): p=ROOT/p
    if not p.is_file(): raise FileNotFoundError(f"Required source {name}: {p}")
    if verify and isinstance(row,dict) and row.get('sha256') and sha256_file(p)!=row['sha256']:
        raise ValueError(f"Source content changed: {name}: {p}")
    return p

def scientific_source_paths():
    """Return the complete scientific/runtime source closure used by production.

    Production experiment identity is intentionally conservative: every Python
    source that can influence numerical generation, validation, provenance or
    reporting, plus frozen JSON methodology specifications, is included.  RUN_02
    keeps its separate narrow annual-solver qualification identity.
    """
    paths=list((ROOT/'tpwrs').rglob('*.py'))+list((ROOT/'jobs').glob('*.py'))
    paths+=[p for p in (ROOT/'specification').glob('*') if p.is_file()]
    return tuple(sorted(set(paths)))

def scientific_source_hashes():
    from tpwrs.storage import sha256_file
    return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in scientific_source_paths()}

def scientific_fingerprint():
    from tpwrs.storage import sha256_json
    return sha256_json(scientific_source_hashes())


def qualified_solver_profile():
    """Use exactly the solver profile qualified on 2028 for the current annual solver.

    The qualification identity is intentionally narrower than the whole-release
    fingerprint.  Downstream M3/features/reporting changes do not invalidate an
    unchanged annual solver, while any change to the annual mathematics, replay,
    numerical contract or qualified solver profile does.
    """
    from tpwrs.execution import annual_solver_fingerprint
    from tpwrs.contracts import SolverProfile
    from tpwrs.storage import sha256_file
    path=ROOT/'artifacts/performance/SELECTED_SOLVER_PROFILE.json'
    gate=ROOT/'results_2028_performance_regression/STATUS.json'
    if not path.is_file() or not gate.is_file():raise RuntimeError('Run the complete 2028 qualification before production solves.')
    record=json.loads(path.read_text(encoding='utf-8'));status=json.loads(gate.read_text(encoding='utf-8'))
    try:
        annual_fp=annual_solver_fingerprint(ROOT)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError('2028 qualification/profile is missing, incomplete, altered or from different annual-solver code.') from exc
    if (record.get('status')!='PASS' or record.get('annual_solver_fingerprint')!=annual_fp
        or status.get('status')!='PASS' or status.get('full_year_completed') is not True
        or status.get('annual_solver_fingerprint')!=annual_fp
        or status.get('selected_profile_sha256')!=sha256_file(path)):
        raise RuntimeError('2028 qualification/profile is missing, incomplete, altered or from different annual-solver code.')
    physical_meta=ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE/physical_case.json'
    if not physical_meta.is_file() or json.loads(physical_meta.read_text(encoding='utf-8')).get('physical_case_hash')!=record.get('parent_physical_case_hash'):
        raise RuntimeError('The currently installed physical case is not the one qualified in 2028.')
    return SolverProfile(**record['profile'])


def _executable_source_relatives():
    """Executable files whose presence can affect this release.

    This exact-set check is what makes overlay extraction safe: an obsolete Python
    file left behind by an older release is rejected instead of silently entering
    the scientific fingerprint/import surface.
    """
    paths=list((ROOT/'tpwrs').rglob('*.py'))
    paths+=list((ROOT/'jobs').rglob('*.py'))
    paths+=list((ROOT/'tpwrs_v3').rglob('*.py'))
    paths+=list((ROOT/'scripts').rglob('*.py'))
    paths+=list((ROOT/'final_runtime').rglob('*.py'))
    paths+=list((ROOT/'data_processing').rglob('*.py'))
    paths+=list((ROOT/'tests').rglob('*.py'))
    # Every root-level Python file is import-visible because runners execute with
    # ROOT on sys.path.  An obsolete file such as json.py could shadow the
    # standard library, so overlay safety must reject it too.
    paths+=list(ROOT.glob('*.py'))
    return {p.relative_to(ROOT).as_posix() for p in paths}

def verify_release_manifest():
    from tpwrs.storage import sha256_file
    path=ROOT/'RELEASE_MANIFEST.json'
    if not path.is_file():raise RuntimeError('RELEASE_MANIFEST.json is missing. Extract the complete new ZIP, not individual scripts.')
    manifest=json.loads(path.read_text(encoding='utf-8'));errors=[]
    managed=manifest.get('source_files',{})
    for relative,expected in managed.items():
        p=ROOT/relative
        if not p.is_file():errors.append('missing: '+relative)
        elif sha256_file(p)!=expected:errors.append('changed: '+relative)
    if not managed:errors.append('empty source manifest')
    expected_executable={name for name in managed if name.endswith('.py') and (name.startswith(('tpwrs/','jobs/','final_runtime/','data_processing/','tests/')) or '/' not in name)}
    actual_executable=_executable_source_relatives()
    for relative in sorted(actual_executable-expected_executable):
        errors.append('unexpected executable source: '+relative)
    for relative in sorted(expected_executable-actual_executable):
        if 'missing: '+relative not in errors: errors.append('missing executable source: '+relative)
    actual_scientific={p.relative_to(ROOT).as_posix() for p in scientific_source_paths()}
    expected_scientific={name for name in managed if ((name.startswith(('tpwrs/','jobs/','final_runtime/')) and name.endswith('.py')) or name.startswith('specification/'))}
    for relative in sorted(actual_scientific-expected_scientific):
        errors.append('unexpected scientific source: '+relative)
    for relative in sorted(expected_scientific-actual_scientific):
        if 'missing: '+relative not in errors: errors.append('missing scientific source: '+relative)
    if errors:raise RuntimeError('Source release integrity failed: '+'; '.join(errors[:20]))
    return {'source_file_count':len(managed),'status':'PASS','release_id':manifest.get('release_id'),'manifest_sha256':sha256_file(path)}
