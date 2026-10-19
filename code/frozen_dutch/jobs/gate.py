"""Compile, collect/run real tests and import actual production entry points."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tpwrs.storage import atomic_write_json
from tpwrs.config import scientific_fingerprint,verify_release_manifest


def execute(command,log,env,timeout=600):
    try:
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        return result.returncode
    except subprocess.TimeoutExpired:
        with log.open('a',encoding='utf-8') as f: f.write('\nGATE_TIMEOUT: command did not complete within its cheap-test budget.\n')
        return 124


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable',action='store_true',help='Allow missing proprietary/engineering dependencies; never authorizes heavy execution.')
    args=parser.parse_args();integrity=verify_release_manifest();print(f'SOURCE_INTEGRITY=PASS files={integrity["source_file_count"]} release={integrity.get("release_id")}',flush=True)
    art=ROOT/'artifacts';art.mkdir(exist_ok=True)
    logs=art/'cheap_tests'/uuid.uuid4().hex;logs.mkdir(parents=True)
    tmp=logs/'tmp';tmp.mkdir()
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),TMPDIR=str(tmp),PYTHONUNBUFFERED='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
    print('[1/4] Compile core, jobs, tests and the single runner.',flush=True)
    targets=['tpwrs','jobs','tests','RUN.py']
    compile_rc=execute([sys.executable,'-m','compileall','-q',*targets],logs/'compile.txt',env)
    print('[2/4] Run the complete test suite; native tests skip only when dependencies are absent.',flush=True)
    junit=logs/'junit.xml'
    test_rc=execute([sys.executable,'-m','pytest','-q','tests','--basetemp',str(tmp/'pytest'),'--junitxml',str(junit)],logs/'pytest.txt',env)
    counts={k:0 for k in ('tests','failures','errors','skipped')}
    if junit.is_file():
        tree=ET.parse(junit)
        for suite in tree.findall('.//testsuite'):
            for k in counts:counts[k]+=int(suite.get(k,0))
    dependency_counts={scope:{key:0 for key in ('passed','failed','errors','skipped')}
                       for scope in ('native_required','portable_reference')}
    if junit.is_file():
        for case in ET.parse(junit).findall('.//testcase'):
            properties={item.get('name'):item.get('value') for item in case.findall('./properties/property')}
            scope=properties.get('dependency_scope','portable_reference')
            outcome=('errors' if case.find('error') is not None else 'failed' if case.find('failure') is not None
                     else 'skipped' if case.find('skipped') is not None else 'passed')
            dependency_counts[scope][outcome]+=1
    print((logs/'pytest.txt').read_text(encoding='utf-8')[-12000:],flush=True)
    print('[3/4] Import actual production targets in one isolated probe; do not construct a scientific case.',flush=True)
    from tpwrs.orchestrator import JOBS
    checks=[]
    for key,(module_name,function_name) in JOBS.items():
        try:
            target=getattr(importlib.import_module(module_name),function_name)
            if not callable(target): raise TypeError("Job target is not callable.")
            checks.append({'script':key,'return_code':0})
        except Exception as exc:
            checks.append({'script':key,'return_code':1,'error':repr(exc)})
    for row in checks:
        if row['return_code']: print(f"IMPORT FAIL: {row['script']}: {row['return_code']}",flush=True)
    print('[4/4] Check native dependencies and a one-variable Gurobi solve.',flush=True)
    modules={};required=('numpy','scipy','pandas','sklearn','pytest','psutil','pyarrow','pandapower','gurobipy')
    for name in required:
        try:
            m=importlib.import_module(name);modules[name]={'imported':True,'version':str(getattr(m,'__version__','available'))}
        except Exception as exc: modules[name]={'imported':False,'error':f'{type(exc).__name__}: {exc}'}
    license_ok=False;probe=None
    if modules['gurobipy']['imported']:
        try:
            import gurobipy as gp
            with gp.Env(params={'OutputFlag':0}) as e:
                with gp.Model(env=e) as m:
                    x=m.addVar(lb=0,ub=1);m.setObjective(x);m.optimize();license_ok=m.Status==gp.GRB.OPTIMAL
                    probe={'status':m.Status,'version':list(gp.gurobi.version())}
        except Exception as exc:probe={'error':f'{type(exc).__name__}: {exc}'}
    portable_ok=compile_rc==0 and test_rc==0 and counts['tests']>0 and not any(c['return_code'] for c in checks)
    native_ok=portable_ok and license_ok and all(v['imported'] for v in modules.values()) and counts['skipped']==0
    payload={'schema':'TPWRS_V3_LIGHTWEIGHT_STATUS_V4','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS' if native_ok else ('PASS_PORTABLE_ONLY' if portable_ok and args.portable else 'FAIL'),
        'native_environment_ready':native_ok,'portable_tests_pass':portable_ok,
        'scientific_fingerprint':scientific_fingerprint(),'release_id':integrity.get('release_id'),'release_manifest_sha256':integrity.get('manifest_sha256'),'python':sys.version,'python_executable':sys.executable,
        'platform':platform.platform(),'compile_return_code':compile_rc,'pytest_return_code':test_rc,
        'pytest_counts':counts,'dependency_test_counts':dependency_counts,'target_import_checks':checks,'modules':modules,
        'gurobi_license_ok':license_ok,'gurobi_probe':probe,'log_root':str(logs),
        'annual_experiments_executed':False,'submission_ready':False}
    atomic_write_json(art/'LIGHTWEIGHT_TEST_STATUS.json',payload)
    print(json.dumps(payload,indent=2),flush=True)
    for key in ('failures','errors','skipped'):
        print(f'pytest {key}={counts[key]}',flush=True)
    print('native_environment_ready='+str(native_ok).lower(),flush=True)
    print(f'LIGHTWEIGHT_GATE={payload["status"]}',flush=True)
    return 0 if native_ok or (portable_ok and args.portable) else 1

if __name__=='__main__':raise SystemExit(main())
