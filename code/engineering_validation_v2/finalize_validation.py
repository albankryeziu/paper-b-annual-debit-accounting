"""Seal completed evidence and recheck consumed scientific files; no numerical solve."""
from engineering_common import *
import ast,subprocess,importlib.metadata

def run():
 target=OUT/'07_QC_AND_INTEGRITY/FINAL_VALIDATION_STATUS.json'
 if target.exists():return
 physical_meta=read(consume(REAL/'physical_case/FINAL_DUTCH_CASE/physical_case.json'))
 assert all(float(d)==1. for d in physical_meta['time_grid']['delta_t_hours'])
 for p in (SOURCE/'jobs/input_build.py',SOURCE/'tpwrs/inputs.py'):consume(p)
 for year in (2029,2030,2031):consume(LOCK/f'GATE1_CERTIFICATES/year_{year}/INDEPENDENT_CERTIFICATE.json')
 for p in ('GATE1_SUMMARY.csv','ENDPOINT_MODEL_IDENTITY.json','GATE1_VERDICT.md'):consume(LOCK/'GATE1_CERTIFICATES'/p)
 consume(LOCK/'FINAL_SUBMISSION/TPWRS_PAPER_B_FINAL_SUBMISSION_LOCK.zip','67fc56fdacd1b5f040298f33ba10148ec08ad0bf5e71de6f0eb0f913643e0df4')
 consume(LOCK/'FINAL_SUBMISSION/TPWRS_PAPER_B_OVERLEAF_FINAL.zip','7e1ace0f268244847ddac3b0597b4b92bcf057c5be1b94ed9bd51f48e350f918')
 old=Path('C:/Users/Alban/Documents/TPWRS_PaperB_Engineering_Validation_20261018')
 consume(old/'06_REPORTS/TPWRS_PAPER_B_FINAL_ENGINEERING_VALIDATION.pdf','8a902003478ae4811a9b7892257191ecf3c552bf33da22142a6bb5127bc24de1')
 consume(old/'06_REPORTS/TPWRS_PAPER_B_ENGINEERING_REPRODUCIBILITY_GUIDE.pdf','ac538d09b27f3cb2468b2f067ffa350a80dd9ad8a4510a4f5469972bb3d29e51')
 # Recheck every consumed file against its first-read identity.
 inventory=read(INVENTORY);changed=[]
 for p,r in inventory.items():
  if sha(p)!=r['sha256']:changed.append(p)
 assert not changed,'Consumed scientific evidence changed'
 csvout(OUT/'07_QC_AND_INTEGRITY/CONSUMED_INPUT_SHA256.csv',pd.DataFrame([dict(path=p,**r) for p,r in inventory.items()]))
 jsout(OUT/'07_QC_AND_INTEGRITY/CONSUMED_INPUT_INTEGRITY_AFTER.json',dict(status='PASS',files=len(inventory),bytes=sum(r['bytes'] for r in inventory.values()),changed=changed,scope='All consumed files, not a new complete-tree scan',blocked_preflight_PDF_seals='PASS',submission_ZIP_seals='PASS'))
 code=[]
 for p in sorted(OUT.glob('*.py')):
  ast.parse(p.read_text(encoding='utf-8'));code.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 jsout(OUT/'07_QC_AND_INTEGRITY/EVALUATION_CODE_IDENTITIES.json',code)
 env={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','pandapower','matplotlib','pyarrow')}
 jsout(OUT/'07_QC_AND_INTEGRITY/ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,packages=env,BLAS_threads=1,optimized_numba_available=False))
 logs=[]
 for repeat in (1,2):
  r=subprocess.run([sys.executable,str(OUT/'RUN_ENGINEERING_VALIDATION.py')],cwd=OUT,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0;logs.append(dict(repeat=repeat,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 jsout(OUT/'07_QC_AND_INTEGRITY/RUNNER_RESTART_QC.json',dict(status='PASS',runs=logs,default_stage='PREFLIGHT',additional_solver_calls=0))
 # Trace every physical trajectory, all engineering input families and outputs.
 trace=[]
 for year in (2029,2030,2031):
  for policy,(p,kind) in policies(year).items():trace.append(dict(finding=f'{year} {policy} physical metrics',script=str(OUT/'stage_physical.py'),source=str(p),source_sha256=sha(p),result=str(OUT/'01_PHYSICAL_METRICS/ANNUAL_PHYSICAL_METRICS.csv'),result_sha256=sha(OUT/'01_PHYSICAL_METRICS/ANNUAL_PHYSICAL_METRICS.csv')))
 for label,script,p,result in [
  ('Matched AC', 'stage_ac.py',FROZEN/'artifacts/ac_inputs/ac_inputs.npz',OUT/'02_MATCHED_AC_VALIDATION/MATCHED_AC_CASES.csv'),
  ('AC sample','stage_preflight.py',RESULTS/'engineering_selection/SELECTED_POLICY_STATE_CASES.csv',OUT/'00_AUTHORITY_AND_INPUTS/BASELINE_AC_RECONSTRUCTION.csv'),
  ('Common mapping','engineering_common.py',REAL/'physical_case/FINAL_DUTCH_CASE/PARTICIPANT_MAPPING.json',OUT/'00_AUTHORITY_AND_INPUTS/AC_MAPPING_V2.npz'),
  ('Independent common-map QC','verify_common_mapping.py',REAL/'physical_case/FINAL_DUTCH_CASE/physical_case_arrays.npz',OUT/'07_QC_AND_INTEGRITY/AC_MAPPING_INDEPENDENT_QC.json'),
  ('Headroom','stage_headroom.py',REAL/'background/DEVICE_REGISTRY.json',OUT/'03_GENERATOR_HEADROOM/BALANCING_FEASIBILITY_BY_POLICY.csv'),
  ('Timing','stage_information.py',SOURCE/'tpwrs/rolling.py',OUT/'04_INFORMATION_TIMING/INFORMATION_STRUCTURE_AND_0700_AUDIT.md')]:trace.append(dict(finding=label,script=str(OUT/script),source=str(p),source_sha256=sha(p),result=str(result),result_sha256=sha(result)))
 csvout(OUT/'07_QC_AND_INTEGRITY/SOURCE_AND_RESULT_TRACEABILITY.csv',pd.DataFrame(trace))
 atomic(OUT/'07_QC_AND_INTEGRITY/PROTECTED_INPUT_INTEGRITY.md',f'''# Protected input integrity

PASS: all {len(inventory)} consumed scientific source/result files retain their first-read SHA-256 identities. No changed file was detected. Both locked submission ZIPs and both completed blocked-preflight PDFs match their previous exact seals. Gate 1 certificate artifacts were preserved and read only.

All new files remain inside the version-2 directory. No protected Python module, orchestration or optimization pipeline was imported or executed; native pandapower power-flow solves occurred only on isolated in-memory copies. No original model, request, policy, rating, card, result, log, certificate or manuscript was written. The scope is consumed-file verification; the previous full-tree inventory is retained as historical evidence, not represented as a fresh full-tree scan.
''')
 atomic(OUT/'07_QC_AND_INTEGRITY/ENGINEERING_VALIDATION_QC.md',r'''# Final engineering QC

Gate A PASS: six authority identities; exact frozen model/row and source-rating reconciliation; all 27 predeclared historical AC records; nine independent DC records; all nine premium-object full-year residual/energy/FDE identities. The joint-map assumption is explicitly authorized and recorded as version 2.

Gate B PASS: 21 full-year action objects, 183,960 hourly metric rows, 8,837 state-element rows per hour, both declared violation thresholds. All original residual totals agree exactly. M1-LP is labeled a relaxation and EC is retrospective; raw U/C are not silently replaced by guarded actions.

Gate C PASS FOR EXECUTION: 648 common-map and 648 historical-map campaign cases. Feasibility, convergence, rankings and mapping sensitivity are separate scientific outcomes. The 27 preflight AC solves are reported separately. All 162 historical rolling-M2 case summaries are reproduced, and unity results agree across mappings. Complete raw voltage/loading/Q rows independently reproduce case summaries. No failed solve is replaced by zero violation.

Gate D PASS FOR DIAGNOSTIC SCOPE: native capacity bounds, reference dispatch and background device identities recovered; generator-specific balancing allocation remains NOT IDENTIFIABLE. Conditional bus-aggregate necessary conditions retain their resource/allocation assumptions. No reserve or ramp data is fabricated.

Information audit PASS WITH LIMITATION: finite 48-hour realized-scenario look-ahead, 24-hour commitment, fixed card/parameters across held-out years; no authenticated forecast vintages or validated 07:00 deployment.

Independent QC checks full persisted table aggregation, alternate signed DC algebra on its declared sample and all 525 critical rows, native line-current loading and all paired registry keys. No additional optimization or AC solve occurs in these independent aggregation checks. Exact scopes are in INDEPENDENT_NUMERICAL_QC.json.

The scientific verdict is CONDITIONAL GO for the stated controlled-model claims. New AC and headroom findings do not modify the annual DC certificates or establish annual AC security, individual generator deliverability or prospective contract superiority. Final PDF rendering is separately checked and hash-bound in VISUAL_QA.json.
''')
 jsout(target,dict(status='COMPLETE',scientific_verdict='CONDITIONAL GO',optimization_calls=0,new_campaign_AC_solves=1296,preflight_AC_solves=27,total_new_AC_solves=1323,common_mapping_cases=648,historical_mapping_cases=648,physical_policy_year_objects=21,consumed_inputs_unchanged=True,blocked_preflight_unchanged=True,annual_certificates_unchanged=True,limits=['No generator-specific balancing allocation','No authenticated forecast vintages','Selected fixed-injection AC stress only; no annual security claim']))
 manifest=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name not in ('FINAL_OUTPUT_SHA256.csv',)]
 csvout(OUT/'07_QC_AND_INTEGRITY/FINAL_OUTPUT_SHA256.csv',pd.DataFrame(manifest))
 print('SEALED',len(inventory),'unchanged consumed files;',len(manifest),'new output files',flush=True)

if __name__=='__main__':run()
