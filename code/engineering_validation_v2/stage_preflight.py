from engineering_common import *
from ac_evaluation import evaluate
import math,platform,time

def run():
 target=OUT/'00_AUTHORITY_AND_INPUTS/PREFLIGHT_QC.json'
 if target.exists():
  r=read(target)
  if r['status']!='PASS':raise RuntimeError('Preflight is blocked; preserve evidence and diagnose.')
  print('PREFLIGHT already complete');return
 started=time.monotonic();checks={}
 manifest=consume(LOCK/'FINAL_SUBMISSION/FINAL_MANUSCRIPT_HASHES.csv')
 identities=[]
 for row in pd.read_csv(manifest).to_dict('records'):
  if row['package_member'] in ('main.tex','main.pdf','TECHNICAL_REPORT_ZENODO.tex','TECHNICAL_REPORT_ZENODO.pdf','GATE1_CERTIFICATE_VERIFICATION_REPORT.pdf','GATE3_FINAL_SCIENTIFIC_AND_SUBMISSION_AUDIT.pdf'):
   consume(row['absolute_source'],row['sha256']);identities.append(row)
 text=(LOCK/'GATE3_MANUSCRIPT/main.tex').read_text(encoding='utf-8')
 assert 'Independently reconstructed' in text and 'marginal-relief premium' in text
 assert all(token in text for token in ('57.909','37.782','23.659'))
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/AUTHORITY_IDENTITIES.csv',pd.DataFrame(identities))
 for name in ('jobs/ac.py','jobs/dc.py','jobs/selection.py','tpwrs/engineering.py','tpwrs/network.py','tpwrs/rolling.py','jobs/debit_rule_family.py','tpwrs/debit_rules.py'):
  consume(SOURCE/name)
 meta,a=load_case();net,m,q=load_ac(meta)
 import pandapower as pp
 dates=a['dates'];assert dates.equals(pd.date_range('2025-01-01','2031-12-31T23:00',freq='h',tz='UTC'))
 assert len(meta['bus_ids'])==45 and len(meta['participant_ids'])==114 and len(meta['network_states'])==94 and len(a['F'])==8837
 assert np.max(np.abs(a['A'].sum(axis=0)))<1e-12
 assert np.array_equal(a['xbar_mw'],a['eligibility_mask']*np.minimum(a['requests_mw'],a['capacities_mw']))
 assert np.all(a['F']>0) and np.all(a['capacities_mw']>0)
 assert q['q_fixed_mvar'].shape==(len(dates),45) and q['generator_p_mw'].shape==(len(dates),69)
 assert np.allclose(q['participant_q_connection'].sum(axis=0),1,atol=1e-12,rtol=0)
 # User-declared v2 common-map assumption; preserve historical Q matrix.
 C,pi=connection_matrices(meta,a,q)
 orient=np.array([1 if d=='import' else -1 for d in meta['participant_directions']])
 keys=C-a['A']*orient
 checks['common_balancing_key_column_spread']=float(np.max(np.abs(keys-keys[:,[0]])))
 assert checks['common_balancing_key_column_spread']<1e-12
 import io
 buf=io.BytesIO();np.savez_compressed(buf,C_P_final=C,C_Q_common=C,C_Q_historical=q['participant_q_connection'],balancing_key=pi)
 atomic(OUT/'00_AUTHORITY_AND_INPUTS/AC_MAPPING_V2.npz',buf.getvalue())
 configuration={'version':'ENGINEERING_AC_V2_COMMON_PQ','modeling_authority':'User decision: C_Q_new=C_P_final; declared participant-level constant-PF assumption, not undocumented device-location reconstruction',
  'mappings':['COMMON_PQ_V2','HISTORICAL_TWO_MAP'],'physical_case_hash':meta['physical_case_hash'],'policy_years':[2029,2030,2031],
  'power_factors':['unity','lagging_098','leading_098'],'solver':{'algorithm':'nr','initializations':['dc','flat'],'enforce_q_lims':True,'max_iteration':100,'tolerance_mva':1e-8},
  'unchanged':'Active A, trajectories, background P/Q, PV reference P/voltage/Q bounds, slack, network, outages, ratings and solver settings',
  'mapping_arrays_sha256':sha(OUT/'00_AUTHORITY_AND_INPUTS/AC_MAPPING_V2.npz'),'planned_common_cases':648,'planned_historical_premium_cases':486}
 jsout(OUT/'00_AUTHORITY_AND_INPUTS/AC_CONFIGURATION_V2.json',configuration)
 checks['A_balance_max']=float(np.max(np.abs(a['A'].sum(axis=0))))
 checks['background_balance_max_mw']=float(np.max(np.abs(a['p_background_mw'].sum(axis=1))))
 # Rating and topology alignment for every row, not dimension matching.
 cfg=read(consume(REAL/'LOCAL_SOURCE_CONFIG.json'))['records']['ratings'];rf=pd.read_csv(consume(cfg['path'],cfg['sha256']));rf=rf[rf.candidate_id.astype(str)=='AGGREGATED_EHV']
 mapping=[];offset=0
 for s in meta['network_states']:
  actual=[]
  for kind in ('line','trafo'):
   for ix,row in net[kind].iterrows():
    rid=kind+'::'+str(ix)
    if row.in_service and s['state_id']!='OUT::'+rid:actual.append(rid)
  assert actual==s['row_ids']
  for j,rid in enumerate(s['row_ids']):
   kind,ix=rid.split('::');r=net[kind].loc[int(ix)]
   if kind=='line':native=np.sqrt(3)*net.bus.at[int(r.from_bus),'vn_kv']*r.max_i_ka*r.parallel*r.df;fr=int(r.from_bus);to=int(r.to_bus)
   else:native=r.sn_mva*r.parallel*r.df;fr=int(r.hv_bus);to=int(r.lv_bus)
   # DC central active-power ratings and native AC current ratings are distinct
   # frozen conventions. Check each against its own source; do not equate them.
   hit=rf[(rf.element_type==kind)&(rf.element_index.astype(int)==int(ix))];assert len(hit)==1
   assert abs(float(hit.iloc[0].central_rating_mw)-a['F'][offset+j])<1e-8
   mapping.append(dict(flat_row=offset+j,state_id=s['state_id'],row_id=rid,from_bus=fr,to_bus=to,dc_rating_mw=float(a['F'][offset+j]),ac_nominal_rating_mva=float(native)))
  offset+=len(s['row_ids'])
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/EXACT_ROW_MAPPING.csv',pd.DataFrame(mapping))
 registry_path=consume(RESULTS/'engineering_selection/SELECTED_POLICY_STATE_CASES.csv')
 freeze=read(consume(RESULTS/'engineering_selection/SELECTED_STATE_REGISTRY_FREEZE.json'))
 assert sha(registry_path)==freeze['outputs']['cases']['sha256'] and freeze['physical_case_hash']==meta['physical_case_hash']
 reg=pd.read_csv(registry_path);sample=reg[['year','time_index','timestamp_utc','state_id']].drop_duplicates().sort_values(['year','time_index','state_id'])
 assert len(sample)==54 and len(sample[['year','time_index']].drop_duplicates())==18
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/FROZEN_AC_SAMPLE.csv',sample)
 # Predeclare all states and PFs at the earliest selected UTC hour in each year.
 base=sample[sample.time_index==sample.groupby('year').time_index.transform('min')].copy()
 base=base.merge(pd.DataFrame({'pf_case':['unity','lagging_098','leading_098']}),how='cross')
 assert len(base)==27
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/BASELINE_CASES_PREDECLARED.csv',base)
 atomic(OUT/'00_AUTHORITY_AND_INPUTS/BASELINE_TOLERANCES.md','Predeclared reconstruction tolerances: voltage extrema and full bus vectors 1e-8 pu; line/transformer loading 1e-5 percentage point; normalized AC overload 1e-6; slack P and Q excess 1e-4 MW/Mvar. Convergence and violation counts must agree exactly. Independent DC row-flow tolerance max(1e-4 MW,1e-8 rating). All 27 baseline cases must pass. No premium AC is executed before this gate.\n')
 prior=read(consume(LOCK/'VERIFICATION_REPORTS/FRESH_REPLAY/INDEPENDENT_NUMERICAL_EVIDENCE.json'))['replays']
 # Fresh whole-year replay of precisely the three premium objects before any AC call.
 policy_qc=[]
 for year in (2029,2030,2031):
  ix=np.flatnonzero(dates.year==year);assert len(ix)==8760
  for label,(p,kind) in policies(year).items():
   x=action(p);assert x.shape==(8760,114) and np.isfinite(x).all()
   old=[r for r in prior if r['source']==str(p)]
   assert len(old)==1 and old[0]['sha256']==sha(p)
   if label not in ('U','C','EC'):continue
   totals=[]
   for start in range(0,8760,128):
    ids=ix[start:start+128];req=a['p_background_mw'][ids]-a['requests_mw'][ids]@a['A'].T
    f=(req+x[start:start+len(ids)]@a['A'].T)@a['H'].T+a['h']
    totals.append(float(np.maximum(np.abs(f)-a['F'],0).sum()))
   S=math.fsum(totals);energy=float(x.sum());fde=float((x/a['capacities_mw']).sum())
   ds=S-old[0]['S'];de=energy-old[0]['E_MWh'];df=fde-old[0]['FDE_h']
   assert abs(ds)<.01 and abs(de)<1e-5 and abs(df)<1e-7
   policy_qc.append(dict(year=year,policy=label,object=kind,path=str(p),sha256=sha(p),S=S,E_MWh=energy,FDE_h=fde,S_difference=ds,energy_difference=de,FDE_difference=df))
   print('PREFLIGHT POLICY',year,label,'residual difference',ds,flush=True)
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/POLICY_IDENTITY_QC.csv',pd.DataFrame(policy_qc))
 old_ac=pd.read_csv(consume(RESULTS/'selected_ac/CASES.csv'))
 old_l=pd.read_csv(consume(RESULTS/'selected_ac/LOADING.csv'));old_v=pd.read_csv(consume(RESULTS/'selected_ac/VOLTAGES.csv'))
 results=[];dcs=[]
 states={s['state_id']:s for s in meta['network_states']};rowoffset={}
 for j,(sid,rid) in enumerate(a['rows']):rowoffset.setdefault(sid,j)
 xs={y:action(policies(y)['ROLLING_M2'][0]) for y in (2029,2030,2031)}
 for row in base.to_dict('records'):
  year=int(row['year']);t=int(row['time_index']);gi=np.flatnonzero(dates.year==year)[t];x=xs[year][t];v=a['requests_mw'][gi]
  assert pd.Timestamp(row['timestamp_utc'])==dates[gi]
  p=a['p_background_mw'][gi]-a['A']@v+a['A']@x
  reactive=q['q_fixed_mvar'][gi]+q_delta(q['participant_q_connection'],v,x,meta['participant_directions'],row['pf_case'])
  test=build_ac(net,row['state_id'],meta['bus_ids'],p,reactive,m['generator_ids'],q['generator_p_mw'][gi])
  s=states[row['state_id']];ident=dict(row,policy='ROLLING_M2',case_id=f"{year}_{t:05d}_{row['state_id']}_RHO15_ROLLING_M2")
  rec,ls,vs,qs=evaluate(test,s,ident)
  old=old_ac[(old_ac.case_id==ident['case_id'])&(old_ac.pf_case==row['pf_case'])];assert len(old)==1;old=old.iloc[0]
  diffs={}
  for col,tol in [('min_vm_pu',1e-8),('max_vm_pu',1e-8),('maximum_loading_percent',1e-5),('ac_apparent_normalized_thermal_exceedance',1e-6),('slack_p_mw',1e-4),('max_q_limit_violation_mvar',1e-4)]:
   diffs[col+'_difference']=rec.get(col,float('inf'))-float(old[col]);assert abs(diffs[col+'_difference'])<=tol,(ident,col,diffs)
  for col in ('converged','voltage_violations','thermal_violations','ac_feasible_under_declared_checks'):assert rec[col]==old[col],(ident,col)
  ol=old_l[(old_l.case_id==ident['case_id'])&(old_l.pf_case==row['pf_case'])].set_index('row_id')
  ov=old_v[(old_v.case_id==ident['case_id'])&(old_v.pf_case==row['pf_case'])].copy();ov.bus_id=ov.bus_id.astype(str);ov=ov.set_index('bus_id')
  assert max(abs(r['loading_percent']-ol.at[r['row_id'],'loading_percent']) for r in ls)<1e-5
  assert max(abs(r['vm_pu']-ov.at[r['bus_id'],'vm_pu']) for r in vs)<1e-8
  results.append(dict(rec,**diffs))
  if row['pf_case']=='unity':
   testdc=build_ac(net,row['state_id'],meta['bus_ids'],p,reactive,m['generator_ids'],q['generator_p_mw'][gi]);pp.rundcpp(testdc,check_connectivity=True)
   native=state_flows(testdc,s);off=rowoffset[s['state_id']];n=len(s['row_ids']);aff=a['H'][off:off+n]@p+a['h'][off:off+n]
   error=float(np.max(np.abs(native-aff)));assert np.all(np.abs(native-aff)<=np.maximum(1e-4,1e-8*a['F'][off:off+n]))
   dcs.append(dict(row,maximum_flow_error_mw=error))
  print('BASELINE PASS',year,t,row['state_id'],row['pf_case'],flush=True)
 csvout(OUT/'00_AUTHORITY_AND_INPUTS/BASELINE_AC_RECONSTRUCTION.csv',pd.DataFrame(results));csvout(OUT/'00_AUTHORITY_AND_INPUTS/BASELINE_DC_RECONSTRUCTION.csv',pd.DataFrame(dcs))
 checks.update(authority='LOCK_MANIFEST_SHA256_AND_CONTENT_VERIFIED',physical_case_hash=meta['physical_case_hash'],baseline_ac_cases=len(results),baseline_dc_cases=len(dcs),premium_trajectory_full_year_checks=len(policy_qc),planned_common_AC_cases=648,planned_historical_premium_cases=486,full_campaign_executed=False,python=platform.python_version(),pandapower=pp.__version__,numpy=np.__version__,elapsed_s=time.monotonic()-started)
 jsout(target,dict(status='PASS',checks=checks))

if __name__=='__main__':run()
