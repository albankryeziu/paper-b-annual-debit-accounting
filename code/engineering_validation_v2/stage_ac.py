"""Versioned matched AC replay; original policies and physical parameters untouched."""
from engineering_common import *
from ac_evaluation import evaluate
import io,time

def stream_table(paths,kind,target):
 fields=set()
 for p in paths:
  for r in read(p)[kind]:fields.update(r)
 fields=sorted(fields);buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader()
 for p in paths:writer.writerows(read(p)[kind])
 atomic(target,buf.getvalue())

def comparisons(cases):
 pairs=[];keys=['year','time_index','state_id','pf_case']
 metrics={'ac_apparent_normalized_thermal_exceedance':1e-4,'maximum_loading_percent':1e-4,'maximum_voltage_limit_deviation_pu':1e-6,'max_q_limit_violation_mvar':1e-6}
 for mapping,group in cases.groupby('mapping'):
  for key,g in group.groupby(keys):
   by=g.set_index('policy')
   for left,right in [('C','U'),('C','ROLLING_M2'),('EC','U')]:
    A=by.loc[left];B=by.loc[right];usable=bool(A.converged and B.converged)
    r=dict(zip(keys,key));r.update(mapping=mapping,left=left,right=right,both_converged=usable,
      feasibility_class='NONCONVERGENCE' if not usable else 'BOTH' if A.ac_feasible_under_declared_checks and B.ac_feasible_under_declared_checks else 'ONLY_A' if A.ac_feasible_under_declared_checks else 'ONLY_B' if B.ac_feasible_under_declared_checks else 'NEITHER')
    for metric,tol in metrics.items():
     d=float(A[metric]-B[metric]) if usable else None;r[metric+'_A_minus_B']=d;r[metric+'_winner']='NONCONVERGENCE' if d is None else 'TIE' if abs(d)<=tol else 'A' if d<0 else 'B'
    pairs.append(r)
 return pd.DataFrame(pairs)

def run():
 require_preflight();folder=OUT/'02_MATCHED_AC_VALIDATION';target=folder/'MATCHED_AC_QC.json'
 if target.exists():assert read(target)['status']=='PASS';return
 started=time.monotonic();meta,a=load_case();net,m,q=load_ac(meta);Cp,pi=connection_matrices(meta,a,q)
 config=OUT/'00_AUTHORITY_AND_INPUTS/AC_CONFIGURATION_V2.json';configsha=sha(config)
 sample=pd.read_csv(OUT/'00_AUTHORITY_AND_INPUTS/FROZEN_AC_SAMPLE.csv');states={s['state_id']:s for s in meta['network_states']}
 xs={};sourcehash={}
 for year in (2029,2030,2031):
  for policy in ('ROLLING_M2','U','C','EC'):
   p,kind=policies(year)[policy];xs[(year,policy)]=action(p);sourcehash[(year,policy)]=sha(p)
 paths=[];inputqc=[]
 for row in sample.to_dict('records'):
  year=int(row['year']);t=int(row['time_index']);gi=np.flatnonzero(a['dates'].year==year)[t];assert pd.Timestamp(row['timestamp_utc'])==a['dates'][gi]
  for policy in ('ROLLING_M2','U','C','EC'):
   x=xs[(year,policy)][t];v=a['requests_mw'][gi];p=a['p_background_mw'][gi]-a['A']@v+a['A']@x
   sign=np.array([-1 if d=='import' else 1 for d in meta['participant_directions']]);individual=sign*(v-x)*np.tan(np.arccos(.98))
   diff_direct=(Cp-q['participant_q_connection'])@individual
   diff_independent=np.zeros(45)
   for j in range(114):diff_independent+=(Cp[:,j]-q['participant_q_connection'][:,j])*individual[j]
   err=float(np.max(np.abs(diff_direct-diff_independent)));assert err<1e-10
   totalerror=float(abs(np.sum(Cp@individual)-np.sum(q['participant_q_connection']@individual)));assert totalerror<1e-9
   inputqc.append(dict(row,policy=policy,active_injections_changed=False,reactive_total_difference_mvar=totalerror,independent_spatial_difference_error_mvar=err,maximum_spatial_Q_change_mvar=float(np.max(np.abs(diff_direct)))))
   for pf in ('unity','lagging_098','leading_098'):
    for mapping,C in [('HISTORICAL_TWO_MAP',q['participant_q_connection']),('COMMON_PQ_V2',Cp)]:
     identity=dict(row,policy=policy,pf_case=pf,mapping=mapping,policy_source_sha256=sourcehash[(year,policy)],configuration_sha256=configsha,case_id=f'{year}_{t:05d}_{row["state_id"]}_{policy}')
     key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();cp=folder/'cases'/f'{key}.json'
     if cp.exists():
      cached=read(cp);assert cached['identity']==identity;paths.append(cp);continue
     reactive=q['q_fixed_mvar'][gi]+q_delta(C,v,x,meta['participant_directions'],pf)
     test=build_ac(net,row['state_id'],meta['bus_ids'],p,reactive,m['generator_ids'],q['generator_p_mw'][gi])
     rec,ls,vs,qs=evaluate(test,states[row['state_id']],identity)
     # Retain frozen normalized DC index solely as a parallel diagnostic.
     state=states[row['state_id']];offset=sum(len(s['row_ids']) for s in meta['network_states'][:state['index']]);nr=len(state['row_ids'])
     dc=a['H'][offset:offset+nr]@p+a['h'][offset:offset+nr]
     rec['dc_active_normalized_thermal_exceedance']=float(np.maximum(np.abs(dc)/a['F'][offset:offset+nr]-1,0).sum())
     jsout(cp,dict(identity=identity,case=[rec],loading=ls,voltage=vs,reactive=qs));paths.append(cp)
     if len(paths)%24==0:print('AC PROGRESS',len(paths),'/1296',year,t,policy,pf,mapping,flush=True)
 assert len(paths)==1296 and len(set(paths))==1296
 cases=pd.DataFrame([read(p)['case'][0] for p in paths]);assert not cases.duplicated(['mapping','year','time_index','state_id','pf_case','policy']).any()
 csvout(folder/'MATCHED_AC_CASES.csv',cases)
 for name,key in [('MATCHED_AC_LOADING.csv','loading'),('MATCHED_AC_VOLTAGES.csv','voltage'),('MATCHED_AC_REACTIVE_LIMITS.csv','reactive')]:stream_table(paths,key,folder/name)
 csvout(folder/'AC_MAPPING_INPUT_QC.csv',pd.DataFrame(inputqc));pair=comparisons(cases);csvout(folder/'MATCHED_AC_PAIRWISE_COMPARISONS.csv',pair)
 summaries=[]
 for keys,g in cases.groupby(['mapping','year','policy']):
  valid=g[g.converged]
  summaries.append(dict(mapping=keys[0],year=keys[1],policy=keys[2],cases=len(g),converged=int(g.converged.sum()),feasible=int(g.ac_feasible_under_declared_checks.sum()),thermal_pass=int((valid.thermal_violations==0).sum()),voltage_pass=int((valid.voltage_violations==0).sum()),Q_pass=int((valid.max_q_limit_violation_mvar<=1e-6).sum()),max_loading_percent=float(valid.maximum_loading_percent.max()),J_AC_sum=float(valid.ac_apparent_normalized_thermal_exceedance.sum()),min_vm_pu=float(valid.min_vm_pu.min()),max_vm_pu=float(valid.max_vm_pu.max()),max_Q_excess_mvar=float(valid.max_q_limit_violation_mvar.max())))
 csvout(folder/'MATCHED_AC_SUMMARY.csv',pd.DataFrame(summaries))
 ps=[]
 for key,g in pair.groupby(['mapping','left','right']):
  r=dict(mapping=key[0],left=key[1],right=key[2],pairs=len(g),nonconvergent_pairs=int((~g.both_converged).sum()))
  for metric in ('ac_apparent_normalized_thermal_exceedance','maximum_loading_percent','maximum_voltage_limit_deviation_pu','max_q_limit_violation_mvar'):
   for winner in ('A','B','TIE'):r[metric+'_'+winner]=int((g[metric+'_winner']==winner).sum())
   r[metric+'_sum_difference']=float(g[metric+'_A_minus_B'].sum())
  for cls in ('BOTH','ONLY_A','ONLY_B','NEITHER'):r['feasibility_'+cls]=int((g.feasibility_class==cls).sum())
  ps.append(r)
 csvout(folder/'MATCHED_AC_PAIRWISE_SUMMARY.csv',pd.DataFrame(ps))
 # Mapping comparisons are separately qualified, never pooled with policy comparisons.
 key=['year','time_index','state_id','pf_case','policy'];hist=cases[cases.mapping=='HISTORICAL_TWO_MAP'].set_index(key);common=cases[cases.mapping=='COMMON_PQ_V2'].set_index(key);sensitivity=[]
 for idx,A in common.iterrows():
  B=hist.loc[idx];r=dict(zip(key,idx));r.update(both_converged=bool(A.converged and B.converged),common_feasible=bool(A.ac_feasible_under_declared_checks),historical_feasible=bool(B.ac_feasible_under_declared_checks))
  for metric in ('ac_apparent_normalized_thermal_exceedance','maximum_loading_percent','maximum_voltage_limit_deviation_pu','max_q_limit_violation_mvar','min_vm_pu','max_vm_pu'):
   r[metric+'_common_minus_historical']=float(A[metric]-B[metric]) if r['both_converged'] else None
  sensitivity.append(r)
 sens=pd.DataFrame(sensitivity);csvout(folder/'AC_MAPPING_SENSITIVITY.csv',sens)
 unity=sens[sens.pf_case=='unity'];numeric=[c for c in sens if c.endswith('_common_minus_historical')]
 assert unity.both_converged.all() and np.max(np.abs(unity[numeric].to_numpy()))<1e-8
 # Full historical rolling-M2 reproduction under the unchanged evaluator.
 old=pd.read_csv(consume(RESULTS/'selected_ac/CASES.csv'));checks=[]
 for _,r in cases[(cases.mapping=='HISTORICAL_TWO_MAP')&(cases.policy=='ROLLING_M2')].iterrows():
  hit=old[(old.year==r.year)&(old.time_index==r.time_index)&(old.state_id==r.state_id)&(old.pf_case==r.pf_case)&(old.policy_state=='RHO15_ROLLING_M2')];assert len(hit)==1;b=hit.iloc[0]
  for col,tol in [('min_vm_pu',1e-8),('max_vm_pu',1e-8),('maximum_loading_percent',1e-5),('ac_apparent_normalized_thermal_exceedance',1e-6),('max_q_limit_violation_mvar',1e-4),('slack_p_mw',1e-4)]:
   d=float(r[col]-b[col]);assert abs(d)<=tol;checks.append(dict(year=int(r.year),time_index=int(r.time_index),state_id=r.state_id,pf_case=r.pf_case,metric=col,difference=d))
 csvout(folder/'FULL_HISTORICAL_M2_RECONSTRUCTION.csv',pd.DataFrame(checks))
 jsout(target,dict(status='PASS',common_mapping_cases=648,historical_mapping_cases=648,AC_solves_in_campaign=1296,preflight_AC_solves=27,total_scheduled_new_AC_solves=1323,converged_cases=int(cases.converged.sum()),nonconverged_cases=int((~cases.converged).sum()),matched_pairs=len(pair),mapping_pairs=len(sens),unity_mapping_max_metric_difference=float(np.abs(unity[numeric]).to_numpy().max()),full_historical_M2_records_checked=162,deadbands={'normalized_AC_overload':1e-4,'peak_loading_percentage_points':1e-4,'voltage_deviation_pu':1e-6,'Q_limit_Mvar':1e-6},elapsed_s=time.monotonic()-started,configuration_sha256=configsha,claim='Selected deterministic stress cases; convergence is not thermal feasibility, dispatch implementability or annual AC security'))

if __name__=='__main__':run()
