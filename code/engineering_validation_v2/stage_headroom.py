"""Fixed counter-injection diagnostic, not a redispatch reconstruction or optimization."""
from engineering_common import *

def run():
 require_preflight();folder=OUT/'03_GENERATOR_HEADROOM';target=folder/'HEADROOM_QC.json'
 if target.exists():return
 meta,a=load_case();net,m,q=load_ac(meta);Cp,pi=connection_matrices(meta,a,q)
 registry=read(consume(REAL/'background/DEVICE_REGISTRY.json'));inventory=[];identifiability=[]
 consume(PROJECT/'PAPER B/Final/artifacts/real_case/background/DEVICE_REGISTRY.json',sha(REAL/'background/DEVICE_REGISTRY.json'))
 for kind in ('gen','ext_grid'):
  for idx,r in net[kind].iterrows():
   inventory.append(dict(device_kind=kind,device_id=str(idx),bus_id=str(int(r.bus)),name=str(r['name']),type=str(r.get('type','REFERENCE_GRID')),in_service=bool(r.in_service),
    model_Pmin_MW=float(r.min_p_mw),model_Pmax_MW=float(r.max_p_mw),model_Qmin_Mvar=float(r.min_q_mvar),model_Qmax_Mvar=float(r.max_q_mvar),
    installed_capacity_source='NOT_SEPARATELY_IDENTIFIED; native P bounds are model limits',available_reserve='NOT_IDENTIFIABLE',ramp_limits='NOT_IDENTIFIABLE_IN_FROZEN_MODEL',complete_Q_capability_curve=False))
   identifiability.append(dict(device_kind=kind,device_id=str(idx),bus_id=str(int(r.bus)),policy_specific_adjustment_status='NOT IDENTIFIABLE FROM AVAILABLE DATA',reason='A assigns balanced bus injections; no generator-specific participation factors identify Delta P_g. PV active-power dispatch is held fixed by the AC evaluator.'))
 csvout(folder/'GENERATOR_CAPACITY_INVENTORY.csv',pd.DataFrame(inventory));csvout(folder/'GENERATOR_HEADROOM_AUDIT.csv',pd.DataFrame(identifiability))
 maprows=[]
 for j,pid in enumerate(meta['participant_ids']):
  sign=1 if meta['participant_directions'][j]=='import' else -1
  for b,bid in enumerate(meta['bus_ids']):
   if Cp[b,j]!=0 or pi[b]!=0:maprows.append(dict(participant_id=pid,direction=meta['participant_directions'][j],bus_id=bid,participant_connection_share=float(Cp[b,j]),balancing_share=float(pi[b]),restriction_delta_participant_P_per_MW=float(sign*Cp[b,j]),restriction_delta_counter_P_per_MW=float(-sign*pi[b]),A_per_MW=float(a['A'][b,j]),generator_allocation='NOT_IDENTIFIED'))
 csvout(folder/'COUNTER_INJECTION_MAPPING.csv',pd.DataFrame(maprows))
 # Recover background device dispatch, keeping generation, renewable curtailment
 # and interchange separate. No net bus injection is substituted into a P bound.
 results=[];busr=[];backgroundqc=[];refqc=[]
 for year in (2029,2030,2031):
  ix=np.flatnonzero(a['dates'].year==year)
  cache=PROJECT/f'PAPER B/Final/artifacts/real_case/background/background_{year}.npz'
  audit=read(consume(cache.with_name(f'BACKGROUND_AUDIT_{year}.json')))
  with np.load(consume(cache,audit['cache_sha256']),allow_pickle=False) as z:
   pg=z['device_dispatch_mw'];bg=z['p_background_mw'];times=pd.to_datetime(z['timestamps_utc'],utc=True)
  assert times.equals(a['dates'][ix]) and np.array_equal(bg,a['p_background_mw'][ix]) and pg.shape[1]==len(registry)
  base=np.zeros((8760,45));lo=np.zeros(45);hi=np.zeros(45);ids={};renewable=[];boundary=[]
  for j,r in enumerate(registry):
   if r['kind']=='INTERNAL_GENERATOR':
    idx=int(r['device_id'].split('::')[1]);nr=net.gen.loc[idx];b=meta['bus_ids'].index(str(int(nr.bus)))
    base[:,b]+=pg[:,j];lo[b]+=float(nr.min_p_mw);hi[b]+=float(nr.max_p_mw);ids.setdefault(b,[]).append(r['device_id'])
   elif r['kind']=='FIXED_SIGNED_NET_INTERCHANGE_GROUP':
    for share,member in zip(r['spatial_weights'],r['members']):
     nr=net[member['type']].loc[member['index']];b=meta['bus_ids'].index(str(member['bus']));base[:,b]+=pg[:,j]*share
     lo[b]+=float(nr.min_p_mw);hi[b]+=float(nr.max_p_mw);ids.setdefault(b,[]).append(member['type']+'::'+str(member['index']))
    boundary.append(r['device_id'])
   elif r['kind']=='RENEWABLE_DOWNWARD_FLEXIBILITY':renewable.append(j)
  baseline_excess=np.maximum(base-hi,0)+np.maximum(lo-base,0)
  backgroundqc.append(dict(year=year,background_cache_hash=sha(cache),background_identity_exact=True,device_count=len(registry),renewable_curtailment_devices=len(renewable),fixed_interchange_groups=len(boundary),max_background_aggregate_P_violation_MW=float(baseline_excess.max())))
  # Frozen AC PV references, distinct from security-adjusted background devices.
  for j,gid in enumerate(m['generator_ids']):
   r=net.gen.loc[int(gid)];v=q['generator_p_mw'][ix,j]
   refqc.append(dict(year=year,generator_id=gid,minimum_reference_P_MW=float(v.min()),maximum_reference_P_MW=float(v.max()),max_bound_excess_MW=float(np.maximum(v-r.max_p_mw,r.min_p_mw-v).max(initial=0)),scope='AC_REFERENCE_PV_P_FIXED_ACROSS_POLICIES'))
  sigma=np.array([1 if d=='import' else -1 for d in meta['participant_directions']])
  for policy,(path,kind) in policies(year).items():
   x=action(path)
   # Total balancing component relative to security-adjusted background:
   # p=background - C_P sigma(v-x) + pi sum sigma(v-x).
   k=(a['requests_mw'][ix]-x)@sigma;bal=k[:,None]*pi
   rebuilt=a['p_background_mw'][ix]-(sigma*(a['requests_mw'][ix]-x))@Cp.T+bal
   exact=a['p_background_mw'][ix]-a['requests_mw'][ix]@a['A'].T+x@a['A'].T
   assert np.max(np.abs(rebuilt-exact))<1e-8
   candidate=base+bal;up=np.maximum(candidate-hi,0);down=np.maximum(lo-candidate,0);viol=up+down
   # Conditional necessary condition only: all fixed balancing assigned to
   # existing generator/interchange fleet, keeping other resource components fixed.
   r=dict(year=year,policy=policy,object=kind,audit_level='CONDITIONAL_BUS_AGGREGATE_NECESSARY_CONDITION',generator_specific_allocation='NOT IDENTIFIABLE FROM AVAILABLE DATA',
    normal_operation_hours=8760,affected_hours=int(np.any(viol>1e-6,axis=1).sum()),affected_buses=int(np.any(viol>1e-6,axis=0).sum()),affected_generators='NOT_IDENTIFIABLE',maximum_upward_excess_MW=float(up.max()),maximum_downward_excess_MW=float(down.max()),sum_upward_excess_MWh=float(up.sum()),sum_downward_excess_MWh=float(down.sum()),
    maximum_absolute_balancing_MW=float(np.abs(bal).max()),maximum_total_balancing_MW=float(k.max()),minimum_total_balancing_MW=float(k.min()),
    interpretation='Conditional fleet-only static bound; failure is not proof against other load/renewable/interchange allocations. Passing does not prove generator allocation, reserves or ramps.',contingency_scope='Same scheduled P in every retained electrical state; no generator outages or remedial redispatch modeled')
   results.append(r)
   for b in np.flatnonzero(pi>0):
    busr.append(dict(year=year,policy=policy,bus_id=meta['bus_ids'][b],balancing_share=float(pi[b]),model_generation_lower_MW=float(lo[b]),model_generation_upper_MW=float(hi[b]),source_devices=json.dumps(ids.get(int(b),[])),hours_outside_aggregate_bounds=int((viol[:,b]>1e-6).sum()),maximum_upward_excess_MW=float(up[:,b].max()),maximum_downward_excess_MW=float(down[:,b].max()),sum_excess_MWh=float(viol[:,b].sum()),scope='CONDITIONAL_FLEET_ONLY_NECESSARY_CONDITION'))
   print('HEADROOM',year,policy,'conditional affected hours',r['affected_hours'],flush=True)
 csvout(folder/'BALANCING_FEASIBILITY_BY_POLICY.csv',pd.DataFrame(results));csvout(folder/'BUS_AGGREGATE_NECESSARY_CONDITIONS.csv',pd.DataFrame(busr));csvout(folder/'AC_REFERENCE_GENERATOR_BOUND_CHECK.csv',pd.DataFrame(refqc));csvout(folder/'BACKGROUND_DEVICE_QC.csv',pd.DataFrame(backgroundqc))
 atomic(folder/'HEADROOM_DATA_LIMITATIONS.md',r'''# Generator headroom: identification limits

The final A identifies participant and balanced counter-injection bus components, not generator-specific adjustments. Its exact decomposition is A[:,n]=sigma_n(C_P[:,n]-pi). Thus p(x)=p_background-C_P diag(sigma)(v-x)+pi sum_n sigma_n(v_n-x_n). No net bus injection is treated as generation.

Native generator and interchange Pmin/Pmax and Qmin/Qmax bounds, source buses, reference PV dispatch and security-adjusted background device dispatch are recoverable. The AC evaluator fixes PV P and places all remaining net P in synthetic PQ devices; slack P covers losses. It does not impose generator active-power feasibility on those synthetic injections.

Generator-specific participation factors for A are NOT IDENTIFIABLE FROM AVAILABLE DATA. Policy-dependent generator headroom violations, counts of affected generators, reserve availability, dispatch availability and ramp feasibility therefore cannot be asserted. Native Pmax is a model bound, not independently verified installed/available capacity. Simple Q bounds are not complete Q-V capability curves.

The bus-aggregate table is an explicitly conditional static necessary-condition audit. It separates recovered background generator/boundary dispatch from load/renewable curtailment, then asks whether assigning the entire balanced counter-injection to that same native generator/interchange fleet would exceed aggregate bounds, holding all other components fixed. It does not assign individual devices, change generation or repair policies. It is neither a proof of general physical infeasibility nor a proof of deployability; alternative load/renewable contributions are outside that conditional test. Background boundary totals are frozen in the DC builder, so allowing their native device ranges in this diagnostic is already a relaxation of that construction.

All scheduled policy injections remain the same across retained line/transformer contingencies. No post-contingency generator redispatch, generator outage, frequency control, reserve or ramp model is introduced. Full-year bounds are evaluated on the one-hour grid only. No ramp values are inferred from capacity.
''')
 jsout(target,dict(status='PASS_DIAGNOSTIC_WITH_IDENTIFICATION_LIMIT',generator_allocation='NOT IDENTIFIABLE FROM AVAILABLE DATA',policy_year_objects=len(results),capacity_inventory_records=len(inventory),background_dispatch_recovered=True,AC_reference_bound_records=len(refqc),conditional_violating_objects=sum(r['affected_hours']>0 for r in results),ramping='NOT_IDENTIFIABLE',generator_specific_violation_counts='NOT_IDENTIFIABLE',optimization_calls=0))

if __name__=='__main__':run()
