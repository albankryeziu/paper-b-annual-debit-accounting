"""Native AC metrics on exact frozen monitored elements; failures retain null metrics."""
from engineering_common import *

def evaluate(net,state,identity):
 ok,attempts=solve_ac(net)
 record=dict(identity,converged=ok,attempts=json.dumps(attempts),failure_cause='' if ok else json.dumps(attempts),solver_iterations=None)
 loading=[];voltage=[];reactive=[]
 if not ok:
  record['ac_feasible_under_declared_checks']=False
  return record,loading,voltage,reactive
 record['solver_iterations']=int(net._ppc.get('iterations',0)) if net._ppc.get('iterations') is not None else None
 for rid in state['row_ids']:
  kind,ix=rid.split('::');ix=int(ix);r=net['res_'+kind].loc[ix];device=net[kind].loc[ix]
  d=dict(identity,row_id=rid,element_kind=kind,element_index=ix,loading_percent=float(r.loading_percent))
  if kind=='line':
   rated=float(device.max_i_ka*device.df*device.parallel)
   # i_ka is max end current in pandapower, not active MW / rating.
   value=float(r.i_ka)
   d.update(native_unit='kA',native_rating=rated,native_value=value,native_exceedance=max(value-rated,0.),p_from_mw=float(r.p_from_mw),q_from_mvar=float(r.q_from_mvar))
  else:
   # Historical runpp default uses current loading. Report its native terminal currents.
   hv=float(device.sn_mva*device.df*device.parallel/(np.sqrt(3)*device.vn_hv_kv))
   lv=float(device.sn_mva*device.df*device.parallel/(np.sqrt(3)*device.vn_lv_kv))
   exhv=max(float(r.i_hv_ka)-hv,0.);exlv=max(float(r.i_lv_ka)-lv,0.)
   d.update(native_unit='kA',native_rating=hv if exhv>=exlv else lv,native_value=float(r.i_hv_ka) if exhv>=exlv else float(r.i_lv_ka),native_exceedance=max(exhv,exlv),hv_rating_ka=hv,lv_rating_ka=lv,hv_current_ka=float(r.i_hv_ka),lv_current_ka=float(r.i_lv_ka),p_from_mw=float(r.p_hv_mw),q_from_mvar=float(r.q_hv_mvar))
  loading.append(d)
 for ix,r in net.res_bus.iterrows():
  v=float(r.vm_pu);voltage.append(dict(identity,bus_id=str(ix),vm_pu=v,voltage_limit_deviation_pu=max(.95-v,v-1.05,0.)))
 for kind in ('gen','ext_grid'):
  for ix,row in net[kind].loc[net[kind].in_service.astype(bool)].iterrows():
   q=float(net['res_'+kind].at[ix,'q_mvar']);lo=float(row.min_q_mvar);hi=float(row.max_q_mvar)
   reactive.append(dict(identity,device_kind=kind,device_id=str(ix),bus_id=str(int(row.bus)),q_mvar=q,qmin_mvar=lo,qmax_mvar=hi,lower_deficit_mvar=max(lo-q,0.),upper_excess_mvar=max(q-hi,0.),limit_violation_mvar=max(lo-q,q-hi,0.)))
 vm=np.array([r['vm_pu'] for r in voltage]);loads=np.array([r['loading_percent'] for r in loading]);qv=np.array([r['limit_violation_mvar'] for r in reactive])
 assert np.isfinite(vm).all() and np.isfinite(loads).all() and np.isfinite(qv).all()
 line=[r['loading_percent'] for r in loading if r['element_kind']=='line'];trafo=[r['loading_percent'] for r in loading if r['element_kind']=='trafo']
 record.update(min_vm_pu=float(vm.min()),max_vm_pu=float(vm.max()),maximum_loading_percent=float(loads.max()),maximum_line_loading_percent=max(line),maximum_transformer_loading_percent=max(trafo),
  voltage_violations=int(((vm<.95)|(vm>1.05)).sum()),thermal_violations=int((loads>100).sum()),overloaded_lines=sum(v>100 for v in line),overloaded_transformers=sum(v>100 for v in trafo),
  maximum_voltage_limit_deviation_pu=max(r['voltage_limit_deviation_pu'] for r in voltage),max_q_limit_violation_mvar=float(qv.max()),generators_outside_q_bounds=int((qv>1e-6).sum()),
  maximum_native_current_exceedance_ka=max(r['native_exceedance'] for r in loading),ac_apparent_normalized_thermal_exceedance=float(np.maximum(loads/100-1,0).sum()),slack_p_mw=float(net.res_ext_grid.p_mw.sum()),
  ac_feasible_under_declared_checks=bool(np.all((vm>=.95)&(vm<=1.05)) and np.all(loads<=100+1e-6) and qv.max()<=1e-6),transformer_loading_method=net._options['trafo_loading'])
 return record,loading,voltage,reactive
