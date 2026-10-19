"""Complete signed-row replay of fixed actions. No optimization or policy repair."""
from engineering_common import *
import math,io

def run():
 require_preflight();folder=OUT/'01_PHYSICAL_METRICS';target=folder/'PHYSICAL_METRICS_VALIDATION.json'
 if target.exists():assert read(target)['status']=='PASS';return
 meta,a=load_case();dates=a['dates'];prior=read(consume(LOCK/'VERIFICATION_REPORTS/FRESH_REPLAY/INDEPENDENT_NUMERICAL_EVIDENCE.json'))['replays']
 intact=np.array([s=='INTACT' for s,r in a['rows']]);records=[];hours=[];crit=[];duration=[]
 for year in (2029,2030,2031):
  ix=np.flatnonzero(dates.year==year);assert len(ix)==8760
  for policy,(p,kind) in policies(year).items():
   checkpoint=folder/'checkpoints'/f'{year}_{policy}.json';hourfile=folder/'checkpoints'/f'{year}_{policy}.parquet'
   if checkpoint.exists():
    saved=read(checkpoint);assert saved['source_sha256']==sha(p) and saved['hourly_sha256']==sha(hourfile)
    records.append(saved['metrics']);hours.append(pd.read_parquet(hourfile));crit+=saved['critical'];duration+=saved['duration'];continue
   x=action(p);old=[r for r in prior if r['source']==str(p)];assert len(old)==1 and old[0]['sha256']==sha(p)
   assert x.shape==(8760,114) and np.isfinite(x).all()
   rowpeak=np.zeros(len(a['F']));rowtime=np.zeros(len(a['F']),int);rowflow=np.zeros(len(a['F']));distinct={e:0. for s,e in a['rows']}
   vm=[];im=[];cm=[];sums=[];worst=[]
   for start in range(0,8760,128):
    ids=ix[start:start+128];xb=x[start:start+len(ids)]
    req=a['p_background_mw'][ids]-a['requests_mw'][ids]@a['A'].T
    flow=(req+xb@a['A'].T)@a['H'].T+a['h'];over=np.maximum(np.abs(flow)-a['F'],0.)
    sums.append(float(over.sum()));vm.extend(over.max(axis=1));im.extend(over[:,intact].max(axis=1));cm.extend(over[:,~intact].max(axis=1));worst.extend(over.argmax(axis=1))
    loc=over.argmax(axis=0);val=over[loc,np.arange(len(rowpeak))];mask=val>rowpeak
    rowpeak[mask]=val[mask];rowtime[mask]=start+loc[mask];rowflow[mask]=flow[loc[mask],np.flatnonzero(mask)]
   V=np.array(vm);I=np.array(im);C=np.array(cm);S=math.fsum(sums);diff=S-old[0]['S'];assert abs(diff)<.01
   for j,(s,e) in enumerate(a['rows']):distinct[e]=max(distinct[e],float(rowpeak[j]))
   peakrow=int(rowpeak.argmax());peakt=int(rowtime[peakrow]);state,element=a['rows'][peakrow]
   r=dict(year=year,policy=policy,object=kind,source=str(p),source_sha256=sha(p),S_mwh_row_state=S,S_archived=old[0]['S'],S_difference=diff,
    curtailed_energy_MWh=float(x.sum()),FDE_hours=float((x/a['capacities_mw']).sum()),Vmax_MW=float(V.max()),mean_hourly_max_MW=float(V.mean()),p95_hourly_max_MW=float(np.percentile(V,95)),p99_hourly_max_MW=float(np.percentile(V,99)),
    max_intact_exceedance_MW=float(I.max()),max_retained_contingency_exceedance_MW=float(C.max()),worst_timestamp_utc=str(dates[ix[peakt]]),worst_state_id=state,worst_element_id=element,worst_flow_MW=float(rowflow[peakrow]),worst_rating_MW=float(a['F'][peakrow]))
   for eps,label in [(1e-4,'numerical'),(1.,'engineering')]:
    r.update({f'overload_hours_{label}':int((V>eps).sum()),f'intact_violation_hours_{label}':int((I>eps).sum()),f'contingency_violation_hours_{label}':int((C>eps).sum()),f'distinct_overloaded_physical_elements_{label}':sum(v>eps for v in distinct.values())})
   frame=pd.DataFrame(dict(year=year,policy=policy,object=kind,timestamp_utc=dates[ix],V_MW=V,intact_max_MW=I,retained_contingency_max_MW=C,worst_flat_row=np.array(worst,int)))
   cr=[]
   for j in np.argsort(-rowpeak,kind='stable')[:25]:
    s,e=a['rows'][j];cr.append(dict(year=year,policy=policy,object=kind,flat_row=int(j),state_id=s,physical_element_id=e,peak_exceedance_MW=float(rowpeak[j]),timestamp_utc=str(dates[ix[rowtime[j]]]),flow_MW=float(rowflow[j]),rating_MW=float(a['F'][j])))
   dr=[dict(year=year,policy=policy,threshold_MW=eps,hours=int((V>eps).sum()),fraction_hours=float((V>eps).mean())) for eps in (1e-4,1,10,100,500,1000)]
   buf=io.BytesIO();frame.to_parquet(buf,index=False);atomic(hourfile,buf.getvalue())
   jsout(checkpoint,dict(source_sha256=sha(p),hourly_sha256=sha(hourfile),metrics=r,critical=cr,duration=dr))
   records.append(r);hours.append(frame);crit+=cr;duration+=dr
   print('PHYSICAL COMPLETE',year,policy,'S',S,'Vmax',V.max(),'N1MW',r['overload_hours_engineering'],flush=True)
 csvout(folder/'ANNUAL_PHYSICAL_METRICS.csv',pd.DataFrame(records));buf=io.BytesIO();pd.concat(hours,ignore_index=True).to_parquet(buf,index=False);atomic(folder/'HOURLY_MAXIMUM_EXCEEDANCE.parquet',buf.getvalue())
 csvout(folder/'CRITICAL_ROWS_BY_POLICY.csv',pd.DataFrame(crit));csvout(folder/'OVERLOAD_DURATION_SUMMARY.csv',pd.DataFrame(duration))
 jsout(target,dict(status='PASS',objects=len(records),hourly_records=sum(len(f) for f in hours),hours_per_object=8760,rows_per_hour=8837,thresholds_MW=[1e-4,1.],quantiles='numpy percentile linear',maximum_objective_discrepancy=max(abs(r['S_difference']) for r in records),objective_units='MW h row-state; not unserved energy or expected risk',peak_scope='maximum over retained contingency states, not simultaneous overload',optimization_calls=0))

if __name__=='__main__':run()
