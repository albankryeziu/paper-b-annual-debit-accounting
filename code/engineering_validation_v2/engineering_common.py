"""Isolated read-only evaluation. No protected project imports or optimizer calls."""
import os,sys
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json,hashlib,tempfile,csv
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
PROJECT=Path('C:/Users/Alban/OneDrive - University of Groningen/Desktop/Final Paper Script')
ARCH=PROJECT/'PAPER B/TPWRS_FINAL_ARCHIVE'
SOURCE=ARCH/'IMPLEMENTATION/source'
FROZEN=ARCH/'IMPLEMENTATION/frozen_inputs'
REAL=FROZEN/'artifacts/real_case'
RESULTS=ARCH/'RESULTS'
LOCK=Path('C:/Users/Alban/Documents/TPWRS_PaperB_Final_Signoff_20261017')
INVENTORY=OUT/'07_QC_AND_INTEGRITY/CONSUMED_INPUTS.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def atomic(p,data):
 p=Path(p);p.resolve().relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise FileExistsError('Refusing to overwrite completed artifact: '+str(p))
 fd,t=tempfile.mkstemp(dir=p.parent,suffix='.tmp')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data if isinstance(data,bytes) else data.encode('utf-8'));f.flush();os.fsync(f.fileno())
  os.rename(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)

def jsout(p,d):atomic(p,json.dumps(d,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def csvout(p,df):atomic(p,df.to_csv(index=False))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def consume(p,expected=None):
 p=Path(p).resolve();digest=sha(p)
 if expected is not None and digest!=expected:raise ValueError('Frozen hash mismatch: '+str(p))
 # Mutable inventory is a journal, not a scientific result; atomic replace only here.
 inv=read(INVENTORY) if INVENTORY.exists() else {}
 if str(p) in inv and inv[str(p)]['sha256']!=digest:raise ValueError('Consumed source changed: '+str(p))
 inv[str(p)]={'sha256':digest,'bytes':p.stat().st_size}
 tmp=INVENTORY.with_suffix('.tmp');tmp.write_text(json.dumps(inv,indent=2),encoding='utf-8');os.replace(tmp,INVENTORY)
 return p

def load_case():
 folder=REAL/'physical_case/FINAL_DUTCH_CASE'
 p=consume(folder/'physical_case.json');meta=read(p)
 z=np.load(consume(folder/meta['arrays_file'],meta['arrays_sha256']),allow_pickle=False)
 keys=['capacities_mw','requests_mw','p_background_mw','A','xbar_mw','eligibility_mask','contract_mask']
 a={k:z[k] for k in keys}
 a['H']=np.vstack([z['H_'+str(s['index'])] for s in meta['network_states']])
 a['h']=np.concatenate([z['h_'+str(s['index'])] for s in meta['network_states']])
 a['F']=np.concatenate([z['rating_'+str(s['index'])] for s in meta['network_states']])
 a['G']=a['H']@a['A'];a['dates']=pd.to_datetime(meta['time_grid']['timestamps_utc'],utc=True)
 a['rows']=[(s['state_id'],r) for s in meta['network_states'] for r in s['row_ids']]
 z.close();return meta,a

def policies(year):
 d={}
 for name in ('M1_LP','M2'):
  f=RESULTS/f'annual_primary/year_{year}/rho_0.15/annual/{name}'
  d['ANNUAL_'+name]=(f/'solution_arrays.npz','ANNUAL_RELAXATION' if name=='M1_LP' else 'ANNUAL_SELECTED')
 for name in ('M1','M2'):d['ROLLING_'+name]=(RESULTS/f'rolling/m1_m2/year_{year}/rho_0.15/rolling/{name}/rolling_arrays.npz','RAW_ROLLING')
 for name,label in [('UNIFORM_PREMIUM','U'),('CONSERVATIVE_STATE','C')]:
  d[label]=(RESULTS/f'debit_rule_family/year_{year}/{name}/RAW_ROLLING_POLICY.npz','RAW_ROLLING')
 ec=RESULTS/f'debit_rule_family/year_{year}/EXPOSURE_CONTROLLED_ANNUAL_WITNESS.npz'
 if not ec.exists():ec=RESULTS/f'debit_rule_family/final_evidence/year_{year}/EXPOSURE_CONTROLLED_ANNUAL_WITNESS.npz'
 d['EC']=(ec,'RETROSPECTIVE_ENERGY_CONTROLLED_WITNESS')
 return d

def action(path):
 with np.load(consume(path),allow_pickle=False) as z:return z['x_mw']

def load_ac(meta):
 import pandapower as pp
 p=consume(REAL/'LOCAL_SOURCE_CONFIG.json');cfg=read(p)['records']['reduced_network']
 network=consume(cfg['path'],cfg['sha256']);net=pp.from_pickle(str(network))
 m=read(consume(FROZEN/'artifacts/ac_inputs/AC_INPUT_FREEZE.json'))
 assert m['network_sha256']==sha(network) and m['physical_case_hash']==meta['physical_case_hash']
 assert m['bus_ids']==meta['bus_ids'] and m['participant_ids']==meta['participant_ids']
 with np.load(consume(FROZEN/'artifacts/ac_inputs/ac_inputs.npz',m['arrays_sha256']),allow_pickle=False) as z:a={k:z[k] for k in z.files}
 return net,m,a

def build_ac(net,state_id,bus_ids,p,q,gen_ids,pg):
 """Independent transcription of frozen construction, preserving devices and controls."""
 from copy import deepcopy
 import pandapower as pp
 test=deepcopy(net)
 assert set(map(str,test.bus.index))==set(bus_ids)
 for name in ('ward','xward','dcline','trafo3w','ssc','svc','tcsc'):
  t=getattr(test,name,None)
  if t is not None and len(t) and ('in_service' not in t or t.in_service.any()):raise ValueError('Unsupported active device '+name)
 for name in ('load','sgen','storage'):
  if len(test[name]):test[name].loc[:,'in_service']=False
 assert len(test.ext_grid.index[test.ext_grid.in_service.astype(bool)])==1
 if state_id!='INTACT':
  assert state_id.startswith('OUT::');kind,idx=state_id[5:].split('::');assert kind in ('line','trafo')
  assert bool(test[kind].at[int(idx),'in_service']);test[kind].at[int(idx),'in_service']=False
 fixed=np.array(p,float,copy=True);assert abs(fixed.sum())<1e-7
 assert tuple(gen_ids)==tuple(map(str,test.gen.index[test.gen.in_service.astype(bool)]))
 bp={b:i for i,b in enumerate(bus_ids)}
 for idx,v in zip(gen_ids,pg):
  test.gen.at[int(idx),'p_mw']=float(v);fixed[bp[str(int(test.gen.at[int(idx),'bus']))]]-=v
 for i,b in enumerate(bus_ids):pp.create_sgen(test,int(b),p_mw=float(fixed[i]),q_mvar=float(q[i]),name='FROZEN_NET_PQ')
 return test

def q_delta(connection,v,x,directions,pf):
 if pf=='unity':return np.zeros(connection.shape[0])
 assert pf in ('lagging_098','leading_098')
 sign=np.array([-1 if d=='import' else 1 for d in directions])
 return connection@(sign*(v-x)*np.tan(np.arccos(.98))*(1 if pf=='lagging_098' else -1))

def solve_ac(net):
 import pandapower as pp
 attempts=[]
 for init in ('dc','flat'):
  try:
   pp.runpp(net,algorithm='nr',init=init,calculate_voltage_angles=True,enforce_q_lims=True,check_connectivity=True,max_iteration=100,tolerance_mva=1e-8)
   if bool(net.converged):return True,attempts
   attempts.append({'init':init,'error':'nonconverged'})
  except Exception as e:attempts.append({'init':init,'error':type(e).__name__+': '+str(e)})
 return False,attempts

def state_flows(net,state):
 return np.array([net['res_'+rid.split('::')[0]].at[int(rid.split('::')[1]),'p_from_mw' if rid.startswith('line') else 'p_hv_mw'] for rid in state['row_ids']])

def connection_matrices(meta,a,q):
 mapping=read(consume(REAL/'physical_case/FINAL_DUTCH_CASE/PARTICIPANT_MAPPING.json'))
 by={r['participant_id']:r for r in mapping['records']};C=np.zeros((45,114));bp={b:i for i,b in enumerate(meta['bus_ids'])}
 for j,pid in enumerate(meta['participant_ids']):
  r=by[pid];assert r['direction']==meta['participant_directions'][j]
  for bus,w in zip(r['support_bus_ids'],r['weights']):C[bp[str(bus)],j]+=w
 with np.load(consume(REAL/'reduction/REDUCTION_EFFECTS.npz'),allow_pickle=False) as z:pi=z['balancing_key']
 sign=np.array([1 if d=='import' else -1 for d in meta['participant_directions']])
 assert np.max(np.abs(a['A']-sign*(C-pi[:,None])))<1e-12
 assert np.min(C)>=0 and np.allclose(C.sum(axis=0),1,atol=1e-12,rtol=0)
 assert np.min(q['participant_q_connection'])>=0 and np.allclose(q['participant_q_connection'].sum(axis=0),1,atol=1e-12,rtol=0)
 return C,pi

def require_preflight():
 status=read(OUT/'00_AUTHORITY_AND_INPUTS/PREFLIGHT_QC.json')
 if status['status']!='PASS':raise RuntimeError('Mandatory PREFLIGHT is not PASS')
 return status
