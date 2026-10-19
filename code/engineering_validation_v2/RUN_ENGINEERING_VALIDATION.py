"""No-argument PyCharm runner for isolated v2; never imports frozen orchestration."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from engineering_common import *

STAGE = "PREFLIGHT"
STAGES=('PREFLIGHT','PHYSICAL_METRICS','AC_REPLAY','HEADROOM','INFORMATION_AUDIT','REPORTS','ALL')

def main():
 if STAGE not in STAGES:raise ValueError('Unknown stage '+STAGE)
 if INVENTORY.exists():
  for p,item in read(INVENTORY).items():
   if sha(p)!=item['sha256']:raise RuntimeError('Consumed input changed: '+p)
 from stage_preflight import run as preflight
 from stage_physical import run as physical
 from stage_ac import run as ac
 from stage_headroom import run as headroom
 from stage_information import run as information
 def reports():
  from build_dossier import run
  run()
 functions=dict(PREFLIGHT=preflight,PHYSICAL_METRICS=physical,AC_REPLAY=ac,HEADROOM=headroom,INFORMATION_AUDIT=information,REPORTS=reports)
 requested=list(functions) if STAGE=='ALL' else [STAGE]
 for stage in requested:
  if stage!='PREFLIGHT':require_preflight()
  print('STAGE',stage,flush=True);functions[stage]()
 print('Requested stages complete; completed outputs retained.',flush=True)

if __name__=='__main__':main()
