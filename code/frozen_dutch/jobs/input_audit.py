from __future__ import annotations


# ============================================================================
# scripts.audit_local_physical_case (consolidated; see SOURCE_MAP.json)
# ============================================================================
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tpwrs.config import existing_workspace, source_records, source_path, scientific_fingerprint
from tpwrs.persistence import load_physical_case
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.storage import atomic_write_json
from tpwrs.execution import utc_now

def audit_case(directory,strict=True):
    directory=Path(directory)
    case=load_physical_case(directory)
    observed={'buses':case.n_buses,'participants':case.n_participants,'hours':case.n_time,
              'states':len(case.network_states),'flat_rows':case.n_flat_rows}
    expected={'buses':45,'participants':case.n_participants,'hours':61344,'states':94,'flat_rows':8837}
    if strict and (case.n_participants not in (79,114) or observed!=expected): raise ValueError(f'Frozen primary inventory differs: {observed}; expected {expected}')
    if strict and (set(case.time_grid.index.year)!=set(range(2025,2032)) or case.time_grid.uniform_delta!=1.0):
        raise ValueError('The primary contract is complete hourly UTC 2025-2031.')
    freeze=directory/'PHYSICAL_CASE_FREEZE.json'
    if freeze.is_file():
        d=json.loads(freeze.read_text())
        if d.get('status')!='PASS' or d.get('physical_case_hash')!=case.physical_case_hash:
            raise ValueError('Physical freeze status/hash mismatch')
        if strict:
            recorded=d.get('case_metadata')
            actual=json.loads((directory/'physical_case.json').read_text(encoding='utf-8'))
            if recorded is None or recorded!=actual:raise ValueError('Serialized time/ID/metadata record differs from the historical freeze.')
    elif strict: raise FileNotFoundError(freeze)
    balance=float(np.max(np.abs(case.A.sum(axis=0)),initial=0.0))
    bg=float(np.max(np.abs(case.p_background_mw.sum(axis=1)),initial=0.0))
    hg=float(np.max(np.abs(case.flat_G-case.flat_H@case.A),initial=0.0))
    if max(balance,bg,hg)>1e-7: raise ValueError(f'Conservation/matrix identity failed: {balance,bg,hg}')
    axes={'bus_ids':case.bus_ids,'participant_ids':case.participant_ids,
          'directions':case.participant_directions,'rows':case.flat_row_ids,'states':case.flat_state_ids}
    return case,{'status':'PASS',**observed,'physical_case_hash':case.physical_case_hash,
        'axis_identity_hash':sha256_json(axes),'maximum_action_balance_mw':balance,
        'maximum_background_balance_mw':bg,'maximum_G_minus_HA':hg,
        'scope':'SERIALIZED INPUT INTEGRITY, CONSERVATION AND AXIS CHECKS; NOT AC FEASIBILITY'}

def reconcile_primary_inputs(root,case):
    """Rebuild the participant gate from sources, retaining compatible physical backgrounds.

    This is input construction before fitting; no M1--M3 outcome is read.
    The original input workspace is never modified.
    """
    from jobs.input_build import discover_sources,build_network_and_reduction,freeze_case
    from tpwrs.storage import sha256_array
    root=Path(root);marker=root/'PRIMARY_SPECIFICATION_RECONCILIATION.json'
    current=scientific_fingerprint()
    if marker.is_file():
        value=json.loads(marker.read_text(encoding='utf-8'))
        if value.get('scientific_fingerprint')==current and value.get('physical_case_hash')==case.physical_case_hash:
            return case,value
    old_hash=case.physical_case_hash
    paths=discover_sources();registry=json.loads((root/'SOURCE_REGISTRY.json').read_text(encoding='utf-8'))
    print('INPUT_RECONCILIATION: rebuilding source-only direct/multibus benchmark.',flush=True)
    net,states,_,reduction,participants=build_network_and_reduction(paths,registry,root)
    if tuple(reduction['bus_ids'])!=case.bus_ids or len(states)!=len(case.network_states):
        raise RuntimeError('Imported background axes differ from reconstructed network; full source reconstruction required.')
    # Cross-version BLAS changes the reconstructed PTDF at machine precision.
    # Retain the original frozen network, after a conservative absolute flow
    # error bound over the imported background box passes its existing 1e-7 MW
    # physical-identity tolerance. No physical arrays or background are edited.
    compatibility=[]
    envelope=np.max(np.abs(case.p_background_mw),axis=0)
    for old,new in zip(case.network_states,states):
        if old.row_ids!=new.row_ids or old.state_id!=new.state_id or not np.array_equal(old.ratings_mw,new.ratings_mw):
            raise RuntimeError('Imported network axes/ratings changed; full source reconstruction required.')
        bound=np.abs(old.H-new.H)@envelope+np.abs(old.h_mw-new.h_mw)
        maximum=float(np.max(bound,initial=0.))
        if maximum>1e-7:
            raise RuntimeError('Reconstructed network fails the frozen physical-identity tolerance.')
        compatibility.append({'state':old.state_id,'flow_error_bound_mw':maximum})
    states=case.network_states
    registry_path=root/'network/NETWORK_STATE_REGISTRY.json'
    network_registry=json.loads(registry_path.read_text(encoding='utf-8'))
    for row,state in zip(network_registry['states'],states):
        row.update(H_sha256=sha256_array(state.H),h_sha256=sha256_array(state.h_mw),
                   ratings_sha256=sha256_array(state.ratings_mw),state_source_hash=state.source_hash)
    network_registry['frozen_network_retention_audit']=compatibility
    atomic_write_json(registry_path,network_registry)
    network_audit_path=root/'network/NETWORK_AUDIT.json'
    network_audit=json.loads(network_audit_path.read_text(encoding='utf-8'))
    network_audit['state_hashes']={state.state_id:state.hash for state in states}
    atomic_write_json(network_audit_path,network_audit)
    freeze=freeze_case(paths,registry,states,reduction,participants,case.time_grid.index,case.p_background_mw,root)
    updated=load_physical_case(root/'physical_case/FINAL_DUTCH_CASE')
    value={'status':'PASS','scientific_fingerprint':current,'source_physical_case_hash':old_hash,
        'physical_case_hash':updated.physical_case_hash,'participants':updated.n_participants,
        'background_reused_sha256':sha256_array(case.p_background_mw),'gate':reduction['gate'],
        'network_retention_audit':compatibility,
        'old_workspace_modified':False,'policy_outcomes_read':False}
    atomic_write_json(marker,value)
    return updated,value


def main():
    argparse.ArgumentParser(description='Import verified backgrounds; reconstruct the source-only participant gate and freeze fitting inputs.').parse_args()
    dest=ROOT/'artifacts/real_case';original=existing_workspace()/'artifacts/real_case'
    record={'status':'NOT_READY','created_utc':utc_now(),'scientific_fingerprint':scientific_fingerprint()}
    try:
        if not (dest/'physical_case/FINAL_DUTCH_CASE/physical_case.json').is_file():
            if not original.is_dir(): raise FileNotFoundError(f'Frozen input root is missing: {original}. Set existing_workspace in LOCAL_SETTINGS.json.')
            if dest.exists() and any(dest.iterdir()): raise RuntimeError('Destination input tree is partial. It will not be overwritten. Preserve it and select a clean extraction.')
            # Only data artifacts, never old code/cards/policies/results.
            staging=ROOT/'artifacts/.real_case_importing'
            if staging.exists(): shutil.rmtree(staging)
            staging.parent.mkdir(parents=True,exist_ok=True)
            print(f'COPYING frozen input artifacts only: {original}',flush=True)
            shutil.copytree(original,staging,ignore=shutil.ignore_patterns('*.lock','*.tmp','__pycache__'))
            audit_case(staging/'physical_case/FINAL_DUTCH_CASE')
            if dest.exists(): dest.rmdir()
            os.replace(staging,dest)
        case,physical=audit_case(dest/'physical_case/FINAL_DUTCH_CASE')
        case,reconciliation=reconcile_primary_inputs(dest,case)
        case,physical=audit_case(dest/'physical_case/FINAL_DUTCH_CASE')
        from tpwrs.persistence import prepare_fitting_inputs
        fitting=prepare_fitting_inputs(case,dest/'physical_case/FINAL_DUTCH_CASE')
        registry=json.loads((dest/'SOURCE_REGISTRY.json').read_text())
        local=json.loads((dest/'LOCAL_SOURCE_CONFIG.json').read_text())
        if local.get('registry_hash')!=registry.get('registry_hash'): raise ValueError('Source registries disagree')
        expected_registry=sha256_json({k:v['sha256'] for k,v in sorted(registry['records'].items())})
        if expected_registry!=registry.get('registry_hash'):raise ValueError('Source registry content/hash mismatch')
        source_checks={};missing=[]
        for name,row in registry['records'].items():
            try:
                p=source_path(name)
                if sha256_file(p)!=row['sha256']: raise ValueError('content hash mismatch')
                source_checks[name]='PASS'
            except (OSError,ValueError) as exc:
                source_checks[name]=str(exc);missing.append(name)
        if missing: raise RuntimeError('Missing/changed raw source(s): '+', '.join(missing))
        probe=ROOT/'artifacts/.write_probe';probe.write_text('probe');probe.unlink()
        free=shutil.disk_usage(ROOT).free
        if free<20*1024**3: raise RuntimeError('Less than 20 GiB free disk; qualification not started.')
        record.update(status='PASS',physical_case=physical,source_checks=source_checks,
                      fitting_partition_hash=fitting['record_hash'],
                      primary_specification_reconciliation=reconciliation,
                      free_disk_gib=free/1024**3,source_registry_hash=registry.get('registry_hash'),
                      ac_inputs_present=(ROOT/'artifacts/ac_inputs/AC_INPUT_FREEZE.json').is_file(),
                      old_results_imported=False,physical_source_code_hashes='HISTORICAL PRODUCER PROVENANCE RETAINED, NOT REWRITTEN')
    except Exception as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
    atomic_write_json(ROOT/'artifacts/FINAL_PHYSICAL_READINESS.json',record)
    print(json.dumps(record,indent=2)); return 0 if record['status']=='PASS' else 1


# ============================================================================
# scripts.freeze_price_series (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Freeze an explicitly chosen complete hourly EUR/MWh series for ex-post reporting."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tpwrs.persistence import load_physical_case
from tpwrs.storage import atomic_write_json, atomic_write_csv
from tpwrs.storage import sha256_file

def price_main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'local/PRICE_INPUT.json');args=parser.parse_args()
    if not args.config.is_file():raise RuntimeError('Optional price input absent. Complete local/PRICE_INPUT.json from its example; no monetary numbers are fabricated.')
    cfg=json.loads(args.config.read_text());p=Path(cfg['path']);p=p if p.is_absolute() else args.config.parent/p
    if cfg.get('units')!='EUR/MWh' or cfg.get('kind') not in ('historical_observation','scenario_projection'):raise ValueError('Declare EUR/MWh units and historical_observation or scenario_projection.')
    if not cfg.get('provenance'):raise ValueError('Price provenance is required.')
    if cfg.get('sha256') and sha256_file(p)!=cfg['sha256']:raise ValueError('Price source changed.')
    f=pd.read_csv(p);f['timestamp_utc']=pd.to_datetime(f['timestamp_utc'],utc=True,errors='raise')
    col=cfg.get('value_column','price_eur_mwh');f['price_eur_mwh']=pd.to_numeric(f[col],errors='raise')
    if f.timestamp_utc.duplicated().any():raise ValueError('Duplicate price timestamps.')
    case=load_physical_case(ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE')
    f=f.set_index('timestamp_utc').reindex(case.time_grid.index)
    if not np.isfinite(f.price_eur_mwh.to_numpy(float)).all():raise ValueError('Missing/nonfinite hourly prices. Interpolation and zero filling are forbidden.')
    f=f[['price_eur_mwh']].reset_index();f.columns=['timestamp_utc','price_eur_mwh']
    out=ROOT/'artifacts/prices';out.mkdir(parents=True,exist_ok=True)
    meta={'schema':'TPWRS_EXPOST_PRICE_V1','status':'PASS','source_sha256':sha256_file(p),'source_path':str(p),'units':'EUR/MWh',
        'kind':cfg['kind'],'provenance':cfg['provenance'],'physical_case_hash':case.physical_case_hash,'time_grid_id':case.time_grid.grid_id,
        'policy_input':False,'monetary_interpretation':'Ex-post price-weighted restriction burden; not welfare or realized cost savings.'}
    old=out/'PRICE_FREEZE.json'
    if old.is_file():
        current=json.loads(old.read_text())
        if any(current.get(k)!=v for k,v in meta.items()):raise ValueError('A different price freeze exists; do not overwrite.')
        if sha256_file(out/'prices.csv')!=current['prices_sha256']:raise ValueError('Frozen prices changed.')
        print('PRICE_FREEZE=ALREADY_VALID');return 0
    atomic_write_csv(out/'prices.csv',f);meta['prices_sha256']=sha256_file(out/'prices.csv');atomic_write_json(old,meta);print(json.dumps(meta,indent=2));return 0
