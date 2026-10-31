"""Authenticate existing required-bank admission without creating authority."""
import os
from pathlib import Path
import re
import sys
import time
from common import GUARD,COMMIT,config,sha
sys.path.insert(0,str(GUARD))
import gpu_registry
import model_cache_adapter as adapter

def authenticate(run,cfg,*,local=False,frozen=True):
    import json
    spec_path=Path(run)/'dashboard_spec.json';spec=json.loads(spec_path.read_text())
    receipt_path=Path(os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT'])
    receipt=gpu_registry.load_receipt(receipt_path)
    binding,_=adapter._load_binding(os.environ['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'])
    contract=adapter.validate_contract(spec,launch=True)
    if (contract['policy']!='required' or binding['policy']!='required'
       or binding['experiment_id']!=cfg['id'] or binding['contract_sha256']!=adapter._json_sha256(contract)
       or binding['spec_sha256']!=adapter._json_sha256(spec)
       or binding['lease_receipt_sha256']!=sha(receipt_path) or binding['lease_id']!=receipt['lease_id']):
        raise RuntimeError('Admission identity mismatch')
    registry=gpu_registry.SharedGpuRegistry(receipt['registry_dir'])
    lease=registry.assert_owned(receipt,cfg['id'],sha(spec_path));registry.assert_owned_nodes(receipt,cfg['nodes'])
    if time.time()-lease['heartbeat_at']>lease['ttl_s']:raise RuntimeError('Expired lease')
    if frozen:
        for d in spec['frozen_inputs']:
            if sha(d['path'])!=d['sha256']:raise RuntimeError('Changed frozen input: '+d['path'])
    if local:
        node=os.uname().nodename.split('.')[0]
        if node not in cfg['nodes'] or not re.search(r'(?:^|/)job_'+cfg['jobs'][node]+r'(?:/|$)',Path('/proc/self/cgroup').read_text()):
            raise RuntimeError('Wrong node/job cgroup')
        if os.environ.get('EXPERIMENT_GUARD_MODEL_CACHE_ADMISSION_ID')!=binding['admission_id']:
            raise RuntimeError('Missing exact exec-local admission')
    return lease,binding
