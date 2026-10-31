import sys,json,time,os
from pathlib import Path
base=Path.cwd();sys.path.insert(0,str(base/'experiments/hy3_idle_runtime_20261019/guard_candidate/scripts'))
from gpu_registry import SharedGpuRegistry
root=base/'experiments/idle_gpu_fleet_20261019/hy3_prefill_20261019';out=root/'validation/live_handoff_manual';out.mkdir(mode=0o700,exist_ok=True)
registry=SharedGpuRegistry('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry');before=json.loads((root/'state.json').read_text());run=Path(before['run']);spec=dict(id='hy3-background-handoff-check-'+str(int(time.time())),port=19300,resource_priority='normal',tasks=[dict(id='claim-only',nodes=['spartan-gpgpu168'])])
started=time.time();receipt=None
try:
 receipt,lease=registry.claim(spec,'cpu-only-handoff-check',['spartan-gpgpu168','spartan-gpgpu169'],out/'private-receipt.json')
 elapsed=time.time()-started
 cleanup=json.loads((run/'task/cleanup.json').read_text());assert cleanup['passed'] and cleanup['gpu_resources_released']
 result=dict(passed=True,timestamp=time.time(),handoff_s=elapsed,normal_claim_without_preempt_flag=True,foreground_nodes=lease['nodes'],background_nodes=['spartan-gpgpu168','spartan-gpgpu169'],old_run=str(run),cleanup=cleanup,inference_sent_by_validation=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='cleanup'}))
finally:
 if receipt:registry.release(receipt,True,'CPU-only handoff validation; no foreground GPU allocation or model started')
