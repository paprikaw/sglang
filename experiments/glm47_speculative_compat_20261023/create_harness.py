"""Create a new, unfrozen MTP harness; never modify completed runs."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'glm47_a100_cached_decode_graph_opt_20261023'
ENGINE = ROOT.parents[1] / '.engine-worktrees/glm47-pp-spec-20261023'
COMMIT = subprocess.check_output(['git','-C',str(ENGINE),'rev-parse','HEAD'],text=True).strip()
shutil.copytree(SOURCE/'runtime', ROOT/'runtime', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(SOURCE/'prepare_measure.py', ROOT/'prepare_measure.py')

def edit(name, old, new):
    p = ROOT/name
    text = p.read_text()
    if text.count(old) != 1:
        raise RuntimeError((name,old,text.count(old)))
    p.write_text(text.replace(old,new))

edit('runtime/common.py', "ENGINE=WORKSPACE/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'", "ENGINE=WORKSPACE/'.engine-worktrees/glm47-pp-spec-20261023'")
edit('runtime/common.py', "COMMIT='870269702285d94bdd926a50cc29ad1a20bcf60c'", f"COMMIT='{COMMIT}'")
edit('runtime/common.py', "'--dtype','bfloat16','--quantization','humming'", "'--speculative-algorithm','EAGLE','--speculative-draft-model-path',str(MODEL),\n        '--speculative-num-steps','2','--speculative-eagle-topk','1','--speculative-num-draft-tokens','3',\n        '--dtype','bfloat16','--quantization','humming'")
edit('runtime/common.py', "CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=", "CUDA_VISIBLE_DEVICES='0,1,2,3',SGLANG_ENABLE_PP_SPEC='1' if cfg['layout']=='pp2ep4' else '0',PYTHONPATH=")
edit('runtime/startup.py', 'from graph_proof import install as install_graph_proof', 'from spec_proof import install as install_graph_proof')
edit('runtime/startup.py', '        result=original(self,*args,**kwargs)', "        from load_logging import loading\n        with loading(self.is_draft_worker):\n            result=original(self,*args,**kwargs)")
edit('runtime/startup.py', "        if len(layers)!=(89 if arm=='ep8' else 43 if p.pp_rank==0 else 46):", "        if self.is_draft_worker:\n            if len(layers)!=1:raise RuntimeError('MTP must load exactly one expert layer')\n            save(f'{arm}/draft_weights_rank{rank}.json',dict(passed=True,rank=rank,layers=layers,local_experts=expected_experts,canonical_key=getattr(self.model,'_canonical_cpu_weight_cache_key',None)))\n            return result\n        if len(layers)!=(89 if arm=='ep8' else 43 if p.pp_rank==0 else 46):")
edit('runtime/scheduler_hooks.py', 'from graph_proof import snapshot', 'from spec_proof import snapshot')
edit('runtime/measure.py', 'from graph_checks import validate_graph_receipts', 'from spec_proof import validate_receipts')
edit('runtime/measure.py', "row['actual_graph_proof']=validate_graph_receipts(after,cfg['layout'],p,128)", "row['actual_graph_proof']=validate_receipts(after,cfg['layout'])\n                    if not all(r['meta'].get('spec_verify_ct',0)>0 for r in rows):\n                        raise RuntimeError('Missing per-request actual MTP verify counts')\n                    row['spec_accept_length']=stats([r['meta'].get('spec_accept_length') for r in rows])\n                    row['spec_accept_rate']=stats([r['meta'].get('spec_accept_rate') for r in rows])\n                    row['spec_verify_ct']=stats([r['meta'].get('spec_verify_ct') for r in rows])\n                    row['synthetic_prefix_acceptance_not_natural_language_representative']=True")
edit('runtime/controller.py', "(run/cfg['layout']/node/'server.log').read_text()", "(run/cfg['layout']/node/'target_weight.log').read_text()")
edit('prepare_measure.py', "default='graph_r1'", "default='spec_r1'")
edit('prepare_measure.py', "id='glm47-a100-cached-decode-graph-20261023-'", "id='glm47-a100-cached-decode-mtp-20261023-'")
edit('prepare_measure.py', "title='GLM A100 matched Full Decode Graph: EP8 vs PP2xEP4'", "title='GLM A100 MTP EAGLE 2/1/3: EP8 vs PP2xEP4'")
edit('prepare_measure.py', "usage[node].append(dict(id=layout+'-'+node,path=str(run/layout/node/'server.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))", "usage[node].append(dict(id=layout+'-target-'+node,path=str(run/layout/node/'target_weight.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))\n            if layout=='ep8' or cfg['nodes'].index(node)==1:\n                usage[node].append(dict(id=layout+'-draft-'+node,path=str(run/layout/node/'draft_weight.log'),expected_ranks=expected_ranks(layout,cfg['nodes'].index(node))))")
edit('prepare_measure.py', "sources=[dict(worktree=str(ENGINE),commit=COMMIT)]", "sources=[dict(worktree=str(ENGINE),commit=COMMIT),dict(worktree=str(WORKSPACE/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),commit='870269702285d94bdd926a50cc29ad1a20bcf60c')]")
print(COMMIT)
