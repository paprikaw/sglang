"""Freeze the isolated MHA PP transfer fix on top of the 2P6D state pilot."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parents[1]/'.engine-worktrees/agentx-mha-pp-kv-layer-map-r8-20261031'
PARENT='797f6488cf5fea2f88e38ccaeb1703443deec537'
TAG='serving/sglang/qwen35b/agentx-mha-pp-layer-map-r8-20261031'


def main():
    spec=importlib.util.spec_from_file_location('pd_state_preparation',ROOT/'prepare_2p6d_20261031.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    commit=subprocess.check_output(['git','-C',str(ENGINE),'rev-parse','HEAD'],text=True).strip()
    assert commit!=PARENT
    assert subprocess.check_output(['git','-C',str(ENGINE),'rev-parse',TAG+'^{commit}'],text=True).strip()==commit
    changed=subprocess.check_output(['git','-C',str(ENGINE),'diff','--name-only',PARENT,commit],text=True).splitlines()
    assert set(changed)=={'python/sglang/srt/disaggregation/mooncake/conn.py',
        'test/registered/unit/disaggregation/test_mooncake_hybrid_mha_pp_transfer.py'}
    log=ROOT/'unit_r8_mha_pp_transfer.log'
    assert 'passed' in log.read_text() and 'failed' not in log.read_text() and 'ERROR' not in log.read_text()
    original_read=base.read
    def read(path):
        value=original_read(path)
        if path==base.SEED/'config.json':
            for cfg in value['node_configs'].values():
                cfg.update(engine_root=str(ENGINE),engine_commit=commit,engine_tag=TAG)
        return value
    base.read=read
    base.COMMIT=commit
    base.main()
    run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
    cfg=base.read(run/'config.json')
    fix=dict(parent=PARENT,commit=commit,tag=TAG,changed_files=changed,
        cpu_checks=base.descriptor(log),kind='MHA uses registered global KV layer IDs, preserving separate K/V occurrences')
    cfg['mha_pp_transfer_fix']=fix
    cfg['protocol']=cfg['protocol'].replace('E7 engine797f6488','E8 engine'+commit[:8])
    cfg['protocol']+=' E8 also maps compacted Hybrid MHA KV buffers by registered global model-layer IDs across PP stages.'
    base.save(run/'config.json',cfg)
    preflight=base.read(run/'preflight.json');preflight['mha_pp_transfer_fix']=fix
    base.save(run/'preflight.json',preflight)
    ds=base.read(run/'dashboard_spec.json')
    for source in ds['sources']:
        if source['role']=='inference_engine':
            source.update(worktree=str(ENGINE),commit=commit,tag=TAG)
    paths=[Path(x['path']) for x in ds['frozen_inputs']]+[Path(__file__).resolve(),log]
    paths += [ENGINE/p for p in changed]
    ds['frozen_inputs']=list({str(p):base.descriptor(p) for p in paths}.values())
    base.save(run/'dashboard_spec.json',ds)
    manifest=base.read(run/'launch_manifest.json')
    manifest.update(engine_root=str(ENGINE),engine_commit=commit,engine_tag=TAG,
        protocol=cfg['protocol'],mha_pp_transfer_fix=fix,frozen_inputs=ds['frozen_inputs'])
    base.save(run/'launch_manifest.json',manifest)
    print('E8 MHA PP mapping frozen:',commit)


if __name__=='__main__':
    main()
