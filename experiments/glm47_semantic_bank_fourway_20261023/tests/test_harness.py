import ast,copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
import common
from common import ARMS,LAYOUTS,select_arm,server_args,environment
from admission_adapter import identity,release_local,new_state
from spec_proof import validate_receipts
from scheduler_hooks import scheduler_mode

def config():
    return dict(id='test',run_dir='/tmp/fourway-test',nodes=['spartan-gpgpu065','spartan-gpgpu167'],
        dist_port=31415,api_port=31416,engine_root=str(common.ENGINE),engine_commit=common.PINNED_COMMIT,
        semantic_inputs=str(ROOT/'semantic_inputs/p4096'))

@pytest.mark.parametrize('arm',list(ARMS))
def test_actual_on_off_arguments_and_environment(arm):
    cfg=select_arm(config(),arm);args=server_args(cfg,0);env=environment(cfg,0,{})
    assert ('--speculative-algorithm' in args)==cfg['mtp']
    if cfg['mtp']:assert args[args.index('--speculative-draft-model-path')+1]==str(common.MODEL)
    assert env['GLM_ARM']==env['ARM']==arm
    assert env['GLM_LAYOUT']==cfg['layout']
    enabled=cfg['layout']=='pp2ep4' and cfg['mtp']
    assert env['SGLANG_PP_SPEC_MICROBATCH']==str(int(enabled))
    assert env['SGLANG_ENABLE_PP_SPEC']==str(int(enabled))
    assert env['DEEPEP_LL_PCIE_P2P_MAX_TOKENS']=='0'
    assert env['DEEPEP_LL_NVLINK_PAIR_ONLY']=='1'
    assert env['SGLANG_ENABLE_COLOCATED_BATCH_GEN']=='0'
    graph=json.loads(args[args.index('--cuda-graph-config')+1])
    assert graph['decode']==dict(backend='full',bs=[4],max_bs=4)
    if cfg['layout']=='pp2ep4':assert args[args.index('--pp-max-micro-batch-size')+1]=='4'

def test_five_fresh_arms_four_consumers():
    assert list(ARMS)==['producer','ep8_mtp_on','ep8_mtp_off','pp2ep4_mtp_on','pp2ep4_mtp_off']
    assert sum(row['role']=='consumer' for row in ARMS.values())==4
    assert ARMS['producer']==dict(layout='pp2ep4',mtp=True,role='producer')

def test_profile_cohort_allowed_with_same_exact_local_b4():
    assert identity('profile-decode-4096-7')==('profile-decode-4096',7)
    class Input:
        def __init__(self,index):self.rid='profile-decode-4096-'+str(index);self.routed_dp_rank=index//4
    state=new_state();items=[Input(index) for index in range(4,8)]
    assert release_local(items[:2],dp_rank=1,state=state,tokenized_type=Input)==[]
    assert release_local(items[2:],dp_rank=1,state=state,tokenized_type=Input)==items
    assert state['proofs'][0]['added_collectives']==0

@pytest.mark.parametrize('layout,mtp,loop,enabled',[('ep8',True,1,False),('ep8',False,1,False),('pp2ep4',True,2,True),('pp2ep4',False,2,False)])
def test_initialized_scheduler_mode(layout,mtp,loop,enabled):
    scheduler=SimpleNamespace(_pp_spec_microbatch=enabled,pp_loop_size=loop,ps=SimpleNamespace(pp_size=LAYOUTS[layout]['pp']),
        _pp_spec_feedback_group=SimpleNamespace(unique_name='feedback',ranks=list(range(8))) if enabled else None)
    assert scheduler_mode(scheduler,dict(layout=layout,mtp=mtp))['enabled']==enabled
    scheduler._pp_spec_microbatch=not enabled
    with pytest.raises(RuntimeError):scheduler_mode(scheduler,dict(layout=layout,mtp=mtp))

def graph_receipts(layout,mtp):
    rows=[]
    for rank in range(8):
        draft=mtp and (layout=='ep8' or rank>=4)
        row=dict(forwards={'target:TARGET_VERIFY' if mtp else 'target:DECODE':12},replays={'target':12},
            widths={'target:4':12},phases={},batch_sizes={'4':12})
        if draft:row['phases']['draft']=12;row['replays']['draft']=12;row['widths']['draft:4']=12
        rows.append(dict(rank=rank,graph=dict(installed=True,cohorts={'timed-decode-4096':row})))
    return rows

@pytest.mark.parametrize('layout,mtp',[('ep8',True),('ep8',False),('pp2ep4',True),('pp2ep4',False)])
def test_actual_graph_and_draft_placement(layout,mtp):
    rows=graph_receipts(layout,mtp)
    assert validate_receipts(rows,layout,mtp=mtp)['passed']
    rows[0]['graph']['cohorts']['timed-decode-4096']['widths']['target:4']=0
    with pytest.raises(RuntimeError):validate_receipts(rows,layout,mtp=mtp)

def test_mtp_off_rejects_draft_runtime():
    rows=graph_receipts('pp2ep4',False);rows[4]['graph']['cohorts']['timed-decode-4096']['phases']['draft']=1
    with pytest.raises(RuntimeError):validate_receipts(rows,'pp2ep4',mtp=False)

def test_all_source_parse_and_no_witness_install():
    for path in [ROOT/'prepare_measure.py',*(ROOT/'runtime').glob('*.py')]:ast.parse(path.read_text())
    source=(ROOT/'runtime/startup.py').read_text()
    assert 'install_witness' not in source
    assert 'install_activity()' in source

def test_uniform_import_outside_timing_before_each_cohort():
    source=(ROOT/'runtime/measure.py').read_text()
    assert "for cohort in ('timed','repeat')" in source
    assert source.index("imported=import_cold(run,cfg,cohort+'_preparation')")<source.index("save(out/'TIMED'")
    assert "if cfg['role']=='producer':return produce_bank" in source
    assert 'prepare_semantic_cohort' not in source

def test_contract_has_unique_actual_usage_and_joint_budget(tmp_path):
    path=ROOT/'prepare_measure.py';spec=importlib.util.spec_from_file_location('fourway_prepare_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    preload=tmp_path/'preload';preload.mkdir()
    for node in config()['nodes']:(preload/(node+'.json')).write_text('{}')
    cfg=dict(config(),run_dir=str(tmp_path/'run'));contract=module.build_contract(cfg,preload)
    assert contract['policy']=='required' and contract['on_miss']=='fail'
    assert set(contract['hicache_reserve_bytes_per_node'].values())=={160*1024**3}
    assert contract['cpu_memory_budget']['kv_banks_per_node']=={node:[] for node in cfg['nodes']}
    for index,node in enumerate(cfg['nodes']):
        rows=contract['usage_logs'][node];assert len({row['path'] for row in rows})==len(rows)
        assert len(rows)==(6 if index==0 else 8)
        assert not any('mtp_off-draft' in row['id'] for row in rows)
