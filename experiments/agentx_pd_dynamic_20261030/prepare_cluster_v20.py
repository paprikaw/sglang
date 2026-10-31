"""Freeze one E7 cohort with corrected branch timing, including colocated arms.

Reuse the durable V13 preparation and apply the explicit capacity, E7 source,
deadline and diagnostic gates. Never modify a previously prepared bundle.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'
PARENT = 'fc014e3553c4d234f38f3088a3158c0333e64830'
HARNESS_COMMIT = 'e4bb2f835ffedbbe6b76252ea5c93caec68c5faf'
HARNESS = ROOT.parents[1]/'datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8'


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--episode-timeout-s', type=int, default=7200)
    parser.add_argument('--decode-extra-slots', type=int, default=8)
    opts, rest = parser.parse_known_args(sys.argv[1:])
    assert 2400 < opts.episode_timeout_s <= 14400
    assert opts.decode_extra_slots > 0
    assert '--reference' in rest and '--reuse-run' not in rest
    sys.argv = [sys.argv[0]] + rest
    run = Path(rest[rest.index('--run')+1]).resolve()
    reference_path = Path(rest[rest.index('--reference')+1]).resolve()
    prep = load('v20_prep', ROOT/'prepare_cluster.py')
    reference = prep.read(reference_path)
    assert reference['source_arm'].startswith('col_')
    assert len(reference['output_ids']) == 6
    assert all(len(ids) == 32 for ids in reference['output_ids'])
    assert '22 passed' in (ROOT/'unit_r7_restore_pressure_green.log').read_text()
    runpy.run_path(str(ROOT/'prepare_cluster_v13.py'), run_name='__main__')
    cfg = prep.read(run/'config.json')
    manifest = prep.read(run/'launch_manifest.json')
    engine = Path(manifest['engine_root'])
    assert manifest['engine_commit'] == COMMIT and not cfg['imported_episodes']
    changed = subprocess.check_output(['git', '-C', str(engine), 'diff', '--name-only',
                                      PARENT, COMMIT], text=True).splitlines()
    assert set(changed) == {
        'python/sglang/srt/disaggregation/decode.py',
        'python/sglang/srt/disaggregation/decode_hicache_mixin.py',
        'test/registered/unit/disaggregation/test_decode_hicache_restore_pressure.py'}
    assert prep.read(run/'evidence/reference.json') == reference
    assert prep.read(run/'reference.input.json') == reference

    before = ast.parse((run/'code/agentx_replay.py').read_text())
    text = (ROOT/'agentx_replay_v4.py').read_text()
    after = ast.parse(text)
    unchanged = ['http', 'stream', 'request_one', 'token_ids', 'lcp', 'pool_clone_lane',
                 'Router', 'assign_clone_balanced_lane']
    for name in unchanged:
        node = lambda tree: next(n for n in tree.body if getattr(n, 'name', None) == name)
        assert ast.dump(node(before)) == ast.dump(node(after)), name
    text = change(text, 'namespace, idle_gap_cap_s=10, copies=1):',
        'namespace, idle_gap_cap_s=10, copies=1, episode_timeout_s=7200, partial_rows=None):')
    text = change(text, '    completed_rows = []',
        '    completed_rows = partial_rows if partial_rows is not None else []\n'
        '    assert not completed_rows and episode_timeout_s > 0')
    text = change(text, '    deadline = start + 2400', '    deadline = start + episode_timeout_s')
    text = change(text, "raise TimeoutError('Finite AgentX episode exceeded 2400 seconds')",
        "raise TimeoutError(f'Finite AgentX episode exceeded {episode_timeout_s} seconds')")
    text = change(text, '                idle_compressed_s=skipped_idle, idle_gap_cap_s=idle_gap_cap_s, slo=slo,',
        '                idle_compressed_s=skipped_idle, idle_gap_cap_s=idle_gap_cap_s, slo=slo,\n'
        '                episode_timeout_s=episode_timeout_s,')
    ast.parse(text)
    (run/'code/agentx_replay.py').write_text(text)
    for name in ['agentx_branch_timing.py', 'episode_capture.py']:
        shutil.copyfile(ROOT/name, run/'code'/name)
    text = (ROOT/'cluster_runner_v9.py').read_text()
    text = change(text, 'import agentx_replay as replay',
        'import agentx_replay as replay\nfrom episode_capture import replay_with_failure_capture')
    text = change(text, '                result = replay.replay(graphs, endpoints, policy,',
        "                result = replay_with_failure_capture(replay.replay, base.h.save, out/f'partial.r{repeat+1}.json',\n"
        '                    graphs, endpoints, policy,')
    text = change(text, "idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1))",
        "idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1), episode_timeout_s=root['episode_timeout_s'])")
    ast.parse(text)
    (run/'code/cluster_runner.py').write_text(text)

    cfg.update(decode_extra_slots=opts.decode_extra_slots, waiting_timeout_s=300,
               episode_timeout_s=opts.episode_timeout_s,
               partial_episode_policy='Save collected responses on exception only; never count partial files as complete samples')
    for local in cfg['node_configs'].values():
        local['decode_extra_slots'] = opts.decode_extra_slots
        assert local['launch_env']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] == '120'
        local['launch_env']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '300'
    sys.path.insert(0, str(run/'code'))
    helper = load('v20_branch_helper', run/'code/agentx_branch_timing.py')
    cfg['branch_timing_policy'] = helper.POLICY
    graph_checks = []
    for graph in prep.read(Path(cfg['graphs_file']))['graphs']:
        vertices = helper.build_vertices(graph, 'preflight:copy0:')
        count = sum(len(c['turns']) for c in graph['conversations'])
        assert len(vertices) == count
        graph_checks.append(dict(id=graph['id'], requests=count))
    assert len(graph_checks) == 24
    cluster = load('v20_cluster_preflight', run/'code/cluster_runner.py')
    slots = []
    for node in cfg['nodes']:
        base, local = cluster.load_base(run, cfg, node)
        for arm in cfg['arms']:
            for i, inst in enumerate(base.layout(local, arm)['instances']):
                args = base.engine_args(local, arm, i)
                assert '--enable-hierarchical-cache' in args
                assert '--max-total-tokens' not in args
                if inst['role'] != 'decode':
                    assert '--disaggregation-decode-extra-slots' not in args
                    continue
                value = lambda name: int(args[args.index(name)+1])
                assert value('--disaggregation-decode-extra-slots') == opts.decode_extra_slots
                assert '--disable-overlap-schedule' not in args
                assert '--mamba-radix-cache-strategy' not in args
                dp = value('--dp-size')
                requested = value('--max-running-requests')//dp
                states = value('--max-mamba-cache-size')//dp
                required = (requested+opts.decode_extra_slots)*3
                assert required <= states, (node, arm, requested, required, states)
                slots.append(dict(node=node, arm=arm, replica=i,
                    requested_active_per_dp=requested, decode_extra_slots=opts.decode_extra_slots,
                    required_state_upper_bound=required, configured_states_per_rank=states))
    assert bool(slots) == any(c['decode'] for c in cfg['role_gpu_counts'].values())
    cfg['restore_pressure_fix'] = dict(engine_commit=COMMIT, parent_commit=PARENT,
        changed_files=changed, cpu_checks=22, gpu_qualification='separate E7 V3 run pending',
        markers=['MAMBA_CACHE_CHECKPOINT_DEFERRED', 'HICACHE_LOAD_BACK_DEFERRED',
                 'HICACHE_LOAD_BACK_RETRY_SUCCEEDED'])
    cfg['protocol'] = (
        'Two existing 4xH100 nodes, eight GPUs per arm, fixed declared role split and cache/state pools. '
        'E7 native model, overlap and kernels; native PP policy0 with declared microbatch cap; native KV+GDN P/D handoff. '
        'HiCache ON every rank, 32 decimal GB CPU/rank, GPU KV auto-sized at .95. '
        'Conversation/Attention-DP affinity; no within-episode flush; cache-isolated trace clones; '
        'all source request lengths and dependency edges retained. '
        'V4 uses the pinned harness branch issue/return offsets: root overlapping SPAWN at parent credit issue, '
        'other SPAWN at return with branch-relative offsets, FORK at return, and join gates. '
        'Finite-root admission, global-idle compression above 10 seconds and source-hash synthetic tokens remain custom. '
        'No official trajectory sampling, chat rendering or AgentX score. '
        'Warmup contexts and offered widths cover every instance/Attention-DP owner outside timing; '
        'each independent formal repeat starts after native cache flush. Three repeats, arithmetic means and all raw requests. '
        'Forward counters use per-node tmpfs notifications outside timing; completed responses are checkpointed before post diagnostics. '
        'Native handoff waiting bound 300 seconds includes Prefill waiting; HTTP timeout 600 seconds; '
        f'finite episode bound {opts.episode_timeout_s} seconds. All waiting is included in measured latency. '
        'Failure-only partial response capture adds no successful-episode timed file write; partial files excluded from means. '
        'E7 retries zero-allocation local host-prefix restoration within its existing wait bound; lost or partial prefixes still fail; '
        'H2D completion and cross-rank transfer gates remain, and current transferred GDN state is retained. '
        'Decode transfer slots and active slots are preflighted against the declared native GDN pool. '
        'No performance samples imported; corrected branch timing is a separate cohort from V3. '
        'Fixed pools only; no dynamic-policy or weight-redistribution claim.')
    prep.save(run/'config.json', cfg)
    preflight = prep.read(run/'preflight.json')
    native_path = HARNESS/'src/aiperf/timing/branch_orchestrator.py'
    preflight.update(decode_slot_budget=slots, restore_pressure_fix=cfg['restore_pressure_fix'],
        serial_reference=dict(source=str(reference_path), sha256=prep.desc(reference_path)['sha256'],
                              source_arm=reference['source_arm'], sequences=6, tokens_per_sequence=32),
        waiting_timeout=dict(seconds=300, timing_discount_seconds=0),
        episode_timeout=dict(seconds=opts.episode_timeout_s, timing_discount_seconds=0, failure_only_diagnostics=True),
        branch_timing=dict(policy=helper.POLICY, harness_commit=HARNESS_COMMIT,
                          native_source=prep.desc(native_path), helper=prep.desc(run/'code/agentx_branch_timing.py'),
                          preserved_function_asts=unchanged, graph_checks=graph_checks))
    prep.save(run/'preflight.json', preflight)
    ds = prep.read(run/'dashboard_spec.json')
    paths = [Path(x['path']) for x in ds['frozen_inputs']]
    paths += [Path(__file__).resolve(), reference_path, native_path]
    paths += [ROOT/p for p in ['agentx_replay_v4.py', 'agentx_branch_timing.py', 'episode_capture.py',
                              'cluster_runner_v9.py', 'unit_r7_restore_pressure_green.log']]
    paths += [run/'code/agentx_branch_timing.py', run/'code/episode_capture.py']
    paths += [engine/p for p in changed]
    paths += [engine/'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py',
              engine/'test/registered/unit/mem_cache/test_mamba_optional_checkpoint_pressure.py']
    ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
    prep.save(run/'dashboard_spec.json', ds)
    manifest.update(protocol=cfg['protocol'], decode_extra_slots=opts.decode_extra_slots,
        decode_slot_budget=slots, restore_pressure_fix=cfg['restore_pressure_fix'],
        serial_reference=preflight['serial_reference'], waiting_timeout_s=300,
        episode_timeout_s=opts.episode_timeout_s, partial_episode_policy=cfg['partial_episode_policy'],
        branch_timing_policy=helper.POLICY, branch_timing=preflight['branch_timing'],
        frozen_inputs=ds['frozen_inputs'])
    prep.save(run/'launch_manifest.json', manifest)
    print('V4 branch timing frozen:', helper.POLICY, 'graphs:', len(graph_checks),
          'Decode instances:', len(slots), 'frozen inputs:', len(ds['frozen_inputs']))


if __name__ == '__main__':
    main()
