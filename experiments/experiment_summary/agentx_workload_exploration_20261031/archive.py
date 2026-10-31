"""Archive only this CPU workload analysis; preserve concurrent experiment topics."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
IDENTIFIER = 'qwen35b_h100_agentx_workload_exploration_20261031'
STORE = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / IDENTIFIER
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    if source.suffix == '.md':
        def absolute_link(match):
            target = match.group(1)
            if target.startswith(('/', '#')) or re.match(r'\w+://', target):
                return match.group(0)
            return '](' + str((source.parent / target).resolve()) + ')'
        destination.write_text(re.sub(r'\]\(([^)]+)\)', absolute_link, source.read_text()))
    else:
        shutil.copyfile(source, destination)
    return dict(source=str(source), source_sha256=sha(source), snapshot=destination.name,
                snapshot_sha256=sha(destination))


def main():
    result = json.loads((HERE / 'exploration.json').read_text())
    assert result['cpu_only'] and len(result['verified_raw_episodes']) == 21
    assert result['corpus_256k_extra']['counters']['calls'] == 68266
    STORE.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in ('.md', '.json', '.csv', '.py', '.txt', '.log'):
            files.append(copy(source, STORE / source.name))
    files.append(copy(HERE / 'REPORT.md', STORE / 'summary.md'))
    files.append(copy(HERE / 'cache_hit_definitions.csv', STORE / 'full_metrics.csv'))
    manifest = dict(id=IDENTIFIER, title='AgentX corpus, replay methodology, session shape and cache characterization',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='complete_cpu_analysis_no_gpu_launch',
        implementation=dict(engine='Historical SGLang', replayer='V3, fixed accepted cohort only',
            colocated_engine_commit='37e0828c642ce4f3795c4c4394f4ce913bcd1a35',
            pd_engine_commit='797f6488cf5fea2f88e38ccaeb1703443deec537',
            official_loader_commit='e4bb2f835ffedbbe6b76252ea5c93caec68c5faf',
            engine_source='Prior frozen runs, not the current worktree; no engine edits or launch'),
        model='Qwen3.6-35B-A3B, BF16', testbed='Historical two-node 8 x H100 80GB',
        workload=dict(full_roots=393,full_calls=98827,capped_256k_calls=68266,
            selected_roots=8,selected_tracks=18,selected_calls=171,
            accepted_episodes=21,accepted_calls=34371,
            selection='Whole families 4-80 calls, <=16000 cumulative output, each input+output<=256000',
            cloning='C8 one copy; C128 sixteen cache-isolated copies of the same eight families',
            timing='Finite closed-loop replay from start, global no-inflight idle cap 10s; V3 branch caveats retained'),
        parameters=dict(hicache=True,cpu_hicache_gb_per_rank=32,formal_start_cache_flush=True,
            within_session_flush=False,pd_cache_metadata='P-side prefill_meta'),
        provenance=str(HERE / 'exploration.json'),
        new_pilot_supplement=json.loads((HERE/'pilot_cache_supplement.json').read_text()), artifacts=files,
        key_results=dict(selected_mean_input=84856.14035087719,selected_mean_output=435.719298245614,
            selected_explicit_subagent_calls=45,selected_explicit_subagent_tracks=2,
            selected_reconstructed_flat_tracks=8,c8_token_weighted_hit_pct=91.4619031866799),
        limitations=result['caveats']+[
            'Sample is biased toward short trajectories and outputs; cloning increases load, not diversity',
            'No official AgentX score, model quality evaluation, or active-prefetch accuracy measurement',
            'Live official methodology and the pinned FAQ have different warmup and duration defaults'])
    (STORE / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    entry = f'''<!-- begin {IDENTIFIER} -->
### {IDENTIFIER}

Snapshot: {STORE}

CPU-only characterization of the 393-root full/256k corpora and the prior 21 accepted E5/V3 and E7/V3 episodes. Current eight families: 171 calls, 18 tracks (8 root, 2 explicit subagent, 8 reconstructed flat); mean input 84856/output 436. Separate token-weighted hit coverage, request hit probability, GPU/host locations and new-P tails. First E8/V4 C8 completed pilot kept as a separate one-episode supplement. No active-prefetch accuracy or official AgentX score. Source hashes, corpus scan, replay protocol and source-version differences preserved.

Report: {HERE/'REPORT.md'}

<!-- end {IDENTIFIER} -->'''
    with REGISTRY.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        text = stream.read()
        start, end = f'<!-- begin {IDENTIFIER} -->', f'<!-- end {IDENTIFIER} -->'
        if start in text:
            left, rest = text.split(start, 1)
            _, right = rest.split(end, 1)
            text = left + entry + right
        else:
            text = text.rstrip() + '\n\n' + entry + '\n'
        stream.seek(0)
        stream.write(text)
        stream.truncate()
    print(json.dumps(dict(snapshot=str(STORE), artifacts=len(files))))


if __name__ == '__main__':
    main()
