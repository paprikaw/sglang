"""Export official AgentX conversation DAGs with original length/hash payloads.

The upstream loader constructs all spawn/join, flattened-chain and interval-order
edges. Its text rendering is replaced with a tiny placeholder because this export
uses source token counts and hash IDs directly, not the rendered chat messages.
This is an AgentX-derived systems experiment, not an official AgentX score.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from aiperf.common.environment import Environment
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
SOURCE = WS / 'datasets/agentx_20261030/cc-traces-weka-062126-256k/traces.jsonl'


def loader():
    cfg = BenchmarkConfig.model_validate({
        'models': ['Qwen3.6-35B-A3B'],
        'endpoint': {'urls': ['http://localhost:8000/v1/chat/completions'], 'wait_for_model_timeout': 0},
        'datasets': [{'name': 'default', 'type': 'file', 'records': [{'text': 'placeholder'}],
                      'format': 'weka_trace'}],
        'phases': [{'name': 'profiling', 'type': 'concurrency', 'requests': 100, 'concurrency': 1}],
        'tokenizer': {'name': 'source-token-graph-export'}, 'runtime': {'ui': 'simple'},
    })
    run = BenchmarkRun(benchmark_id='agentx-graph-export', cfg=cfg,
                       artifact_dir=cfg.artifacts.dir, random_seed=42)
    obj = WekaTraceLoader(filename=None, run=run)
    obj.prompt_generator = SimpleNamespace(
        _cache={}, _hash_id_corpus_rng=HashIdRandomGenerator(42, _internal=True),
        _tokenized_corpus=list(range(1000, 11000)), _corpus_size=10000,
        _bpe_stable_terminator_tokens=[], tokenizer=SimpleNamespace(decode=lambda tokens: 'x'),
    )
    Environment.DATASET.WEKA_PARALLEL_WORKERS = 1
    return obj


def requests(trace):
    for outer, req in enumerate(trace['requests']):
        if req['type'] == 'subagent':
            for inner, child in enumerate(req.get('requests', [])):
                yield outer, inner, child
        else:
            yield outer, None, req


def graph(obj, raw):
    parsed = WekaTrace.model_validate(raw)
    conversations = obj.convert_to_conversations({parsed.id: [parsed]})
    rows = []
    for conv in conversations:
        item = conv.model_dump(mode='json', exclude={'turns'})
        item['replay_scope_id'] = conv.replay_scope_id
        item['turns'] = []
        for turn in conv.turns:
            r = raw['requests'][turn.source_outer_idx]
            if turn.source_inner_idx is not None:
                r = r['requests'][turn.source_inner_idx]
            meta = turn.model_dump(mode='json', include={
                'timestamp', 'delay', 'api_time_ms', 'source_trace_id', 'source_outer_idx',
                'source_inner_idx', 'source_kind', 'branch_ids', 'prerequisites',
                'replay_predecessors', 'reset_context', 'max_tokens', 'input_kind',
            })
            meta.update(input_tokens=r['in'], output_tokens=max(1, r['out']), hash_ids=r.get('hash_ids', []))
            meta['replay_predecessors'] = [p.model_dump(mode='json') for p in turn.replay_predecessors]
            assert meta['output_tokens'] == turn.max_tokens
            item['turns'].append(meta)
        rows.append(item)
    return {'id': raw['id'], 'block_size': raw.get('block_size', 64), 'conversations': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'agentx_graph_candidates.json')
    ap.add_argument('--limit', type=int, default=24)
    args = ap.parse_args()
    candidates = []
    for line_no, line in enumerate(SOURCE.open()):
        raw = json.loads(line)
        reqs = [r for _, _, r in requests(raw)]
        count = len(reqs)
        if not reqs:
            continue
        total_out = sum(max(1, r['out']) for r in reqs)
        max_context = max(r['in'] + max(1, r['out']) for r in reqs)
        # Whole sessions only; no token truncation or discarded child calls.
        if 4 <= count <= 80 and total_out <= 16000 and max_context <= 256000:
            candidates.append((max_context, count, total_out, line_no, raw))
    candidates.sort(key=lambda x: (x[0], x[1], x[3]))
    selected = candidates if len(candidates) <= args.limit else [
        candidates[round(i * (len(candidates) - 1) / (args.limit - 1))]
        for i in range(args.limit)
    ]
    obj = loader()
    exported = []
    for max_ctx, n, out, line_no, raw in selected:
        result = graph(obj, raw)
        retained = sum(len(c['turns']) for c in result['conversations'])
        result.update(source_line_zero_based=line_no, source_requests=n,
                      retained_requests=retained, max_context=max_ctx, output_tokens=out,
                      source_record_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
        exported.append(result)
        print(json.dumps({k: result[k] for k in ['id', 'source_requests', 'retained_requests', 'max_context', 'output_tokens']}), flush=True)
    payload = {'schema_version': 1, 'source': str(SOURCE),
               'harness_commit': 'e4bb2f835ffedbbe6b76252ea5c93caec68c5faf',
               'eligibility': 'Whole sessions, 4..80 calls, <=16000 output tokens, <=256000 total context; evenly spaced across max context; upstream orphan policy retained.',
               'candidate_count': len(candidates), 'graphs': exported, 'official_score': False}
    args.output.write_text(json.dumps(payload, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
