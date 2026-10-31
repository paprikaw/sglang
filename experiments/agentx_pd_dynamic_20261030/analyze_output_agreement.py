"""Describe timed output agreement, separately from controlled correctness tests."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', action='append', required=True,
                    help='run_directory_name/arm_name')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    baseline = ROOT/'runs/eight_gpu_col_ep4_durable_r3'
    baseline_manifest = read(baseline/'launch_manifest.json')
    graph_digest = hashlib.sha256((baseline/'agentx_graphs.json').read_bytes()).hexdigest()
    results, sources = [], {}
    for choice in args.candidate:
        name, arm = choice.split('/')
        run = ROOT/'runs'/name
        manifest = read(run/'launch_manifest.json')
        assert manifest['model_revision'] == baseline_manifest['model_revision']
        assert hashlib.sha256((run/'agentx_graphs.json').read_bytes()).hexdigest() == graph_digest
        for p in sorted(run.glob(f'results/{arm}/*/sticky_hash/formal.r*.json')):
            case = p.parent.parent.name
            repeat = p.stem.split('.')[-1]
            reference = baseline/'results/col_ep4'/case/'sticky_hash'/p.name
            if not reference.exists():
                continue
            assert read(p.with_name('completed.'+repeat+'.json'))['passed']
            a, b = read(reference), read(p)
            key = lambda r: (r['root_id'], r['conversation_id'], r['turn'])
            left = {key(r):r for r in a['requests']}
            right = {key(r):r for r in b['requests']}
            assert left.keys() == right.keys()
            count = dict(sequences=0, equal_sequences=0, equal_first_tokens=0,
                         common_prefix_tokens=0, total_output_tokens=0,
                         cold_on_both=0, cold_first_token_mismatches=0)
            for k, x in left.items():
                y = right[k]
                assert (x['input_tokens'], x['output_tokens']) == (y['input_tokens'], y['output_tokens'])
                u, v = x['output_ids'], y['output_ids']
                assert len(u) == len(v) == x['output_tokens']
                count['sequences'] += 1
                count['equal_sequences'] += u == v
                count['equal_first_tokens'] += u[0] == v[0]
                count['common_prefix_tokens'] += next((i for i,(s,t) in enumerate(zip(u,v)) if s != t), len(u))
                count['total_output_tokens'] += len(u)
                if not x['meta_info'].get('cached_tokens') and not y['meta_info'].get('cached_tokens'):
                    count['cold_on_both'] += 1
                    count['cold_first_token_mismatches'] += u[0] != v[0]
            results.append(dict(run=name, arm=arm, engine_commit=manifest['engine_commit'],
                                case=case, repeat=repeat, **count))
            for source in (p, reference):
                sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'output_agreement.json').write_text(json.dumps(dict(
        rows=results, sources=sources, baseline=str(baseline),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Descriptive output agreement under different live batch/cache schedules. Not a correctness pass/fail test or attribution to numerical errors.'), indent=2)+'\n')
    lines = ['# 计时回放的输出一致程度', '',
        '参考为E5合并EP4；同一源请求、repeat命名空间和输入构造器，模型revision与图文件哈希匹配。'
        '每行是一轮完整episode，不合并重复或部分运行。不同布局面对不同实时batch和缓存历史。', '',
        '| 候选 | 负载/轮次 | 调用数 | 全输出相同 | 首token相同 | 共同输出前缀token | 总输出token | 双方均无缓存时首token不同/调用数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['arm']} ({r['run']}) | {r['case']}/{r['repeat']} | {r['sequences']} | {r['equal_sequences']} | {r['equal_first_tokens']} | {r['common_prefix_tokens']} | {r['total_output_tokens']} | {r['cold_first_token_mismatches']}/{r['cold_on_both']} |")
    lines += ['',
        '计时输出并非逐请求完全相同；不能把受控串行参考及host恢复检查的通过，表述成整个动态batch回放逐token一致。'
        '无缓存请求也存在差异，所以这些配对本身不能定位为缓存错误；同样尚未通过logits等控制将差异归因为浮点归约。', '',
        '回放固定来源记录的输入/输出长度，后续请求使用来源hash合成输入，不拼接本轮实际生成文本。'
        '因此布局间提供的源请求长度一致，但生成内容、实际专家路由和实际缓存工作量不保证一致。'
        '性能结论限定于该派生回放；受控KV/GDN交接、恢复和压力修复仍各自需要对应验证证据。', '',
        '[逐轮数据与来源哈希](output_agreement.json)']
    (args.out/'OUTPUT_AGREEMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(paired_complete_episodes=len(results), paired_calls=sum(r['sequences'] for r in results))))


if __name__ == '__main__':
    main()
