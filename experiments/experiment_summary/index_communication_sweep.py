"""Index an accepted communication sweep as one table row; never launch GPUs."""
import csv
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'single_layer_ep4_ep8_matched_20261024/ce_rdma_r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    item = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    if path.suffix == '.csv':
        with path.open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        item.update(rows=len(rows), columns=reader.fieldnames)
    return item


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    config = json.loads((RUN / 'config.json').read_text())
    validation = json.loads((RUN / 'independent_validation.json').read_text())
    assert manifest['status'] == 'complete' and validation['passed']
    assert manifest['expert_compute'] is manifest['attention_compute'] is manifest['actual_model_weights_loaded'] is False
    rank_files = sorted(RUN.glob('*/result_rank*.json'))
    assert len(rank_files) == 64
    for path in rank_files:
        row = json.loads(path.read_text())
        assert row['passed']
        assert row['expert_compute'] is row['attention_compute'] is row['model_loaded'] is False
    comparisons = list(csv.DictReader((RUN / 'transport_comparison.csv').open()))
    assert len(comparisons) == 80
    assert source(RUN / 'phase_metrics.csv')['rows'] == 280
    assert source(RUN / 'full_metrics.csv')['rows'] == 840
    def point(model, tokens):
        return next(row for row in comparisons if row['model'] == model and row['ep4_node'] == '065'
                    and row['boundary'] == 'whole_vs_half' and int(row['global_batch_tokens']) == tokens)
    glm = point('glm', 4096)
    primary = source(RUN / 'transport_comparison.csv')
    dataset = {
        'id': 'T01', 'experiment_id': manifest['id'], 'family': 'communication_only',
        'title': 'A100 单层通信形状：GLM／Hy3，EP4／EP8，CE／RDMA',
        'status': 'complete', 'source_run': str(RUN),
        'archive_root': '/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/' + manifest['id'],
        'primary_csv': primary, 'phase_metrics': source(RUN / 'phase_metrics.csv'),
        'aligned_samples': source(RUN / 'full_metrics.csv'),
        'report': source(RUN / 'summary.md'), 'figure': source(RUN / 'transport_curves.png'),
        'manifest': source(RUN / 'manifest.json'), 'validation': source(RUN / 'independent_validation.json'),
        'nodes': config['nodes'], 'gpu_model': 'A100', 'nic_model': 'ConnectX-6 Dx',
        'native_sha256': {'glm': manifest['native_glm'], 'hy3': manifest['native_hy3']},
        'actual_model_weights_loaded': False, 'expert_compute': False, 'attention_compute': False,
        'kv_cache_used': False, 'cuda_graph': False, 'timing_unit': 'ms',
        'measured': ['dispatch_with_receive_hook', 'combine_with_receive_hook', 'communication_pair', 'cpu_wall'],
        'input': 'deterministic synthetic BF16 vectors; balanced top-8 routing; no expert GEMM',
        'observed_axes': {'model_geometry': ['glm', 'hy3'], 'transport': ['rdma', 'ce'],
            'whole_batch_unique_tokens': config['token_sweep'], 'ep4_node': ['065', '167'],
            'comparison_boundary': ['whole_vs_half', 'equal_call']},
        'routing_metadata_is_not_expert_compute': True,
        'boundaries': {
            'whole_vs_half': 'EP8 processes the whole batch; EP4 processes each half in a separate call; equal assignments per GPU. EP4 table value averages the two call medians.',
            'equal_call': 'Same unique tokens per call; EP4 receives twice as many token-expert assignments per GPU as EP8.'},
        'warmup_per_point': 3, 'formal_samples_per_point': 3,
        'aggregation': 'maximum across aligned ranks per sample, then median; separately maximized dispatch/combine are not additive',
        'pair_points': 280, 'aligned_sample_count': 840,
        'representative_rows': [point(model, tokens) for model in ('glm', 'hy3') for tokens in (8, 1024, 4096)],
        'cannot_answer': ['expert GEMM latency', 'full PP speedup', 'request TTFT or TPOT', 'Attention latency'],
    }
    inventory = {'generated_date': '2026-10-24', 'scope': 'pure communication sweeps indexed in README section 6',
                 'no_gpu_launch': True, 'datasets': [dataset]}
    (HERE / 'communication_sweeps.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    readme = HERE / 'README.md'
    text = readme.read_text()
    obsolete = '## 8. 本轮新增：单层 EP4／EP8 的 CE 与 RDMA 通信扫描'
    if obsolete in text:
        start = text.index(obsolete)
        end = text.index('## 9. 以后怎么追踪', start)
        text = text[:start] + text[end:]
    text = text.replace('## 9. 以后怎么追踪', '## 8. 以后怎么追踪')
    text = text.replace('第 1–7 节整理历史数据；第 8 节记录本轮新增的单层 CE／RDMA 通信对照，运行状态单独标明。',
                        '历史请求结果见第 2–5 节；通信扫描见第 6 节（含本轮 T01）；单层 Attention／Expert 扫描见第 7 节。')
    start = text.index('## 6. ')
    end = text.index('## 7. ', start)
    section = text[start:end].replace('| 入口 | 测了什么 | 没有测什么 |', '| ID／入口 | 测了什么 | 没有测什么 |')
    entry = (f'| [T01：A100 · GLM／Hy3 · EP4／EP8 · CE／RDMA]({RUN / "summary.md"}) · '
             f'[80 行比较表]({RUN / "transport_comparison.csv"})／[Dispatch、Combine 分项]({RUN / "phase_metrics.csv"})／'
             f'[曲线]({RUN / "transport_curves.png"}) | '
             '本轮新增：整批唯一 token 从 8 逐次翻倍到 4096；分别比较 EP8 整批／EP4 每次半批，以及单次 token 相同。'
             '065／167 同型 ConnectX-6 Dx；eager，每点正式 3 次，共 280 点。'
             f'整批 4096、EP4 每次 2048 token 时，065 上 GLM 的通信 RDMA／CE 为 '
             f'{float(glm["ep4_rdma_ms"]):.4f}／{float(glm["ep4_ce_ms"]):.4f} 毫秒；全部形状与节点见报告。 | '
             '**只有通信，没有 Expert GEMM、专家权重、Attention 或 KV Cache。** '
             '专家数／top-8 只用于构造路由。CE 在节点内大消息上明显受益，但跨节点 EP8 收益不稳定；'
             '不能直接解释成完整 PP 加速或 Prefill／Decode 延迟。 |')
    section = re.sub(r'^\| \[T01：.*\n?', '', section, flags=re.MULTILINE)
    header = '|---|---|---|\n'
    assert header in section
    section = section.replace(header, header + entry + '\n', 1)
    note = (f'本轮及后续纯通信扫描以表内 ID 检索；[通信扫描清单]({HERE / "communication_sweeps.json"})'
            '保存 T01 的数据路径、哈希、token 范围、后端和计时边界。**T01 不包含 Expert 计算；'
            '要看已有的真实 Expert 计算，请看第 7.2 节 E02／E04／E05，旧协议不同的数据不能直接相加。**\n\n')
    if '本轮及后续纯通信扫描以表内 ID 检索' not in section:
        section += note
    text = text[:start] + section + text[end:]
    readme.write_text(text)
    links = re.findall(r'\]\((/[^)]+)\)', text)
    assert all(Path(link).exists() for link in links)
    old = json.loads((HERE / 'component_validation.json').read_text())
    for name, digest in old['original_request_metrics_and_profile_hashes_unchanged'].items():
        assert sha(HERE / name) == digest, name
    old.update(readme_sha256=sha(readme), markdown_absolute_links_checked=len(links))
    (HERE / 'component_validation.json').write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')
    docs = [readme, *(HERE / 'component_reports').glob('*.md')]
    all_links = [link for doc in docs for link in re.findall(r'\]\((/[^)]+)\)', doc.read_text())]
    assert all(Path(link).exists() for link in all_links)
    report_validation = json.loads((HERE / 'component_reports_validation.json').read_text())
    report_validation.update(markdown_links_checked=len(all_links), missing_links=[])
    (HERE / 'component_reports_validation.json').write_text(json.dumps(report_validation, ensure_ascii=False, indent=2) + '\n')
    assert text.count('| [T01：') == 1 and obsolete not in text
    receipt = {'passed': True, 'no_gpu_launch': True, 'table_id': 'T01', 'readme_section': 6,
               'communication_inventory_sha256': sha(HERE / 'communication_sweeps.json'),
               'readme_sha256': sha(readme), 'raw_rank_flags_checked': len(rank_files),
               'expert_compute': False, 'markdown_links_checked': len(all_links),
               'historical_request_and_profile_csv_unchanged': True,
               'appended_inline_report_removed': True}
    (HERE / 'communication_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
