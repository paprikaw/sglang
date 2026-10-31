"""Archive small, verified offline artifacts and link Q10 without replacing results."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INDEX = ROOT / 'experiments/experiment_summary'
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / HERE.name
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_block(path, begin, end, content):
    text = path.read_text()
    block = begin + '\n' + content.rstrip() + '\n' + end
    if begin in text:
        assert text.count(begin) == text.count(end) == 1
        updated = re.sub(re.escape(begin) + r'.*?' + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        updated = text.rstrip() + '\n\n' + block + '\n'
    if updated != text:
        path.write_text(updated)
    assert updated.count(begin) == updated.count(end) == 1


def main():
    validation = json.loads((HERE / 'validation.json').read_text())
    assert validation['passed'] and validation['two_gpu_component_rows'] == 6
    assert validation['eight_gpu_rank_layer_samples'] == 24
    assert validation['missing_eight_gpu_decode_marked']
    grouped = json.loads((HERE / 'group_validation.json').read_text())
    assert grouped['passed'] and grouped['all_pp_stages_combined']
    assert grouped['eight_gpu_expert_sums_not_a_controlled_speed_comparison']
    sources = json.loads((HERE / 'source_manifest.json').read_text())
    assert all(sha(Path(r['path'])) == sha(HERE / r['snapshot']) == r['sha256'] for r in sources)
    manifest_path = HERE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['primary_report'] = 'GROUP_COMPARISON.md'
    manifest['combined_view'] = {
        'eight_gpu': 'All recorded ranks of layers20/50 summed; 16 EP vs8 PP records; not full stages or synchronized latency',
        'eight_gpu_expert_scope': 'Per rank-layer mean EP0.255803ms vs PP0.493588ms. Raw sums4.092843ms vs3.948703ms are not a controlled Expert speed comparison.',
        'two_gpu_prefill': 'Whole global-B8 cohort: EP card mean x2 vs sum PP stages',
        'two_gpu_decode': 'B8-equivalent derived kernel work: EP Graph mean x2 vs 2 slots x sum PP stage Graph means',
    }
    manifest['artifacts'] = list(dict.fromkeys(manifest['artifacts'] +
        ['GROUP_COMPARISON.md', 'two_gpu_combined_group_work.csv', 'eight_gpu_combined_sample_work.csv', 'group_validation.json']))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    short = f'''### 补图：通信和 Expert 的绝对耗时与占比

[PP合并后的主图与对照表]({HERE / 'GROUP_COMPARISON.md'}) · [原逐阶段明细]({HERE / 'REPORT.md'})。

主图已将PP合成一条，EP也按整组累计，避免拿PP两卡之和去比EP每卡均值。两卡Qwen：Prefill按整批8请求的全部GPU内核累计；Decode从每个4请求Graph换算到8请求，正确计入PP两槽、每槽经过两个阶段。百分比**不是端到端独占占比**。Decode的Attention／Expert只识别核心／GEMM，其余保留在其他，PP Graph外边界传输没有被当成零。八卡GLM只找到关联的4请求Prefill层20／50的独立rank分区：EP共16份、PP共8份，分别全部合并。本地新token为EP512／PP1024，含等待，不是完整模型或同步全局轮次。八卡4／16请求Decode和16请求Prefill的匹配模型分项仍缺失，未用纯通信微基准填补。

**八卡路由Expert统计说明：** 每卡每层的平均本地跨度是EP8约0.256 ms、PP约0.494 ms（PP约1.93倍）。图中累计4.093／3.949 ms来自16／8份记录，不能据此说PP Expert更快；本地token、实际路由负载和采样轮次不匹配，约3.52%的合计差也不是受控速度差。

归档保留 6 行两卡分项、24 个八卡原始 rank-layer 样本及其 4 行分层均值、原图 12 行正式请求指标、原始路径和哈希。没有新 GPU 实验，也没有改写历史结果或失败状态。'''
    (HERE / 'index_Q10_followup.md').write_text(short + '\n')
    replace_block(INDEX / 'component_reports/Q10.md', '<!-- BEGIN Q10 COMPONENT SHARES -->',
                  '<!-- END Q10 COMPONENT SHARES -->', short)
    readme_path = INDEX / 'README.md'
    readme = readme_path.read_text()
    rows = readme.splitlines()
    matches = [i for i, row in enumerate(rows) if row.startswith('| [Q10：')]
    assert len(matches) == 1
    i = matches[0]
    target = str(HERE / 'GROUP_COMPARISON.md')
    legacy = str(HERE / 'REPORT.md')
    if legacy in rows[i]:
        rows[i] = rows[i].replace(legacy, target).replace('通信／Expert占比补图', 'PP合并的通信／Expert对比')
        readme_path.write_text('\n'.join(rows) + '\n')
    if target not in rows[i]:
        assert rows[i].endswith(' |')
        rows[i] = rows[i][:-2] + f' **[通信／Expert占比补图]({target})**：两卡Prefill／Decode、八卡Prefill单层；八卡Decode分项缺失。 |'
        readme_path.write_text('\n'.join(rows) + '\n')
    assert readme_path.read_text().count(target) == 1

    registry_text = f'''### {HERE.name}

- Snapshot: {SNAPSHOT}; offline Q10 component followup, zero GPU launches or new measurements.
- H100 2GPU Qwen122 FP8: exact screenshot B8 cold4096->1 Prefill and realGPUcached4096/new1->128 Decode. Six per-card rows, kernel-duration composition only; Decode core Attention/routed GEMM scope is narrower. PP Graph-external boundary not zero-filled.
- H100 8GPU GLM INT4 on174/175: related C4 Prefill diagnostic, cached/processed2048 at layers20/50, EP local512 vs PP1024 new tokens, 24 independently anchored exclusive rank-layer partitions aggregated to four layer/layout means. Not synchronized global steps or full-model E2E shares; failed outer r3 evidence preserved.
- No matching 8GPU Decode compute breakdown (C4/C16), nor C16 Prefill layer profile. Missing data explicitly marked, no independent communication microbenchmark stacked into model proportions.
- User-requested merged PP view: both EP/PP sum all sampled ranks; eight-GPU sums the two sampled layers only (16 EP vs8 PP independent rank-layer records), not full stages or synchronized E2E. Two-GPU Prefill uses whole B8 cohort kernel work; Decode B8-equivalent correctly counts both PP slots through both stages, derived from short Graph means, not a measured global-step latency.
- Expert scope audit: eight-GPU mean per rank-layer is EP0.255803ms vs PP0.493588ms (PP1.93x); raw sums4.092843/3.948703ms multiply16/8 independent records. The3.52% lower PP sum does not establish an Expert speedup; local chunks, routes and anchored global ticks differ.
- Primary report: {HERE / 'GROUP_COMPARISON.md'}; original per-stage report: {HERE / 'REPORT.md'}. PNG/SVG/PDF absolute and relative plots, 12 original formal E2E rows, copied small source records/SHA256, regeneration and arithmetic validation.
'''
    replace_block(REGISTRY, f'<!-- begin {HERE.name} -->', f'<!-- end {HERE.name} -->', registry_text)
    files = ['REPORT.md', 'summary.md', 'manifest.json', 'source_manifest.json', 'validation.json',
             'two_gpu_component_shares.csv', 'eight_gpu_prefill_layer_shares.csv', 'full_metrics.csv',
             'build_report.py', 'archive_report.py', 'index_Q10_followup.md',
             'GROUP_COMPARISON.md', 'build_group_comparison.py', 'group_validation.json',
             'two_gpu_combined_group_work.csv', 'eight_gpu_combined_sample_work.csv']
    files += [f'{stem}.{ext}' for stem in ('two_gpu_component_shares', 'eight_gpu_prefill_layer_shares',
                                         'two_gpu_combined_comparison', 'eight_gpu_combined_comparison')
              for ext in ('png', 'svg', 'pdf')]
    files += [r['snapshot'] for r in sources]
    for name in files:
        source, target = HERE / name, SNAPSHOT / name
        assert source.stat().st_size < 4 * 1024 * 1024
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    checksums = ''.join(f'{sha(HERE / name)}  {name}\n' for name in sorted(files))
    (HERE / 'checksums.sha256').write_text(checksums)
    (SNAPSHOT / 'checksums.sha256').write_text(checksums)
    print(json.dumps(dict(snapshot=str(SNAPSHOT), verified_files=len(files), index_entry='Q10 followup', new_gpu_runs=0)))


if __name__ == '__main__':
    main()
