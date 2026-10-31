"""Index the validated single-layer Expert run as a table row, not an appendix."""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'single_layer_ep4_ep8_matched_20261024/compute_v2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check=json.loads((DATA/'validation.json').read_text())
    assert check['passed'] is True and check['primary_points']==20
    rows=list(csv.DictReader((DATA/'expert_comparison.csv').open()))
    sample={r['model']:r for r in rows if int(r['whole_batch_unique_tokens'])==128}
    row=(f"| [E06：本轮 GLM／Hy3 单层 Expert]({DATA/'REPORT.md'}) · A100 065 GPU0；"
         f"[全部比较表]({DATA/'expert_comparison.csv'})／[曲线]({DATA/'expert_curves.png'}) | "
         "整批唯一 token 从 8 翻倍到 4096；分别对比 EP8 整批／EP4 前后半批，以及相同单次输入。"
         "只用真实一层权重、均衡 top-8 路由，不扫专家数。每卡实际配对数与 T01 通信扫描对齐。 | "
         f"整批128 token、两边每卡均128配对行：GLM 的 EP8／EP4半批均值为 {float(sample['glm']['ep8_whole_ms']):.4f}／{float(sample['glm']['ep4_mean_half_ms']):.4f} 毫秒；"
         f"Hy3 为 {float(sample['hy3']['ep8_whole_ms']):.4f}／{float(sample['hy3']['ep4_mean_half_ms']):.4f} 毫秒。 | "
         "预热 eager 主表；GLM AWQ/Humming、Hy3 BF16/Triton。单GPU模拟一个代表性目的rank，"
         "不含 Dispatch／Combine、Attention、KV或PP调度；不是完整PP吞吐／TPOT。 |")
    joined=DATA.parent/'component_comparison/REPORT.md'
    if joined.exists():
        row=row.replace('／[曲线]',f'／[通信与Expert完整联表]({joined})／[曲线]')
    p=HERE / 'README.md';text=p.read_text()
    if re.search(r'^\| \[E06：',text,re.M):
        text=re.sub(r'^\| \[E06：.*$',lambda _:row,text,flags=re.M)
    else:
        marker='\n\n### 7.3 '
        assert text.count(marker)==1
        text=text.replace(marker,'\n'+row+marker)
    text=text.replace('这里整理的是此前已测的 **14 组归档**，不是新实验，也不是第 2/3 节的完整推理结果。',
                      '这里保留此前的 **14 组归档**，并补入本轮单层 Expert 扫描 **E06**；这些均不是第 2/3 节的完整推理结果。')
    text=text.replace('想把EP8/PP的单卡计算形状对齐看 **E02、E04**',
                      '想把EP8/PP的单卡计算形状对齐看 **E02、E04、E06**')
    text=text.replace('**T01 不包含 Expert 计算；要看已有的真实 Expert 计算，请看第 7.2 节 E02／E04／E05，旧协议不同的数据不能直接相加。**',
                      '**T01 不包含 Expert 计算；本轮按相同路由补测的真实 Expert 计算见第 7.2 节 E06，历史数据见 E02／E04／E05。不同计时边界的组件不能直接相加冒充完整层观测。**')
    p.write_text(text)
    artifacts={name:dict(path=str(DATA/name),sha256=sha(DATA/name)) for name in
               ['REPORT.md','expert_comparison.csv','expert_all_rows.csv','expert_samples.csv',
                'glm_repeat_controls.csv','expert_curves.png','expert_curves.pdf','validation.json']}
    inventory=dict(schema_version=1,datasets=[dict(id='E06',date='2026-10-24',
                   title='Real single-layer Expert matched to T01 routing',models=['GLM-4.7-AWQ','Hy3-BF16'],
                   nodes=['spartan-gpgpu065'],actual_gpu_count=1,gpu_index=0,
                   primary_unique_tokens=[2**n for n in range(3,13)],
                   boundaries=['EP8 whole versus EP4 halves','same unique tokens per call'],
                   real_weights=True,balanced_topk=8,full_model_loaded=False,
                   attention=False,communication=False,primary_mode='eager',
                   primary_comparisons=20,raw_timing_rows=check['raw_rows'],raw_samples=check['raw_samples'],
                   artifacts=artifacts,evidence=check['evidence'])])
    if joined.exists():
        inventory['datasets'][0]['joined_communication_report']=dict(path=str(joined),sha256=sha(joined))
    (HERE/'expert_sweeps.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    missing=[]
    for path in re.findall(r'\]\((/[^)]+)\)',row):
        if not Path(path).exists():missing.append(path)
    assert not missing,missing
    (HERE/'expert_validation.json').write_text(json.dumps(dict(passed=True,id='E06',
        readme_sha256=sha(p),expert_sweeps_sha256=sha(HERE/'expert_sweeps.json'),
        row_in_section='7.2',local_links_checked=len(re.findall(r'\]\((/[^)]+)\)',row)),source_validation=artifacts['validation.json']),indent=2)+'\n')
    print(json.dumps(dict(passed=True,id='E06',primary_comparisons=20)))


if __name__=='__main__':main()
