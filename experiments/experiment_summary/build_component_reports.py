"""Render existing component CSVs and figures as Markdown; no GPU execution."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from component_wording import DESCRIPTIONS, VALUES, label

HERE = Path(__file__).resolve().parent
REPORTS = HERE / 'component_reports'
SPEC = {
 'A01': ('Attention：总长度与缓存前缀', ['total_context','cached_prefix','query_tokens','attention_phase'], 'cold_median_ms'),
 'A02': ('Attention：独立 P×Q', ['cached_context','prefill_tokens'], 'cold_median_ms'),
 'A03': ('Attention：TP/DP 与 workload', ['layout','B_global_requests','P_cached_tokens_per_request','Q_current_tokens_per_request'], 'median_ms'),
 'A04': ('Attention：长 context 下 batch 增长', ['run','layout','B','P','Q'], 'median_ms'),
 'A05': ('Attention：两个请求合批', ['point_id','B','P_i','Q_per_request'], 'median_ms'),
 'A06': ('Attention：batch 和请求组成的余量', ['role','B','P_i','Q_i'], 'median_ms'),
 'E01': ('Expert：本地配对数×活跃专家', ['local_routed_assignments','active_experts'], 'median_ms'),
 'E02': ('Expert：EP8 与 PP2×EP4 单卡形状对照', ['layout','pp_slot','global_logical_tokens','local_experts','active_experts','local_assignments','mode'], 'median_ms'),
 'E03': ('Expert：活跃权重工作集', ['total_local_assignments','active_experts','mode'], 'median_ms'),
 'E04': ('Expert：Hy3 T×E 与布局控制', ['cohort','component','mode','T','E','resident'], 'median_ms'),
 'E05': ('Expert：Hy3 / GLM 双轴倍增', ['model','mode','T','E'], 'median_ms'),
 'C01': ('A100：Attention / Expert 平台', ['stage','kind','timing_mode','B','P','Q','T_unique','active_experts_local','duplication_factor','P_i','Q_i'], 'median_ms'),
 'C02': ('H100：Attention / Expert 平台', ['stage','kind','timing_mode','B','P','Q','T_unique','active_experts_local','duplication_factor','P_i','Q_i'], 'median_ms'),
 'C03': ('A100：Attention / Expert / 通信网格', ['P_cached_context','Q_current_tokens_per_sequence','global_batch_sequences','execution_mode'], 'a_core_span_cuda_ms_median_ms'),
}

def fmt(value):
 if value in ('', None): return '—'
 value=VALUES.get(str(value),value)
 if str(value).startswith('glm47_attention_long_context_batch_'):
  value='第一轮' if str(value).endswith('_r1') else '第二轮'
 try:
  number = float(value)
  if '.' in str(value) or 'e' in str(value).lower(): return f'{number:.4f}'
 except (ValueError, TypeError): pass
 return str(value).replace('|', '\\|').replace('\n', ' ')

def table(rows, keys, headings=None):
 return '\n'.join(['| '+' | '.join(headings or [label(k) for k in keys])+' |', '| '+' | '.join(['---']*len(keys))+' |'] + ['| '+' | '.join(fmt(r.get(k)) for k in keys)+' |' for r in rows])

def pivot(rows, row_key, col_key, metric):
 def sort(values): return sorted(values, key=float)
 columns = sort({r[col_key] for r in rows})
 values = {}
 for r in rows:
  key=(r[row_key], r[col_key])
  assert key not in values, key
  values[key]=r[metric]
 records=[{row_key:x, **{y:values.get((x,y)) for y in columns}} for x in sort({r[row_key] for r in rows})]
 return table(records, [row_key]+columns, [label(row_key)]+[label(col_key)+'：'+y for y in columns])

def primary(rows):
 mode=next((k for k in ['mode','timing_mode'] if k in rows[0]), None)
 return [r for r in rows if not mode or r[mode]=='cuda_graph']

def short_tables(sid, rows, dims, metric):
 selected=primary(rows)
 if sid=='A01': return table(selected,dims+[metric])
 if sid in ('A01','A02','E01'):
  axis={'A01':('total_context','cached_prefix'),'A02':('cached_context','prefill_tokens'),'E01':('local_routed_assignments','active_experts')}[sid]
  return pivot(selected,*axis,metric)
 if sid in ('E03','E04','E05'):
  if sid=='E04': selected=[r for r in selected if r['cohort']=='surface' and r['component']=='expert_core']
  groups=sorted({r.get('model','') for r in selected})
  return '\n\n'.join((('### '+VALUES.get(g,g)+'\n\n') if g else '')+pivot([r for r in selected if r.get('model','')==g], 'total_local_assignments' if sid=='E03' else 'T', 'active_experts' if sid=='E03' else 'E', metric) for g in groups)
 if sid=='C03':
  keys=dims+['a_core_span_cuda_ms_median_ms','e_core_span_cuda_ms_median_ms','dispatch_span_cuda_ms_median_ms','combine_span_cuda_ms_median_ms','comm_span_cuda_ms_median_ms']
  return table(selected,keys)
 if sid in ('C01','C02'):
  # Keep the short view to homogeneous main cells; full controls remain below.
  selected=[r for r in selected if r['kind']=='main']
  dims=[k for k in dims if k not in ['kind','timing_mode','P_i','Q_i']]
 return table(selected,dims+[metric])

def main():
 inventory=json.loads((HERE/'component_sweeps.json').read_text())
 index=(HERE / 'README.md').read_text()
 start=index.index('## 7. 补充：')
 end=index.index('## 8. ',start)
 section=index[start:end].replace('`P`、`Q` 都按每条请求计算','已缓存前缀长度和本轮新增输入量都按每条请求计算').replace('P×Q 热图','缓存前缀与新增输入热图')
 section='\n'.join('**活跃专家数不是常驻专家数。** 在固定常驻权重下，把相同数量的实际 token–专家配对行分到更多专家，不会增加理想有效计算量，但会改变每专家输入行数、补齐、权重复用和调度成本。一行表示一个 token 在一个专家上的一次计算，不是一个请求；同一个 token 命中多个专家时会形成多行。所有扫描均衡分配，不测路由倾斜。' if line.startswith('**E 是活跃专家数') else line for line in section.split('\n'))
 index=index[:start]+section+index[end:]
 REPORTS.mkdir(exist_ok=True)
 receipts=[]
 for ds in inventory['datasets']:
  sid=ds['id']; title,dims,metric=SPEC[sid]
  title,setup,finding,boundary,axes=DESCRIPTIONS[sid]
  if sid=='A03' and 'phase' not in dims: dims=dims+['phase']
  p=Path(ds['primary_csv']['path']); raw=p.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==ds['primary_csv']['sha256']
  rows=list(csv.DictReader(io.StringIO(raw.decode())))
  line=next(l for l in index.splitlines() if l.startswith('| '+sid+'：') or l.startswith(f'| [简短报告 {sid}]'))
  cells=line.strip('|').split('|')
  description='\n\n'.join([setup,finding,boundary])
  directory=Path(ds['report']['path']).parent
  figures=sorted(directory.glob('*.png'))
  if (directory/'analysis').is_dir(): figures+=sorted((directory/'analysis').glob('*.png'))
  body=[f'# {sid} · {title}',f'[返回总索引]({HERE / "README.md"}) · [原始 CSV]({p}) · [原报告]({ds["report"]["path"]})', '## 测了什么与主要结果',description,'## 数据表', axes, '延迟单位为**毫秒，越低越好**。主表为耗时中位数；有 CUDA Graph 时优先展示开启后的数据，关闭时的数据和控制点保留在下方完整统计。`—` 表示未测／不适用，不是 0。不同计时模式、来源复制协议不可混算。',short_tables(sid,rows,dims,metric)]
  stats=[k for k in rows[0] if (k.endswith('_ms') and any(s in k for s in ['mean','median','p99'])) or k=='p99_ms_descriptive']
  if sid=='A06': stats=[k for k in stats if not k.startswith('model_total') and not k.startswith('model_longest') and k!='block_medians_ms']
  body+=['<details>',f'<summary>完整统计与控制点（{len(rows)} 行；平均值／中位数／第99百分位，以原始表实际提供为准）</summary>','',table(rows,list(dict.fromkeys(dims+stats))),'','</details>','## 已有图表']
  if figures:
   legend='旧图保留了归档时的轴标；图中 P 表示每请求已缓存前缀长度，Q 表示每请求本轮处理的输入 token 数，B 表示请求数。'
   if sid=='A01':legend='旧图中 C 表示每请求处理后的总长度，P 表示已缓存的前缀长度；本轮新增输入数为总长度减去已缓存前缀长度。'
   if sid.startswith('E'):legend='旧图中 E 表示活跃专家数；T/S 表示本 GPU 实际执行的 token–专家配对行数，不是请求数。'
   if sid=='E02':legend='旧图中 B 表示全局新增逻辑 token 数，不是请求数；实际配对行数和旧来源复制协议见上文。'
   if sid in ('C01','C02'):legend+=' 专家曲线中的 T 表示全局逻辑 token 数，须结合上文的来源复制协议换算实际配对行数；E 表示每卡活跃专家数。'
   body += [legend]
  if not figures: body+=['该归档没有现成图片；上方已直接列出数据表，未补造热图。']
  for number,fig in enumerate(figures,1): body += [f'### 已有图表 {number}',f'![{title}，归档图表{number}]({fig})']
  body += ['## 口径与来源','这是历史单层／组件测试，不是完整请求的首 token 等待时间或平均每个输出 token 的耗时。已有图只引用，不重新计算或重绘；原始字段和全部精度保留在原始 CSV。',f'原始表：`{len(rows)}` 行；SHA-256：`{ds["primary_csv"]["sha256"]}`。']
  dest=REPORTS/f'{sid}.md'; dest.write_text('\n\n'.join(body)+'\n')
  # Make the report the first entry without removing original CSV/figure links.
  replacement=line.replace('| '+sid+'：',f'| [简短报告 {sid}]({dest}) · ',1)
  parts=replacement.strip('|').split('|')
  if len(parts)==4: replacement='| '+' | '.join([parts[0].strip(),setup,finding,boundary])+' |'
  elif len(parts)==3: replacement='| '+' | '.join([parts[0].strip(),setup+' '+finding,boundary])+' |'
  index=index.replace(line,replacement)
  receipts.append(dict(id=sid,report=str(dest),csv_rows=len(rows),figure_count=len(figures),source_sha256=ds['primary_csv']['sha256']))
 (HERE / 'README.md').write_text(index)
 missing=[]; total=0
 for doc in [HERE / 'README.md',*REPORTS.glob('*.md')]:
  for link in re.findall(r'\]\((/[^)]+)\)',doc.read_text()):
   total+=1
   if not Path(link).exists(): missing.append(link)
 assert not missing,missing
 old_validation=json.loads((HERE/'component_validation.json').read_text())
 for name,expected in old_validation['original_request_metrics_and_profile_hashes_unchanged'].items():
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
 old_validation['readme_sha256']=hashlib.sha256((HERE / 'README.md').read_bytes()).hexdigest()
 old_validation['markdown_absolute_links_checked']=len(re.findall(r'\]\((/[^)]+)\)',index))
 (HERE/'component_validation.json').write_text(json.dumps(old_validation,ensure_ascii=False,indent=2)+'\n')
 result=dict(no_gpu_launch=True,reports=receipts,markdown_links_checked=total,missing_links=missing,source_rows=sum(r['csv_rows'] for r in receipts),figure_references=sum(r['figure_count'] for r in receipts))
 (HERE/'component_reports_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))

if __name__=='__main__': main()
