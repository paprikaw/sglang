"""Offline model, figures, evidence extracts and reproducible report. No GPU use."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import (A100, H100, uniform_activation, expert_weight_bytes,
                   communication_time, critical_weight_bytes,
                   critical_expert_count, critical_network_bytes_s, layer_times)

ROOT = Path(__file__).resolve().parent
CWD = Path("/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling")
ARCHIVES = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments")
HF = Path("/data/gpfs/projects/punim2715/huggingface")
SOURCES = set()


def read_json(path):
    p = Path(path).resolve(); SOURCES.add(p)
    return json.loads(p.read_text())


def read_csv(path):
    p = Path(path).resolve(); SOURCES.add(p)
    return list(csv.DictReader(p.open()))


def write_json(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(name, rows):
    with (ROOT/name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


if (ROOT/"official_sources.json").exists():
    read_json(ROOT/"official_sources.json")

models = []
for name, precision, size in [("Qwen3.6-35B-A3B", "BF16", 2),
                              ("Qwen3.5-122B-A10B-FP8", "FP8", 1),
                              ("GLM-4.7-AWQ", "INT4", .5)]:
    config = read_json(HF/name/"config.json")
    c = config.get("text_config", config)
    E = c.get("num_experts", c.get("n_routed_experts"))
    we = expert_weight_bytes(c["hidden_size"], c["moe_intermediate_size"], size)
    models.append(dict(model=name, precision=precision, hidden=c["hidden_size"],
                       intermediate=c["moe_intermediate_size"], experts=E,
                       top_k=c["num_experts_per_tok"], layers=c["num_hidden_layers"],
                       bytes_per_expert=we, layer_weight_bytes=E*we))
write_csv("model_geometry.csv", models)
q35 = models[0]

# Conditional boundaries: r=.8 is the identical-active-set, 80%-union case.
thresholds = []
for hw in [A100, H100]:
    for dc_us in [5, 10, 30, 34, 50, 100, 300]:
        for r in [.05, .2, 2*(1-math.sqrt(.2))-.8, .8]:
            thresholds.append(dict(hardware=hw.name, hbm_TB_s=hw.hbm_bytes_s/1e12,
                dc_us=dc_us, overlap_fraction=r,
                critical_total_layer_weight_GB=critical_weight_bytes(hw.hbm_bytes_s,dc_us*1e-6,r)/1e9,
                critical_overlap_weight_MB=2*hw.hbm_bytes_s*dc_us*1e-6/1e6,
                critical_shared_experts_35b=critical_expert_count(hw.hbm_bytes_s,dc_us*1e-6,q35["bytes_per_expert"])))
write_csv("thresholds.csv", thresholds)

# Independent uniform routing instance: 52 total tokens gives 80.8% union.
# kappa=8: no destination dedup, k/2 remote assignments/token and two operations.
# kappa=1: ideal per-destination token dedup/aggregation approximation.
activation = uniform_activation(256, 8, 26)
network = []
for hw in [A100, H100]:
    for kappa in [1, 8]:
        payload = 26 * 2048 * 2 * kappa
        for n in [5, 12.5, 25, 50, 150, 300]:
            for fixed_us in [0, 30]:
                dc = communication_time(payload, n*1e9, fixed_us*1e-6)
                network.append(dict(hardware=hw.name, B=52, b=26, E=256, k=8,
                    union_fraction=activation["union_fraction"], overlap_fraction=activation["overlap_fraction"],
                    kappa=kappa, logical_payload_bytes=payload, one_way_GB_s=n,
                    fixed_us=fixed_us, dc_us=dc*1e6,
                    critical_total_layer_weight_GB=critical_weight_bytes(hw.hbm_bytes_s, dc, activation["overlap_fraction"])/1e9,
                    qwen35b_winner_in_memory_model=("PP2" if q35["layer_weight_bytes"] < critical_weight_bytes(hw.hbm_bytes_s,dc,activation["overlap_fraction"]) else "EP2")))
write_csv("network_scenarios.csv", network)

# HBM and network thresholds at one common, explicitly hypothetical message size.
limits=[]
for m in models:
    for hw in [A100, H100]:
        for r in [.05, .2, .8]:
            for fixed_us in [0, 30]:
                ncrit=critical_network_bytes_s(m["layer_weight_bytes"],r,hw.hbm_bytes_s,1024**2,fixed_us*1e-6)
                limits.append(dict(model=m["model"],hardware=hw.name,overlap_fraction=r,
                    overlap_expert_count=r*m["experts"],
                    extra_pp_weight_us=m["layer_weight_bytes"]*r/(2*hw.hbm_bytes_s)*1e6,
                    dc_payload_bytes=1024**2,fixed_us=fixed_us,
                    critical_one_way_GB_s=(ncrit/1e9 if math.isfinite(ncrit) else "all_bandwidths")))
write_csv("model_network_limits.csv",limits)

steady=ARCHIVES/"qwen35b_a100_h100_pp_ep_steady_decode_20261026"
components=read_csv(steady/"data/components.csv")
formal=read_csv(steady/"full_metrics.csv")
measured=[]
for hw in ["A100","H100"]:
    ep=next(r for r in components if r["hardware"]==hw and r["arm"]=="ep2" and r["execution"]=="native")
    pp=next(r for r in components if r["hardware"]==hw and r["arm"]=="pp2" and r["execution"]=="native")
    key="expert.routed" if hw=="A100" else "expert.routed_gemm"
    dc=(float(ep["communication.dispatch"])+float(ep["communication.combine"]))*1000
    for arm,row in [("ep2",ep),("pp2",pp)]:
        f=next(x for x in formal if x["hardware"]==hw and x["arm"]==arm)
        measured.append(dict(cohort="35B native historical",hardware=hw,arm=arm,
            expert_us=float(row[key])*1000, dc_us=(dc if arm=="ep2" else 0),
            component_scope=("routed module" if hw=="A100" else "routed GEMM"),
            tpot_ms=float(f["tpot_ms"]),output_tok_s=float(f["output_tokens_per_s"]),
            phase="Decode",B=8,b=4,P=4096,Q=1,O=128,
            note="A100 eager/normal vs H100 Graph/LL; not hardware-only"))

latest=CWD/"experiments/h100_pp_graph_gap_opt_20261027/analysis/shared_gemm_final_r8_steady/results.json"
cur=read_json(latest)
for arm,policy in [("ep2",0),("pp2",0),("pp2",9)]:
    c=[r for r in cur["components"] if r["arm"]==arm and r["policy"]==policy]
    f=next(r for r in cur["formal_metrics"] if r["arm"]==arm and r["policy"]==policy)
    measured.append(dict(cohort="35B shared GEMM / "+str(policy),hardware="H100",arm=arm,
        expert_us=mean(c,"expert.routed_gemm")*1000,
        dc_us=(mean(c,"communication.dispatch")+mean(c,"communication.combine"))*1000 if arm=="ep2" else 0,
        component_scope="routed GEMM", tpot_ms=f["tpot_ms"],output_tok_s=f["output_tok_s"],
        phase="Decode",B=8,b=4,P=4096,Q=1,O=128,
        note="drop20/select16 independent profile; formal5 cohorts; policy9 PP-specific lookahead"))
write_csv("measured_comparison.csv",measured)

# 122B short diagnostic, kept separate from 35B steady profile.
q122rows=read_csv(CWD/"experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7/analysis/component_followup/comparison.csv")
short=[]
for r in q122rows:
    if r["case"]=="moderate":
        short.append(dict(model="Qwen3.5-122B-A10B-FP8",arm=r["arm"],stage=r["stage"],
            expert_us=float(r["routed_gemm_ms"])/int(r["layers"])*1000,
            dc_us=(float(r["dispatch_ms"])+float(r["combine_ms"]))/int(r["layers"])*1000,
            scope="1-2 Graph/rank diagnostic; sums can overlap; not E2E",B=8,b=4,P=4096,Q=1,O=128))
write_csv("122b_short_diagnostic.csv",short)

# Routing snapshots are not the same window as complete-trace component means.
# Store them as snapshots; do not infer cross-slot overlap from unrelated runs.
fair=CWD/"experiments/h100_ep_pp_fair_expert_20261026"
routes=read_json(fair/"analysis/ep_routing.json")
snapshots=[]
for run in ["fair_native_20261026_v3","fair_triton_20261026_v1"]:
    rr=[r for r in routes if r["run"]==run]
    snapshots.append(dict(run=run,arm="ep2",samples=len(rr),active_per_invocation=mean(rr,"active_experts"),
        scope="one routed B4 snapshot per layer/rank; remote assignments included"))
    rr=[]
    for p in (fair/"runs"/run/"pp2/routing").glob("*.json"):
        d=json.loads(p.read_text())
        if d.get("phase")=="D" and d.get("tokens")==4:
            SOURCES.add(p.resolve());rr.append(d)
    snapshots.append(dict(run=run,arm="pp2",samples=len(rr),active_per_invocation=mean(rr,"active_experts"),
        scope="first routed B4 snapshot per layer; not two synchronized slots"))
write_csv("routing_snapshots.csv",snapshots)
headroom=read_json(CWD/"experiments/h100_pp_expert_fragmentation_20261026/deliverables/headroom_estimate.json")

# Newer controlled workload rerun: all three layouts use the same kernel family.
workload_root=ARCHIVES/"qwen35b_h100_pp_lookahead_workloads_20261027"
wcomp=read_csv(workload_root/"analysis/components.csv")
wmetrics=read_csv(workload_root/"analysis/metrics.csv")
read_json(workload_root/"manifest.json")
read_json(ARCHIVES/"qwen35b_h100_expert_workload_validation_20261027/manifest.json")
route_summary=read_csv(ARCHIVES/"qwen35b_h100_expert_workload_validation_20261027/analysis/routing_summary.csv")
workload_rows=[]
for workload in ["b8_repeated","b16_repeated","b32_repeated","b32_diverse"]:
    ce=[x for x in wcomp if x["workload"]==workload and x["arm"]=="ep2"]
    cp=[x for x in wcomp if x["workload"]==workload and x["arm"]=="pp2" and x["policy"]=="9"]
    fe=next(x for x in wmetrics if x["workload"]==workload and x["arm"]=="ep2" and x["phase"]=="decode")
    fp=next(x for x in wmetrics if x["workload"]==workload and x["arm"]=="pp2" and x["policy"]=="9" and x["phase"]=="decode")
    ee=mean(ce,"expert.routed_gemm")*1000;pe=mean(cp,"expert.routed_gemm")*1000
    comm=(mean(ce,"communication.dispatch")+mean(ce,"communication.combine"))*1000
    workload_rows.append(dict(workload=workload,B=int(fe["global_requests"]),b=int(fe["local_tokens"]),
        ep_expert_us=ee,pp_expert_us=pe,pp_extra_expert_us=pe-ee,dc_kernel_us=comm,
        kernel_budget_remainder_us=comm-(pe-ee),
        ep_tpot_ms=float(fe["tpot_ms"]),pp_tpot_ms=float(fp["tpot_ms"]),
        ep_output_tok_s=float(fe["throughput_tok_s"]),pp_output_tok_s=float(fp["throughput_tok_s"]),
        pp_throughput_gain_percent=100*(float(fp["throughput_tok_s"])/float(fe["throughput_tok_s"])-1),
        ep_ttft_ms=float(fe["ttft_ms"]),pp_ttft_ms=float(fp["ttft_ms"]),
        scope="lookahead_r2; profile drop20/select16; same generic T<=16 GEMM; kernel budget is not exposed critical-path time"))
write_csv("workload_model_check.csv",workload_rows)
write_csv("newer_routing_snapshots.csv",route_summary)

# Model verification against exhaustive explicit routing in a small case.
cases=0;su=0;sr=0
for route in itertools.product(range(4),repeat=4):
    a=set(route[:2]);b=set(route[2:]);u=a|b;r=a&b
    assert len(a)+len(b)-len(u)==len(r)
    times=layer_times(4,len(a)/4,len(b)/4,len(u)/4,1,0)
    assert math.isclose(times["extra_pp_weight_seconds"],len(r)/2)
    cases+=1;su+=len(u)/4;sr+=len(r)/4
expect=uniform_activation(4,1,2)
assert math.isclose(su/cases,expect["union_fraction"])
assert math.isclose(sr/cases,expect["overlap_fraction"])
assert layer_times(100,.4,.4,.8,1e3,1)["extra_pp_weight_seconds"]==0
assert math.isclose(layer_times(100,.8,.8,.8,1e3,1)["extra_pp_weight_seconds"],.04)

ep=next(x for x in measured if x["cohort"]=="35B shared GEMM / 0" and x["arm"]=="ep2")
pp=next(x for x in measured if x["cohort"]=="35B shared GEMM / 9")
pp0=next(x for x in measured if x["cohort"]=="35B shared GEMM / 0" and x["arm"]=="pp2")
dc=ep["dc_us"]
bound=next(x for x in cur["independent_profile"] if x["arm"]=="pp2" and x["policy"]==9)
verdict=dict(
    status="offline_model_no_new_GPU_run",latest_dc_us=dc,
    latest_expert_penalty_us=pp["expert_us"]-ep["expert_us"],
    latest_component_budget_saving_us=ep["expert_us"]+dc-pp["expert_us"],
    component_score_is_not_additive_e2e=True,
    max_shared_experts_B4_top8=32,
    latest_H100_critical_shared_experts=critical_expert_count(H100.hbm_bytes_s,dc*1e-6,q35["bytes_per_expert"]),
    maximum_B4_weight_penalty_us=32*q35["bytes_per_expert"]/(2*H100.hbm_bytes_s)*1e6,
    latest_PP_tpot_advantage_percent=100*(1-pp["tpot_ms"]/ep["tpot_ms"]),
    latest_PP_output_advantage_percent=100*(pp["output_tok_s"]/ep["output_tok_s"]-1),
    conditional_profile_pipeline_bound_ms=bound["conditional_compute_bound_ms"],
    profile_cycle_ms=bound["cycle_ms"],
    conditional_profile_bound_attainment=bound["conditional_compute_bound_ms"]/bound["cycle_ms"],
    conditional_bound_over_formal_tpot=bound["conditional_compute_bound_ms"]/pp["tpot_ms"],
    old_route_sample_HBM_us=headroom["roofline"]["ideal_HBM_read_us"],
    old_route_sample_measured_expert_us=next(x for x in headroom["comparisons"] if x["configuration"]=="PP2 current")["expert_us"],
    activation_52=activation,
    newer_workload_rows=workload_rows,
    routing_overlap_measured=False,
    actual_DRAM_traffic_measured=False)
write_json("calibration.json",verdict)

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.facecolor":"white","axes.titleweight":"bold"})
fig,axs=plt.subplots(1,2,figsize=(13.5,5.5))
x=np.linspace(0,105,300)
for hw,color in [(A100,"#cf8a2f"),(H100,"#15958a")]:
    axs[0].plot(x,2*hw.hbm_bytes_s*x*1e-6/1e6,color=color,lw=2.5,label=hw.name)
axs[0].set(xlim=(0,105),ylim=(0,730),xlabel="Exposed dispatch + combine time (us / layer)",
    ylabel="Break-even shared active weight (MB)",title="PP2 can afford more redundant weight reads")
axs[0].text(55,100,"PP2 favored below each line",color="#176c63")
axs[0].text(6,560,"EP2 favored above each line",color="#6d4d36")
axs[0].scatter([dc],[32*q35["bytes_per_expert"]/1e6],color="#394b72",zorder=5)
axs[0].annotate("Current B4 top-8: overlap <= 32 Experts",(dc,32*q35["bytes_per_expert"]/1e6),
    xytext=(7,330),arrowprops=dict(arrowstyle="->",color="#394b72"),fontsize=9)
axs[0].legend(loc="upper left",bbox_to_anchor=(0,1.00));axs[0].grid(alpha=.15)
n=np.geomspace(5,350,250)
for hw,color in [(A100,"#cf8a2f"),(H100,"#15958a")]:
    for fixed_us,ls in [(0,"--"),(30,"-")]:
        w=2*hw.hbm_bytes_s*(fixed_us*1e-6+(26*2048*2*8)/(n*1e9))/activation["overlap_fraction"]
        axs[1].plot(n,w/1e9,color=color,ls=ls,lw=2,label=f"{hw.name.split(' 80GB')[0]}: fixed {fixed_us} us")
axs[1].axhline(q35["layer_weight_bytes"]/1e9,color="#394b72",lw=1.5,label="35B BF16: 1.611 GB/layer")
axs[1].set(xscale="log",yscale="log",xlabel="Effective one-way send bandwidth per GPU (GB/s)",
    ylabel="Break-even total Expert weights / layer (GB)",title="52 tokens: ~81% active across two slots")
axs[1].grid(alpha=.15);axs[1].legend(fontsize=8,loc="lower left")
fig.suptitle("EP2 vs PP2: extra HBM reads versus Expert exchange",fontsize=17)
fig.text(.5,.015,"Memory-only model; balanced stages; one GPU/stage. Right: no destination dedup (kappa=8); PP2 favored below curves.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.93))
for ext in ["png","pdf"]:fig.savefig(ROOT/f"bandwidth_boundary.{ext}",dpi=170)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.8))
xx=np.arange(4);width=.34
labels=["B8 repeated","B16 repeated","B32 repeated","B32 diverse"]
for ax in axs:ax.set_xticks(xx,labels,fontsize=9);ax.set_axisbelow(True);ax.grid(axis="y",alpha=.15)
axs[0].bar(xx-width/2,[r["dc_kernel_us"] for r in workload_rows],width,color="#9162bd",label="EP2 dispatch + combine")
axs[0].bar(xx+width/2,[r["pp_extra_expert_us"] for r in workload_rows],width,color="#cf8a2f",label="Extra PP2 Expert GEMM")
axs[0].set(ylabel="Kernel service time (us / layer)",title="Expert cost consumes the communication saving",ylim=(0,49))
axs[0].legend(fontsize=8,loc="upper left")
axs[1].bar(xx-width/2,[r["ep_tpot_ms"] for r in workload_rows],width,color="#15958a",label="EP2")
axs[1].bar(xx+width/2,[r["pp_tpot_ms"] for r in workload_rows],width,color="#394b72",label="PP2 + lookahead")
axs[1].set(ylabel="TPOT (ms), unprofiled requests",title="At B32, EP2 has lower Decode TPOT",ylim=(0,15))
axs[1].legend(fontsize=8,loc="upper left")
fig.suptitle("Same H100 pair and Expert kernel: workload changes the decision",fontsize=16)
fig.text(.5,.015,"GPU-cached 4096 + new 1 -> 128; five cohorts/config. Left: independent profile, nonadditive. PP cohort throughput ties at B32 due to lower TTFT.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.92))
for ext in ["png","pdf"]:fig.savefig(ROOT/f"workload_tradeoff.{ext}",dpi=170)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.8))
labels=["EP2","PP2 shared GEMM","PP2 + lookahead"]
data=[ep,pp0,pp]
axs[0].bar(labels,[d["expert_us"] for d in data],color=["#15958a","#cf8a2f","#cf8a2f"],label="Expert GEMM")
axs[0].bar(labels,[d["dc_us"] for d in data],bottom=[d["expert_us"] for d in data],color="#9162bd",label="Dispatch + combine")
for i,d in enumerate(data):
    axs[0].text(i,d["expert_us"]+d["dc_us"]+1.5,f"{d['expert_us']+d['dc_us']:.1f}",ha="center")
axs[0].set(ylabel="Kernel-time service score (us / layer)",title="Expert + exchange favors PP2")
axs[0].legend(fontsize=9);axs[0].set_ylim(0,105)
axs[1].bar(labels,[d["tpot_ms"] for d in data],color=["#15958a","#cf8a2f","#394b72"])
for i,d in enumerate(data):axs[1].text(i,d["tpot_ms"]+.13,f"{d['tpot_ms']:.2f}",ha="center")
axs[1].axhline(bound["conditional_compute_bound_ms"],color="#777777",ls="--",label="PP2 profile stage bound: 6.48 ms")
axs[1].set(ylabel="TPOT (ms), unprofiled requests",title="PP2 needs scheduling to realize the gain",ylim=(0,11))
axs[1].legend(fontsize=8)
for ax in axs:ax.tick_params(axis="x",labelsize=9);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
fig.suptitle("H100 35B B8 scheduling control: shared GEMM in EP2 and PP2",fontsize=16)
fig.text(.5,.015,"B=8, two B4 slots; GPU-cached 4096 + new 1 -> 128. Left: independent profile, sums can overlap; not E2E slices.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.92))
for ext in ["png","pdf"]:fig.savefig(ROOT/f"measured_check.{ext}",dpi=170)
plt.close(fig)

def networkrow(hw,n,fixed,kappa=8):
    return next(r for r in network if r["hardware"]==hw.name and r["one_way_GB_s"]==n and r["fixed_us"]==fixed and r["kappa"]==kappa)

lines=[
"# EP/PP：Expert 权重读取与 dispatch/combine 的简化模型", "",
"日期：2026-10-27。离线解析建模与既有结果复核；本次没有新启动 GPU 实验。",
"",
"核心判据：**PP 重复读取权重的时间，是否小于 EP 逐层交换 token 的时间。** 模型使用逐层活跃权重，而非模型总参数量。",
"",
"## 1. 公平比较的最小设置", "",
"固定两张 GPU。EP2 为 Attention DP2×TP1、Expert EP2，每卡全部 L 层；PP2 为两个一 GPU stage，每卡 L/2 层。固定总 batch B=2b，PP 用两个 b-token 微批次，考察平衡流水线的稳态整批周期。Attention、CPU、PP 边界、共享 Expert、量化排列暂不放入核心式子。这里的周期可近似稳态 Decode TPOT；不是冷启动首批流水线延迟。",
"",
"设每个 routed Expert 的 gate/up/down 权重为 w=3h i s_w；每层 E 个 Expert，总权重 W=Ew。两个微批次激活的 Expert 集合为 A1、A2，大小分别 M1、M2，并集 U，交集 R=M1+M2-U。H 是每 GPU 有效 HBM 读取带宽；C 是暴露在临界路径上的每层 dispatch+combine 时间。",
"",
"```text",
"T_EP / L = U*w/(2H) + C",
"T_PP / L = (M1+M2)*w/(2H)",
"PP 多出的权重读取时间 = R*w/(2H)",
"PP 更好 <=> R*w/(2H) < C",
"```", "",
"等式取平均且假设 Expert 均衡分片、PP stage 平衡。实际应按 rank/stage 的最大服务时间计，不能把两卡均值当作严格下界。PP 每卡常驻完整层的所有 Expert，不等于每一步都会读取不活跃 Expert 的矩阵；实际还可能重复加载、存在 L2 命中及 tile/launch 开销。",
"",
"## 2. 为什么只给“激活 80%”还不够", "",
"同样是整批共激活 80% Expert：若两个微批次各激活相同的 80%，R/E=0.8，PP 的活跃权重读取是 EP 的 2 倍；若两个微批次各激活互不重叠的 40%，R=0，稳态每 GPU 读取量相同。EP 的收益来自把重复读取合到一次，而不只是“每卡只放一半 Expert”。",
"",
"独立均匀 top-k 路由可用 a(b)=1-(1-k/E)^b，u=2a-a²，r=a² 估算。若整批并集 u=0.8，则 a=0.5528、r=0.3056，PP 比 EP 多读约 38.2%，而不是一律两倍。真实路由相关性很强，必须用真实集合。",
"",
"当前 B8 / b4 / top8 的每个微批次最多激活 32/256=12.5%，整批最多 25%，因此“激活 80%”不是当前 Decode 的实际形状。保存的路由快照有约 10–11 个活跃 Expert/次，但并非同步记录两个 slot，不能据此声称已测得 R。",
"",
"## 3. A100/H100 下的阈值", "",
"GPU 标称 HBM：A100 80GB PCIe 为 1.935 TB/s；H100 80GB SXM 为 3.35 TB/s。以下按理想一次读取估算；真实部署应把 H 换成同 shape、同 kernel 的有效带宽。",
"",
"令 r=R/E：W_crit=2HC/r；W<W_crit 时 PP 更有利。先取两个微批次重复激活相同 80%（r=0.8）：", "",
"| 每层 dispatch+combine | A100：PP 有利的总 Expert 权重/层 | H100：PP 有利的总 Expert 权重/层 |",
"|---|---:|---:|",
]
for us in [10,30,100]:
    lines.append(f"| {us} µs | < {critical_weight_bytes(A100.hbm_bytes_s,us*1e-6,.8)/1e6:.1f} MB | < {critical_weight_bytes(H100.hbm_bytes_s,us*1e-6,.8)/1e6:.1f} MB |")
lines += ["", "若整批共激活 80%、按独立均匀路由 r=0.3056，以上门槛约放宽 2.62 倍。例如 C=30 µs 时，A100/H100 为约 380/658 MB/层。若 HBM 有效带宽仅峰值一半，门槛也减半。更大的逐层权重或更多重复活跃 Expert 偏向 EP；更高 HBM、较大 C、较少重复活跃 Expert 偏向 PP。", "",
"等价地，PP 容许的重复活跃 Expert 数为 R_crit=2HC/w。以 35B BF16 每 Expert 6 MiB 为例：", "",
"| 每层 dispatch+combine | A100：R_crit | H100：R_crit |", "|---|---:|---:|"]
for us in [10,30,100]:
    lines.append(f"| {us} µs | {critical_expert_count(A100.hbm_bytes_s,us*1e-6,q35['bytes_per_expert']):.1f} | {critical_expert_count(H100.hbm_bytes_s,us*1e-6,q35['bytes_per_expert']):.1f} |")
lines += ["", "## 4. 把通信带宽放进模型", "",
"用 C=λ+V/N。λ 包括 dispatch/combine 固定协议、启动、通知与不可隐藏等待；V 是每 GPU 两个 operation 的实际串行发送字节；N 是同路径的单向有效发送带宽。接收可与发送全双工，不能再把字节重复加一遍。若 kernel 还占 SM/争用 HBM，需将这部分暴露成本留在 C。",
"", "```text", "PP 更好 <=> rW/(2H) < λ + V/N",
"若 rW/(2H) > λ：PP 更好 <=> N < V / (rW/(2H)-λ)",
"若 rW/(2H) <= λ：仅 Expert+交换模型已偏向 PP，与大消息带宽无关。", "```", "",
"无目标 rank 去重时，EP2 每个 rank 的 D+C 发送量约 V=b*k*h*s_a；若先按目标 GPU 合并，可小很多。下面用 E256/top8、B52、b26、h2048、BF16 activation：并集激活 80.8%、重复比例约 31.6%，无去重 V=832 KiB。固定开销 30 µs 只是敏感性假设，不是已测得的截距。", "",
"| N：单向有效 GB/s | C，λ=0 / 30 µs | A100 W_crit，λ=0 / 30 | H100 W_crit，λ=0 / 30 |", "|---|---:|---:|---:|"]
for n in [150,25,12.5]:
    a0=networkrow(A100,n,0); a1=networkrow(A100,n,30);h0=networkrow(H100,n,0);h1=networkrow(H100,n,30)
    lines.append(f"| {n:g} | {a0['dc_us']:.1f} / {a1['dc_us']:.1f} µs | {a0['critical_total_layer_weight_GB']:.3f} / {a1['critical_total_layer_weight_GB']:.3f} GB | {h0['critical_total_layer_weight_GB']:.3f} / {h1['critical_total_layer_weight_GB']:.3f} GB |")
lines += ["", "例如该假设下 N=12.5 GB/s、λ=30 µs，H100 门槛约 2.08 GB/层，35B BF16 的 1.61 GB/层偏向 PP；A100 门槛约 1.20 GB/层，同模型偏向 EP。**这是可构造的同通信条件硬件反转，不是对现有 A100/H100 实测的因果解释。** 采用目标 GPU 去重近似（κ=1）时，H100 同一例的门槛降至约 0.82 GB，选择也可能反转；所以 profiler 必须记录实际流量/通信时间。所有敏感性格点见 CSV。", "",
"实际已存双卡拓扑是 A100069 NV12、H100178 NV6。官方每 link 单方向 25 GB/s，对应聚合直连发送上界约 300/150 GB/s；实际有效带宽必须测。产品页 600/900 GB/s 是所有链路双向总和，不能直接填作该双卡发送带宽。跨节点同理，100/200/400 Gb/s 端口的线速上界是 12.5/25/50 GB/s，但不是应用实测。", "",
"## 5. 哪些模型与 setup 值得用 PP", "",
"下表仅计 routed Expert 矩阵，排除共享 Expert、Attention、量化 scale、padding；不能按 checkpoint 总参数直接排序。", "",
"| 模型/权重精度 | 每 Expert | 每层全部 routed Expert | 重复 R=32 时 PP 多读时间，A100/H100 |", "|---|---:|---:|---:|"]
for m in models:
    costs=[32*m['bytes_per_expert']/(2*hw.hbm_bytes_s)*1e6 for hw in [A100,H100]]
    lines.append(f"| {m['model']} / {m['precision']} | {m['bytes_per_expert']/1024**2:.2f} MiB | {m['layer_weight_bytes']/1e9:.3f} GB | {costs[0]:.1f} / {costs[1]:.1f} µs |")
lines += ["", "R=32 是共同假设，不是各模型的已测路由。PP 最值得评估的是低 token、Expert 权重较小或量化后的 Decode，以及交换协议/跨域较慢的 setup；尤其当 R 很小，EP 能省的重复读取本来就有限。高重复活跃比例、大 Expert 矩阵、高效低延迟通信更支持 EP。单模型总权重需加 KV/runtime 后适配 L/2 层/stage；GLM-4.7-AWQ 不能仅凭逐层表格认为可在两张 80GB 上 PP2，须增加 stage 或混合并行。", "",
"Prefill token 多时，必须先检查 compute-bound：每 Expert t 个 assignment 的 AI≈2t/s_w FLOP/byte。BF16 小 t 的 Decode 容易受权重读取和 kernel 粒度限制；Prefill 的高复用可能由计算吞吐主导，此时比较 max(权重字节/H, 有效 FLOPs/F_effective)+固定开销，不能直接沿用纯带宽门槛。", "",
"## 6. 既有数据是否实现了模型给出的优势", "",
"本节先复核 shared_gemm_final_r8 的固定 B8 调度对照；第6.1节加入后续 lookahead_r2 的多负载重新测量，保留两轮来源，不混成一次实验。", "",
"以下仅列每卡实际 layer invocation 的 Expert 与交换计时。它们是服务成本诊断，不是可相加的端到端切片；跨流时 kernel 时长会重叠，C 的临界路径暴露比例还需 dependency trace/消融校准。A100 routed module 与 H100 GEMM 的范围也不同。", "",
"| 同一实验内的对照 | EP Expert | EP D+C | PP Expert | PP Expert 额外代价 |", "|---|---:|---:|---:|---:|"]
for hw in ["A100","H100"]:
    e=next(x for x in measured if x['hardware']==hw and x['arm']=='ep2' and x['cohort']=='35B native historical')
    p=next(x for x in measured if x['hardware']==hw and x['arm']=='pp2' and x['cohort']=='35B native historical')
    lines.append(f"| {hw} 35B 原生 | {e['expert_us']:.2f} µs | {e['dc_us']:.2f} µs | {p['expert_us']:.2f} µs | {p['expert_us']-e['expert_us']:.2f} µs |")
lines.append(f"| H100 35B 双方共享优化，PP lookahead | {ep['expert_us']:.2f} µs | {dc:.2f} µs | {pp['expert_us']:.2f} µs | {pp['expert_us']-ep['expert_us']:.2f} µs |")
lines += ["", f"当前 H100 的小形状：b4/top8 使 R≤32，理想额外权重成本最多 {verdict['maximum_B4_weight_penalty_us']:.2f} µs；实测 D+C kernel 预算为 {dc:.2f} µs，折算 R_crit≈{verdict['latest_H100_critical_shared_experts']:.1f}。若这些交换成本确实暴露，纯 HBM 模型倾向 PP。实测 GEMM 的 PP 额外代价为 {verdict['latest_expert_penalty_us']:.2f} µs，也小于交换预算。这表明 PP 的模型内优势已有依据，端到端能否兑现还取决于调度。", "",
"| 当前同双卡、相同通用 GEMM 优化 | TPOT ↓ | 输出吞吐 ↑ |", "|---|---:|---:|"]
for label,d in zip(["EP2","PP2 原调度","PP2 + lookahead"],[ep,pp0,pp]):lines.append(f"| {label} | {d['tpot_ms']:.3f} ms | {d['output_tok_s']:.1f} tok/s |")
lines += ["", f"PP 优化后比 EP 的 TPOT 低 {verdict['latest_PP_tpot_advantage_percent']:.1f}%，完整请求输出吞吐高 {verdict['latest_PP_output_advantage_percent']:.1f}%。吞吐包含 TTFT/队列尾部，不能由 8/TPOT 直接等同得到。此前 PP 原调度即便 Expert+交换预算较优仍然更慢：PP 额外 CPU/反馈等待占掉了收益。lookahead 主要改变这部分，未再独占一个通用 GEMM 优化。", "",
f"是否达到理论极限：还不能声称达到 HBM 极限。早期 r5 的 11.4 活跃 Expert 路由样本对应一次读取 {headroom['roofline']['ideal_HBM_read_us']:.2f} µs，彼时整模型 profile Expert 均值 {verdict['old_route_sample_measured_expert_us']:.2f} µs，约为模型值 2.69 倍。但路由是样本而非逐 kernel 同步 DRAM 流量，不能把二者比值当成已测 HBM 利用率，也不能拿该样本给最新 66.4 µs trace 定利用率。",
"",
f"调度方面，当前 profile stage span 给出的条件流水线下界 2×max(stage0,stage1)={bound['conditional_compute_bound_ms']:.3f} ms；同 profile 的周期 {bound['cycle_ms']:.3f} ms，达到约 {100*verdict['conditional_profile_bound_attainment']:.1f}% 条件服务率；独立无 profiler TPOT 为 {pp['tpot_ms']:.3f} ms，仅作量级对照。这是“保持现有模型 kernel 不变”的调度下界，和只读 Expert 权重的 HBM 下界不同。",
"",
"122B 的短诊断（每 rank 仅 1–2 Graph）也有类似取舍：EP Expert 49.34 µs/层，D+C 39.87 µs/层；PP 两阶段 Expert 为 86.88 和 76.53 µs/层。PP Expert 多付约 27–38 µs，已接近通信预算，所以 PP 的余量更小，更容易被 stage 不均衡与边界成本吃掉。该轮 EP/PP 用不同 NV6 对且调度未做 35B lookahead，不作为精确 HBM 模型校准。",
"",
"## 6.1 更新的 B8/B16/B32 对照：模型能解释什么", "",
"今天新增的 lookahead_r2 对照覆盖不同 batch 与多样输入，双方均使用同一 Expert 内核家族、T≤16 门限，EP shared overlap 已开启。下面的 Expert/通信来自同一组独立 profile；TPOT 来自另一组无 profiler 正式请求。", "",
"| Workload | PP 多付的 Expert | EP D+C kernel 预算 | 预算余量 | EP / PP TPOT | PP 整批吞吐相对 EP |",
"|---|---:|---:|---:|---:|---:|"]
for r in workload_rows:
    lines.append(f"| {r['workload']} | {r['pp_extra_expert_us']:.2f} µs | {r['dc_kernel_us']:.2f} µs | {r['kernel_budget_remainder_us']:.2f} µs | {r['ep_tpot_ms']:.3f} / {r['pp_tpot_ms']:.3f} ms | {r['pp_throughput_gain_percent']:+.2f}% |")
lines += ["", "B32 从重复文档改成多样输入时，D+C 只增加约 0.74 µs，但 PP 相对 EP 的 Expert 代价增加约 17.9 µs，通信预算几乎被吃尽。它支持“必须结合 Expert 执行特性作决策”；仅用 token 数或网络带宽不够。B32 下 EP 的 TPOT 更低，PP 的整批输出吞吐接近持平，来自 PP 较低的 TTFT；不能以 +0.32%/+0.89% 的整批吞吐声称 PP 稳态 Decode 更好。", "",
"这还不能验证“差额全部由一次权重读取解释”：现有路由诊断取每层第一次匹配调用，包含单 token Extend/早期 Decode，与后期 Graph profile 不同步。例如 B32 diverse 保存的 EP 每卡平均活跃 28.59，而 PP 一次快照平均 14.88，无法视作同一 token 波次的集合来计算交集；那会违反 U≤M1+M2。当前应保留理论条件与实测服务时间两级判断，不能拟合一个虚假的 HBM 利用率。若要把本模型升级为可预测 advisor，只需同步采集两个微批次的 Expert 集合与对应 kernel 时间，再加实际 DRAM bytes 或同形状有效带宽。", "",
"![Workload 对模型的检验](workload_tradeoff.png)", "",
"## 7. 用作 advisor 的最小输入", "",
"1. 从模型读取 w、E、L、精度，检查单 stage 容量。",
"2. 对目标 b 与路由，记录两个微批次的 M1、M2、U，得到 R；不必全 sweep。",
"3. 在相同 backend/Graph 下，记录 T_expert_PP、T_expert_EP 与暴露 D+C；若想拟合带宽项，用两个消息尺寸估 λ、N，并保存目标 rank 去重方式。",
"4. 先比较 Rw/(2H_eff) 与 C 作筛选，再用 ΔT_expert<C 筛选实际 kernel，最后用一组端到端对照确认 PP 调度能否兑现。",
"",
"推广到 G 个一 GPU PP stage 时，对 G 个微批次激活数 M_j、全局并集 U，重复权重代价变为 (sum M_j-U)w/(G H)。仍需另计 stage 不平衡、boundary、有限批次填充/排空与 autoregressive feedback。",
"",
"## 图表与可复现数据", "",
"![HBM 与通信边界](bandwidth_boundary.png)", "",
"![当前实测对照](measured_check.png)", "",
"`model.py` 给出最小函数；`build_report.py` 从原 CSV/配置生成本报告；`thresholds.csv`、`network_scenarios.csv`、`model_network_limits.csv`、`model_geometry.csv` 与 `measured_comparison.csv` 保留全部数值。来源文件路径与 SHA256 见 `sources.json`。", "",
"官方规格/机制来源（2026-10-27 直接核对原页面；web 搜索接口故障，使用官方 HTTPS 读取）：", "",
"- [A100 官方规格](https://www.nvidia.com/en-us/data-center/a100/)：PCIe 1935 GB/s、SXM 2039 GB/s。",
"- [H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)：80GB SXM 3.35 TB/s。",
"- [NVIDIA Hopper 架构](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：每 NVLink 每方向 25 GB/s，H100 18 links / A100 12 links。",
"- [NVIDIA GEMM 性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：算术强度、tile 与 memory/math bound 判据。", ""]
(ROOT/"REPORT.md").write_text("\n".join(lines))
for p in [Path(__file__),ROOT/"model.py"]:SOURCES.add(p.resolve())
write_json("sources.json",[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SOURCES)])
write_json("validation.json",dict(status="passed",explicit_routing_cases=cases,
    expected_uniform_activation_matches_enumeration=True,identical_vs_disjoint_cases_passed=True,
    no_gpu_launch=True,routing_overlap_from_unpaired_snapshots=False,
    actual_DRAM_utilization_claimed=False,figures=["bandwidth_boundary.png","measured_check.png","workload_tradeoff.png"]))
write_json("manifest.json",dict(id="ep_pp_bandwidth_model_20261027",
    title="EP2/PP2 Expert weight bandwidth and dispatch/combine break-even model",
    created_at="2026-10-27",status="complete_offline_analysis",
    implementation={"engine":"SGLang saved experiments; no engine modifications",
        "analysis_source":"model.py and build_report.py, SHA256 in sources.json",
        "measured_source_commit":"870269702285d94bdd926a50cc29ad1a20bcf60c",
        "measured_source_tag":"experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated",
        "frozen_run_manifests":"Referenced source archives; no replacement source tag"},
    deployment={"new_gpu_experiment":False,"new_endpoint":None,"new_slurm_job":None},
    testbed={"analytical_hardware":[A100.__dict__,H100.__dict__],"measured_topologies":{"A100069":"NV12","H100178":"NV6"},
        "EP2":"Attention DP2/TP1, Expert EP2, L layers/GPU",
        "PP2":"two one-GPU stages, L/2 layers/GPU, two b-token slots"},
    workload={"main_measured":"B8: GPU cached4096 + new1 -> output128; no CPU restore",
        "hypothetical_network":"B52, h2048, E256, k8; uniform independent routing; kappa1/8; lambda0/30us"},
    caveats=["Memory-bound steady-state model, not additive E2E decomposition",
        "Logical weight reads are not measured DRAM traffic",
        "Unpaired routing snapshots cannot identify actual cross-slot overlap",
        "Hardware peak != achieved bandwidth; wire payload/dedup changes boundaries",
        "A100 eager/normal vs H100 Graph/LL are not hardware-only controls"],
    key_results=verdict,artifacts={"report":"REPORT.md","source_hashes":"sources.json"}))
print(json.dumps({"status":"complete","root":str(ROOT),"model_geometry":models,"calibration":verdict},ensure_ascii=False,indent=2))
