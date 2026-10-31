"""Small scientific figures and a concise workload-dependent conclusion."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
CASES = ["b8_repeated", "b16_repeated", "b32_repeated", "b32_diverse"]
LABELS = ["4 tokens/GPU\nB8, repeated", "8 tokens/GPU\nB16, repeated",
          "16 tokens/GPU\nB32, repeated", "16 tokens/GPU\nB32, diverse"]
CONFIGS = [("native","ep2","EP2 native","#89cdc5"),
           ("triton_small","ep2","EP2 small-tile","#138f83"),
           ("native","pp2","PP2 native","#efd19b"),
           ("triton_small","pp2","PP2 small-tile","#cc8a2e")]


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def main():
    result = read(ROOT / "analysis/results.json")
    assert not result["partial"] and not result["profiles_skipped"]
    OUT.mkdir(exist_ok=True)
    metrics = {(r["mode"],r["arm"],r["workload"],r["phase"]):r for r in result["metrics"]}
    assert len(metrics) == 32
    def metric(mode,arm,case,phase="decode"):
        return metrics[mode,arm,case,phase]
    def component(mode,arm,case,key):
        values = [r.get(key,0)*1000 for r in result["components"]
                  if (r["mode"],r["arm"],r["workload"],r["phase"]) == (mode,arm,case,"decode")]
        assert len(values)==2
        return st.mean(values)
    def routing(mode,arm,case,key):
        selected=[r for r in result["routing_summary"] if (r["mode"],r["arm"],r["workload"]) == (mode,arm,case)]
        assert len(selected)==1
        return selected[0][key]
    comparisons=[]
    for case in CASES:
        for arm in ("ep2","pp2"):
            a,b=metric("native",arm,case),metric("triton_small",arm,case)
            ga,gb=[component(mode,arm,case,"expert.routed_gemm") for mode in ("native","triton_small")]
            comparisons.append(dict(workload=case,arm=arm,global_requests=a["global_requests"],local_tokens=a["target_local_tokens"],
                native_output_tok_s=a["throughput_tok_s"],optimized_output_tok_s=b["throughput_tok_s"],
                throughput_gain_pct=(b["throughput_tok_s"]/a["throughput_tok_s"]-1)*100,
                native_tpot_ms=a["tpot_ms"],optimized_tpot_ms=b["tpot_ms"],
                native_e2e_ms=a["e2e_ms"],optimized_e2e_ms=b["e2e_ms"],
                native_expert_gemm_us=ga,optimized_expert_gemm_us=gb,expert_reduction_pct=(1-gb/ga)*100,
                native_active_experts=routing("native",arm,case,"active_experts_mean"),
                optimized_active_experts=routing("triton_small",arm,case,"active_experts_mean"),
                native_assignments=routing("native",arm,case,"assignments_mean"),
                optimized_assignments=routing("triton_small",arm,case,"assignments_mean")))
    table(OUT/"comparisons.csv",comparisons)
    decode_components=[]
    for case in CASES:
        for mode,arm,label,color in CONFIGS:
            decode_components.append(dict(workload=case,arm=arm,mode=mode,
                attention_core_us=component(mode,arm,case,"attention.core"),
                expert_gemm_us=component(mode,arm,case,"expert.routed_gemm"),
                dispatch_us=component(mode,arm,case,"communication.dispatch"),
                combine_us=component(mode,arm,case,"communication.combine"),
                ep_index_adapter_us=component(mode,arm,case,"expert.layout_adapter"),
                scope="mean of two ranks, us/GPU/actual layer call; native Decode Graph; nonadditive"))
    table(OUT/"decode_components.csv",decode_components)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.axisbelow":True,"savefig.dpi":180,"pdf.fonttype":42})
    x=np.arange(len(CASES));width=.18
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    for i,(mode,arm,label,color) in enumerate(CONFIGS):
        pos=x+(i-1.5)*width
        y=[metric(mode,arm,c)["throughput_tok_s"] for c in CASES]
        err=[metric(mode,arm,c)["throughput_stdev"] for c in CASES]
        bars=axes[0].bar(pos,y,width,label=label,color=color,yerr=err,capsize=2)
        axes[0].bar_label(bars,fmt="%.0f",fontsize=8,padding=3,rotation=0)
        y=[component(mode,arm,c,"expert.routed_gemm") for c in CASES]
        bars=axes[1].bar(pos,y,width,label=label,color=color)
        axes[1].bar_label(bars,fmt="%.1f",fontsize=8,padding=3)
    for ax in axes:
        ax.set_xticks(x,LABELS);ax.grid(axis="y",alpha=.18)
        ax.margins(y=.16)
    axes[0].set_title("End-to-end Decode throughput (higher is better)")
    axes[0].set_ylabel("Output tokens/s; mean ± cohort SD")
    axes[1].set_title("Routed Expert GEMMs (lower is better)")
    axes[1].set_ylabel("µs / GPU / actual layer call")
    fig.suptitle("Does the same small-tile Expert optimization generalize?",fontsize=18,y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    fig.text(.5,.035,"Same H100 NVLink pair · Qwen3.6-35B-A3B BF16 · cached4096 + new1 → output128\n"
             "Five profiler-off cohorts; independent GPU profiles. EP2: Attention DP2/TP1. PP2: two native slots, microbatch B/2.",
             ha="center",fontsize=10)
    fig.subplots_adjust(top=.79,bottom=.20,wspace=.25)
    for suffix in ("png","pdf"):
        fig.savefig(OUT/f"workload_generalization.{suffix}",bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5.8))
    for i,(mode,arm,label,color) in enumerate(CONFIGS):
        pos=x+(i-1.5)*width
        for ax,key in zip(axes,("active_experts_mean","assignments_per_active_expert")):
            y=[routing(mode,arm,c,key) for c in CASES]
            bars=ax.bar(pos,y,width,label=label,color=color)
            ax.bar_label(bars,fmt="%.0f" if key=="active_experts_mean" else "%.1f",fontsize=8,padding=3)
    for ax in axes:
        ax.set_xticks(x,LABELS);ax.grid(axis="y",alpha=.18);ax.margins(y=.20)
    axes[0].set_title("Actual distinct active experts per GPU/layer")
    axes[0].set_ylabel("Mean active experts in real routed layer samples")
    axes[1].set_title("Token reuse within each active expert")
    axes[1].set_ylabel("Assignments / active expert")
    fig.suptitle("Batch size and prompt diversity change Expert execution geometry",fontsize=17,y=.98)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    fig.text(.5,.025,"First matching routed call per layer in an untimed cohort; not an average over all Decode steps.\n"
             "Assignments are token–expert pairs; active experts are distinct experts touched by those pairs.",
             ha="center",fontsize=10)
    fig.subplots_adjust(top=.78,bottom=.20,wspace=.25)
    for suffix in ("png","pdf"):
        fig.savefig(OUT/f"routing_geometry.{suffix}",bbox_inches="tight")
    plt.close(fig)
    full_table=[]
    expert_table=[]
    phase_table=[]
    for case in CASES:
        for arm in ("ep2","pp2"):
            a,b=metric("native",arm,case),metric("triton_small",arm,case)
            row=next(r for r in comparisons if r["workload"]==case and r["arm"]==arm)
            full_table.append(f"| {case} | {arm.upper()} | {a['throughput_tok_s']:.1f} ± {a['throughput_stdev']:.1f} | {b['throughput_tok_s']:.1f} ± {b['throughput_stdev']:.1f} | {row['throughput_gain_pct']:+.2f}% | {a['tpot_ms']:.3f} → {b['tpot_ms']:.3f} | {a['e2e_ms']:.1f} → {b['e2e_ms']:.1f} |")
            expert_table.append(f"| {case} | {arm.upper()} | {row['native_active_experts']:.2f} → {row['optimized_active_experts']:.2f} | {row['native_assignments']:.1f} → {row['optimized_assignments']:.1f} | {row['native_expert_gemm_us']:.2f} → {row['optimized_expert_gemm_us']:.2f} | {row['expert_reduction_pct']:+.2f}% |")
            a,b=metric("native",arm,case,"prefill"),metric("triton_small",arm,case,"prefill")
            phase_table.append(f"| {case} | {arm.upper()} | {a['throughput_tok_s']:.1f} ± {a['throughput_stdev']:.1f} | {b['throughput_tok_s']:.1f} ± {b['throughput_stdev']:.1f} | {a['e2e_ms']:.1f} → {b['e2e_ms']:.1f} |")
    pp=[r for r in comparisons if r["arm"]=="pp2"]
    ep=[r for r in comparisons if r["arm"]=="ep2"]
    best=[]
    for case in CASES:
        selected={arm:max((metric(mode,arm,case) for mode in ("native","triton_small")),key=lambda r:r["throughput_tok_s"])
                  for arm in ("ep2","pp2")}
        a,b=selected["ep2"],selected["pp2"]
        best.append(dict(workload=case,ep_mode=a["mode"],ep_output_tok_s=a["throughput_tok_s"],
                         pp_mode=b["mode"],pp_output_tok_s=b["throughput_tok_s"],
                         pp_over_ep=b["throughput_tok_s"]/a["throughput_tok_s"]))
    table(OUT/"best_tested.csv",best)
    best_table=[f"| {r['workload']} | {r['ep_mode']} / {r['ep_output_tok_s']:.1f} | {r['pp_mode']} / {r['pp_output_tok_s']:.1f} | {r['pp_over_ep']:.3f}× |" for r in best]
    lines=["# H100：Expert 小 tile 优化能否推广到其他 workload？", "",
        f"**可以推广到本轮测到的每卡16 token。** PP2在每卡4/8/16 token重复文档上的吞吐分别提升{pp[0]['throughput_gain_pct']:.1f}%、{pp[1]['throughput_gain_pct']:.1f}%、{pp[2]['throughput_gain_pct']:.1f}%；16 token多主题输入提升{pp[3]['throughput_gain_pct']:.1f}%。同一优化也提升EP2，双方各自选择较快候选后，本轮四组均为EP2吞吐更高。",
        "",
        f"随着batch扩大，PP2相对收益下降，但Expert内核改善仍存在：每卡16 token重复文档下，PP2 GEMM由{pp[2]['native_expert_gemm_us']:.1f}降至{pp[2]['optimized_expert_gemm_us']:.1f}µs/层。保持16 token改成多主题后，PP原生GEMM升至{pp[3]['native_expert_gemm_us']:.1f}µs，优化后仍为{pp[3]['optimized_expert_gemm_us']:.1f}µs。因此仅用token数描述workload不足；实际路由和Expert执行形状也是profiler应测的特征。",
        "",
        "本轮暂停 TBO 适配，只验证此前公平应用于 EP2、PP2 的 Expert 优化。没有启用此前 PP lookahead/feedback 调度原型。",
        "",
        "## 实验范围", "",
        "完整 Qwen3.6-35B-A3B BF16，同一节点 spartan-gpgpu178、同一对 H100 NVLink GPU0/1；整节点 Guard 租约、两 GPU 实际运行。EP2 为 Attention DP2×TP1、MoE EP2；PP2 为两个20层 stage、两个原生流水槽位。全局 B8/B16/B32 对应目标每卡 Decode 输入4/8/16 token，PP 的 microbatch 使用原生空闲期 API 设为 B/2。",
        "",
        "前三组保留此前重复文档组成；第四组保持 B32，改成32个不同主题。所有请求均有不同的首 token，避免请求之间共享前缀。每条 Decode 请求使用真实 GPU 缓存4096 token，再输入1 token、生成128 token；temperature0、ignore EOS。独立冷 Prefill 为每条4096新输入→1输出。每配置、每 workload、每阶段5个正式 cohort，保留全部轮次。",
        "",
        "原来的已归档策略只在本地 T≤4 启用。本轮为检验该内核的适用范围，将双方的门限同样扩到 T≤16；内核实现、形状对应的默认 tile 选择、BF16精度、路由和 DeepEP dispatch/combine 均沿用原实现。原始 T≤4 策略在 T8/T16 会回退到 native，本轮不是宣称原策略自动覆盖了大形状。EP 原生 shared-expert overlap 保持开启，TBO 关闭。",
        "", "## Decode 端到端结果", "",
        "吞吐为全部请求生成的 output token / cohort makespan，均值±样本SD；TPOT和E2E为请求均值，单位ms。TPOT按首token到末token除以127计算。",
        "", "| Workload | 配置 | Native output tok/s ↑ | Small-tile output tok/s ↑ | 吞吐变化 | TPOT ms ↓ | E2E ms ↓ |",
        "|---|---|---:|---:|---:|---:|---:|", *full_table, "",
        f"PP2 在重复文档的每卡4/8/16 token三档吞吐变化分别为 {pp[0]['throughput_gain_pct']:+.2f}% / {pp[1]['throughput_gain_pct']:+.2f}% / {pp[2]['throughput_gain_pct']:+.2f}%；16 token多主题为 {pp[3]['throughput_gain_pct']:+.2f}%。EP2 对应为 {ep[0]['throughput_gain_pct']:+.2f}% / {ep[1]['throughput_gain_pct']:+.2f}% / {ep[2]['throughput_gain_pct']:+.2f}% / {ep[3]['throughput_gain_pct']:+.2f}%。",
        "", "![End-to-end and Expert](workload_generalization.png)", "",
        "双方都给相同内核选择机会后，各自保留本轮平均吞吐更高的实现；若EP的native更快，比较时保留native。下面是两种已测试候选的比较，不代表穷尽调优的上限。",
        "", "| Workload | EP2较快候选 / output tok/s | PP2较快候选 / output tok/s | PP / EP |",
        "|---|---:|---:|---:|", *best_table, "",
        "## 真实路由和 Expert 时间", "",
        "活跃专家数来自独立 eager 诊断中的真实路由，取每层第一个匹配目标本地输入数的调用，不是全部128步的平均，也不是预设32或20。该诊断包括cached单token Extend与Decode。Assignments是到达该卡的token–expert配对。GEMM时间来自另一轮原生 CUDA Graph profile，每rank跳过前4个稳定Decode forward，再选16个；两rank均值，单位µs/GPU/实际层调用。它不含全部路由、激活、索引和流水等待，不能直接相加代替E2E。",
        "", "| Workload | 配置 | Active experts native→opt | Assignments native→opt | Expert GEMM µs native→opt ↓ | GEMM降幅 |",
        "|---|---|---:|---:|---:|---:|", *expert_table, "",
        "![Actual routing](routing_geometry.png)", "",
        "## Prefill 对照", "",
        "Cold Prefill大形状继续走native DeepGEMM；小 tile 策略本身不预期改善该部分。吞吐为新增input token/s，E2E单位ms，所有波动轮次均保留。",
        "", "| Workload | 配置 | Native input tok/s ↑ | Small-tile input tok/s ↑ | E2E ms native→opt ↓ |",
        "|---|---|---:|---:|---:|", *phase_table, "",
        "## 检查与可复现材料", "",
        "正式计时不含模型初始化、JIT、图捕获、prefix准备、路由诊断、数值比较、概率验证或profiler。所有正式请求逐条核验精确缓存数及输出长度。每rank使用真实权重检验T4/T8/T16的EP索引覆盖、block边界、空专家及全部专家活跃；实际路由另做EP层输出对照与PP两stage的相同输入 native/Triton 对照，实验相对L2门限1.5%。独立8-token top-10概率验证仅在相同生成历史处比较；没有声明长生成逐token一致或任务质量等价。",
        "",
        "引擎继续使用冻结commit `870269702285d94bdd926a50cc29ad1a20bcf60c` 和原不可变tag；模型revision `995ad96eacd98c81ed38be0c5b274b04031597b0`。运行代码、模型/通信来源、Guard/cache/cleanup检查和全部冻结输入哈希随manifest保存。不存在兼容的Qwen混合GDN CPU-KV checkpoint，沿用真实模型产生的GPU radix前缀与GDN状态。",
        "",
        "逐配置、逐轮、逐请求、各rank分项、真实路由及概率对照分别保存在 `analysis/metrics.csv`、`cohorts.csv`、`requests.csv`、`components_per_rank.csv`、`routing.csv`、`probability_positions.csv`。`comparisons.csv`为本页逐行对照。大profile与原始SSE仍在各run原路径，归档保留哈希引用。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    (OUT/"results.json").write_text(json.dumps(dict(comparisons=comparisons,best_tested=best,**result),indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":
    main()
