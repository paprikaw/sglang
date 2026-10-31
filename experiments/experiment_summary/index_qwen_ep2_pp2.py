"""Archive the saved two-GPU Qwen comparison; never launch or edit a run."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "ep2_pp2_qwen_a100_20261024"
R7 = EXPERIMENT / "runs/qwen122_ep2_pp2_r7"
R8 = EXPERIMENT / "runs/qwen122_pp2_two_slots_r8"
REPORT = HERE / "component_reports/Q01.md"
METRICS = HERE / "qwen_ep2_pp2_metrics.csv"
PROFILES = HERE / "qwen_ep2_pp2_prefill_profiles.csv"
BEGIN = "<!-- BEGIN Q01 QWEN EP2 PP2 -->"
END = "<!-- END Q01 QWEN EP2 PP2 -->"
ARMS = [("EP2", R7, "ep2", None), ("PP2，四槽", R7, "pp2", 4),
        ("PP2，两槽", R8, "pp2", 2)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError((actual, expected))


def measured(path, workload):
    data = json.loads(path.read_text())
    assert data["valid"] is True and data["profiler_enabled"] is False
    assert len(data["rounds"]) == 3
    requests = []
    for row in data["rounds"]:
        assert len(row["requests"]) == 8
        requests.extend(row["requests"])
        for request in row["requests"]:
            assert request["cached_tokens"] == (0 if workload == "prefill" else 4096)
            assert request["prompt_tokens"] == (4096 if workload == "prefill" else 4097)
            assert request["output_tokens"] == (1 if workload == "prefill" else 128)
        tokens = sum(r["prompt_tokens"] if workload == "prefill" else r["output_tokens"]
                     for r in row["requests"])
        field = "input_tokens_per_s" if workload == "prefill" else "output_tokens_per_s"
        close(row[field], tokens / row["makespan_s"])
    for field in ["ttft_s", "e2e_s"] + (["tpot_ms"] if workload == "decode" else []):
        values = sorted(r[field] for r in requests)
        for stat, value in [("mean", statistics.mean(values)),
                            ("median", statistics.median(values)),
                            ("p99", values[math.ceil(len(values) * .99) - 1])]:
            close(data[field][stat], value)
    if workload == "prefill":
        assert data["tpot_ms"] is None
    field = "input_tokens_per_s" if workload == "prefill" else "output_tokens_per_s"
    close(data[field], statistics.mean(r[field] for r in data["rounds"]))
    return data


def triple(data, field):
    return "—" if data[field] is None else " / ".join(f"{data[field][s]:.4f}" for s in ["mean", "median", "p99"])


def main():
    sources = {}

    def source(path):
        sources[str(path)] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)

    performance, profile_rows, views = [], [], {}
    for label, run, arm, slots in ARMS:
        source(run / "config.json")
        for workload in ["prefill", "decode"]:
            path = run / arm / f"metrics.{workload}.json"
            data = measured(path, workload)
            source(path)
            views[label, workload] = data
            row = dict(configuration=label, workload=workload, actual_gpus=2,
                       pipeline_slots=slots, requests_per_round=8, rounds=3,
                       cached_tokens_per_request=0 if workload == "prefill" else 4096,
                       fresh_input_tokens_per_request=4096 if workload == "prefill" else 1,
                       output_tokens_per_request=1 if workload == "prefill" else 128,
                       throughput_kind="input_tokens/s" if workload == "prefill" else "output_tokens/s",
                       throughput=data["input_tokens_per_s" if workload == "prefill" else "output_tokens_per_s"])
            for field in ["ttft_s", "tpot_ms", "e2e_s"]:
                for stat in ["mean", "median", "p99"]:
                    row[f"{field}_{stat}"] = data[field][stat] if data[field] else None
            row.update(source_path=str(path), source_sha256=sha(path))
            performance.append(row)
        paths = sorted((run / "analysis").glob(f"{arm}-prefill-*.json"))
        assert len(paths) == 2
        for path in paths:
            analysis = json.loads(path.read_text())
            source(path)
            raw = Path(analysis["path"])
            raw = raw if raw.is_absolute() else HERE.parent.parent / raw
            assert sha(raw) == analysis["sha256"]
            source(raw)
            calls = [r["gpu_component_union_ms"] for r in analysis["invocations"]
                     if r["step"].startswith("step[EXTEND bs=1 toks=4096 ")]
            assert len(calls) == 192
            avg = lambda names: statistics.mean(sum(c.get(n, 0) for n in names) for c in calls)
            profile_rows.append(dict(configuration=label, trace=path.name, layer_calls=len(calls),
                unit="ms/rank/layer invocation; local fresh input 4096; GPU activity, not E2E",
                attention_ms=avg(["attention.full", "attention.linear"]),
                routed_expert_ms=avg(["expert.routed"]), shared_expert_ms=avg(["expert.shared"]),
                dispatch_ms=avg(["communication.dispatch"]), combine_ms=avg(["communication.combine"]),
                source_path=str(path), source_sha256=sha(path)))
    write_csv(METRICS, performance)
    write_csv(PROFILES, profile_rows)

    lines = [
        "# Q01：两张 A100 上的 Qwen3.5-122B，EP2 与 PP2",
        "", "归档日期：2026-10-25。读取已经完成的测量，不重新运行 GPU。四槽与两槽是不同的 PP 调度设置，分别保留。",
        "", "## 1. 术语、数量和计时边界", "",
        "本报告不使用单个字母代替请求数、上下文长度或输入长度。后文缩写与术语的确切含义如下。", "",
        "| 名称 | 本报告的含义与范围 |", "|---|---|",
        "| token | 模型分词后的单位；输入、已缓存上下文和生成输出分开计数，不是字符数。 |",
        "| Prefill | 本轮有 8 个请求；每条没有已缓存上下文，处理 4096 个新输入 token，只生成 1 token。整批新输入共 32768 token。 |",
        "| Decode | 本轮有 8 个请求；每条已有 4096 token 的真实模型上下文缓存在 GPU，新增输入 1 token，生成 128 token。前缀构建不计时。 |",
        "| EP／专家并行 | 同一层的路由专家分布在几张 GPU 上。EP2 为两卡共享该层的 256 个路由专家，每卡 128 个；不是两个独立模型副本。 |",
        "| PP／流水并行、stage | 模型按层拆为前后阶段。PP2 每卡负责连续 24 层，每个已分配层含全部 256 个路由专家；一条请求仍经过两阶段。 |",
        "| TP／张量并行、DP／数据并行 | TP 拆分同层张量计算；DP 分配不同请求。EP2 的 Attention 为 TP1×DP2，每卡处理自己的请求；PP2 每阶段 Attention 为 TP1。 |",
        "| 流水槽位、microbatch、chunk | 槽位是调度器轮转维护的批次位置，不是 GPU 或 stage 数。microbatch 是一次流水处理的请求子集，本轮最多 4 请求；Prefill 实际每次每卡处理 1 请求的 4096 个新输入 token。chunk 是单次输入分块。 |",
        "| 四槽／两槽 | 原生循环槽数＝PP stage 数＋额外异步深度。额外深度 2 对应四槽；额外深度 0 对应两槽。槽数不证明持续满流水。 |",
        "| Attention | 注意力及本轮注解内的投影；模型有 36 层 GatedDeltaNet 线性 Attention、12 层 Full Attention。下表按实际层调用加权。 |",
        "| Routed／Shared Expert | Routed 是路由选择的专家计算，每 token 选择 8 个专家；Shared 是共享专家计算。两者分别统计。 |",
        "| Dispatch／Combine | 前者将 token 发到专家所在卡，后者取回并合并结果。PP 没有跨卡专家交换，但仍有阶段间激活传输、反馈与同步。 |",
        "| TTFT | 客户端放行该请求至收到第一个输出 token 的时间，包含排队与初始输入处理，不是纯 GPU 计算时间。 |",
        "| TPOT | 首输出 token 至末输出 token 的时间除以输出数量减一；单位毫秒。只输出 1 token 的 Prefill 点不适用，留空而非写 0。 |",
        "| 端到端延迟／吞吐 | 延迟从请求放行至最后输出。每轮吞吐为全批输入或输出 token 数除以整批完成耗时，再对三轮吞吐取平均；不是平均请求延迟的倒数。 |",
        "| mean／median／P99 | 分别是均值、中位数、99 百分位数。延迟汇总 3 轮×8 请求共 24 个观测；最近秩法 P99 等于样本最大值，不代表稳定服务尾延迟。 |",
        "| FP8／W8A16／BF16 | 序列化权重为 8 位浮点；Marlin 以 8 位权重、16 位激活运行；激活和 KV 均为 BF16。A100 不使用原生 FP8 Tensor Core。 |",
        "| GPU 活动时间 | 通过真实 launch correlation 将 kernel 归属到组件，计算活动区间并集。不同组件可重叠；CPU 范围、NCCL kernel 驻留与网卡传输耗时不是同一个量。 |",
        "", "## 2. 设备、模型与公平性", "",
        "完整 Qwen3.5-122B-A10B-FP8 文本模型，48 层、每层 256 个路由专家，序列化张量约 118.42 GiB；不截层。固定节点 spartan-gpgpu069 的 GPU0／1，两张 A100 80GB，经 NV12 NVLink 直连。两个布局顺序运行。",
        "", "两组均使用 Marlin、Triton Attention，关闭 CUDA Graph 和 overlap schedule。EP2 为 DeepEP normal 的原生 NVLink 路径；不是之前跨节点的 CE、PCIe 或 RDMA low-latency 对照。PP2 无远端专家交换。完整权重从 canonical CPU bank 复用；加载、首次编译和预热不计时。",
        "", "每项三轮正式测量，计时不开 profiler；profile 单独采集。Prefill 每轮清空原生 GPU 前缀缓存，Decode 每轮复用同样的真实前缀。本轮容量充足，没有合成 KV、CPU HiCache 换入或容量不足换出。两槽 PP 是后续补测，复用原 EP2 基线，不是一次同时重跑全部配置。",
        "", "## 3. Prefill：包含约 8811 token/s 的四槽结果", "",
        "延迟单元格均为 **均值 / 中位数 / P99**；单位秒。每条只输出 1 token，所以 TTFT 等于端到端延迟。", "",
        "| 配置 | 输入吞吐 token/s ↑ | TTFT 秒 ↓ | 端到端秒 ↓ | TPOT |",
        "|---|---:|---|---|---|",
    ]
    for label, _, _, _ in ARMS:
        data = views[label, "prefill"]
        lines.append(f"| {label} | {data['input_tokens_per_s']:.2f} | {triple(data, 'ttft_s')} | {triple(data, 'e2e_s')} | 不适用 |")
    lines += ["", "三轮输入吞吐分别如下；轮号只表示各配置自己的重复测量，不是交错配对。", "",
              "| 配置 | 第 1 轮 | 第 2 轮 | 第 3 轮 |", "|---|---:|---:|---:|"]
    for label, _, _, _ in ARMS:
        lines.append("| " + label + " | " + " | ".join(f"{r['input_tokens_per_s']:.2f}" for r in views[label, "prefill"]["rounds"]) + " |")
    lines += ["", "四槽 PP 输入吞吐比 EP2 高 1.61%，平均 TTFT 则高 7.77%；其最后一条请求较早完成，不代表每条请求都更早。EP2 三轮有波动，不能据 1.61% 判定 PP 稳定胜出。两槽 PP 的 Prefill 吞吐低于 EP2 5.60%，不能把两槽数据替换四槽后继续使用同一结论。",
              "", "## 4. Prefill：对应每卡每层的计算与通信", "",
              "下表对两张卡的匹配本地形状调用取均值，单位 **毫秒／每卡／一次实际层调用**。每条调用处理 1 个请求的 4096 个新输入 token；EP 还可能接收另一张卡发来的专家 token，因此不是严格相同全局输入的隔离 Expert 微基准。Attention 按 12 层 Full、36 层线性类型加权。", "",
              "| 配置 | Attention | Routed Expert | Shared Expert | Dispatch | Combine |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, _, _, _ in ARMS:
        rows = [r for r in profile_rows if r["configuration"] == label]
        values = [statistics.mean(r[k] for r in rows) for k in ["attention_ms", "routed_expert_ms", "shared_expert_ms", "dispatch_ms", "combine_ms"]]
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "四槽 PP 的 Routed Expert 增加约 0.498 ms，Attention 增加约 0.603 ms，同时避免约 1.710 ms 的逐层专家通信。但 PP 阶段传输在层范围外，GPU 分项可重叠，以上数字不能直接相加得到端到端关键路径。",
              "", "每层本地专家从 128 增到 256、每个专家收到的 token 行数改变，可能增加权重读取和小矩阵成本；这是与观察相符的解释，不是已经测得的 HBM 带宽瓶颈。真实路由、不同层和通信／计算重叠尚未完全隔离。只改槽数就改变 Prefill 排名，也说明不能把差距全部归因于专家权重。",
              "", "## 5. Decode：保留相同实验的槽位对照", "",
              "每条 GPU 已缓存 4096 token，新增 1 token，输出 128 token。延迟仍为 **均值 / 中位数 / P99**。这是另一种 workload，不能与上一节 Prefill 的毫秒／层直接相减。", "",
              "| 配置 | 输出吞吐 token/s ↑ | TTFT 秒 ↓ | TPOT 毫秒 ↓ | 端到端秒 ↓ |",
              "|---|---:|---|---|---|"]
    for label, _, _, _ in ARMS:
        data = views[label, "decode"]
        lines.append(f"| {label} | {data['output_tokens_per_s']:.2f} | {triple(data, 'ttft_s')} | {triple(data, 'tpot_ms')} | {triple(data, 'e2e_s')} |")
    lines += ["", "两槽 PP 的 Decode 吞吐比 EP2 高 34.92%、平均 TPOT 低 25.80%；四槽 PP 的 Decode 则明显更差。Prefill 与 Decode 的合适槽位可能不同；本轮没有做完整槽位调参或容量边界扫描。",
              "", "## 6. 来源、状态与可复查入口", "",
              "原始 r7 完成测量、cache 审计及 GPU 清理，但退出包装器误捕获正常 SystemExit(0)，原始 Guard 状态仍保留为失败。不能把它写成 Guard 全通过；本归档独立复算六份测量的 24 请求统计与三轮吞吐，并核对 profile 原始 trace 哈希。最终两槽 r8 的 Guard、cache 审计和清理均通过。没有修改原始运行记录。", "",
              f"- [请求指标 CSV，六条独立配置×workload 记录]({METRICS})；[Prefill 逐卡分项 CSV]({PROFILES})。",
              f"- [来源路径与 SHA-256 清单]({HERE / 'qwen_ep2_pp2_archive.json'})；[本次归档校验]({HERE / 'qwen_ep2_pp2_validation.json'})。",
              f"- [原完整实验报告]({EXPERIMENT / 'results/summary.md'})；[原逐层表]({EXPERIMENT / 'results/stage_metrics.csv'})；[原测量完整性检查]({EXPERIMENT / 'results/data_validation.json'})。"]
    for label, run, arm, _ in ARMS:
        lines.append(f"- {label}：[Prefill 原始测量]({run / arm / 'metrics.prefill.json'})；[Decode 原始测量]({run / arm / 'metrics.decode.json'})；[冻结配置]({run / 'config.json'})。")
    diagnosis = HERE / "qwen_prefill_diagnosis/REPORT.md"
    if diagnosis.exists():
        lines += ["", f"### 进一步解释 Prefill 的吞吐与 Attention 慢点", "",
                  f"[详细诊断：Attention TP、槽位、请求完成分布与 NCCL／投影重叠]({diagnosis})。这是对已有 trace 的 CPU 分析，不是新 GPU 测量；保留原报告和原始 Guard 状态。"]
    REPORT.write_text("\n".join(lines) + "\n")

    readme = HERE / 'README.md'
    text = readme.read_text()
    text = text.replace("更新：2026-10-24。", "更新：2026-10-25。", 1)
    text = text.replace("单层 Attention／Expert 扫描见第 7 节。",
                        "单层 Attention／Expert 扫描见第 7 节；单节点双卡补充见第 8 节 Q01。", 1)
    text = text.replace("## 8. 以后怎么追踪", "## 9. 以后怎么追踪")
    section = "\n".join([BEGIN, "## 8. 补充：单节点双卡 EP2 与 PP2（不是八卡 EP8）", "",
        "以下是独立资源规模的相关实验，不混入第 2–4 节的八卡排名或旧 metrics.csv／profiles.csv。每个条目先读简短报告中的术语和计时边界。", "",
        "| ID／简短报告 | 设备与明确的请求负载 | 已有结果 | profile／原始数据 |",
        "|---|---|---|---|",
        f"| [Q01：Qwen3.5-122B，EP2／四槽 PP2／两槽 PP2]({REPORT}) | 单节点同一对 A100 NVLink；8 请求。Prefill：每条无旧缓存、新输入4096、输出1。Decode：每条GPU已缓存4096、新输入1、输出128；无CPU换入。 | Prefill：EP2 **8670.75**、四槽 PP **8810.58**、两槽 PP **8185.11** input token/s。Decode：EP2 **72.08**、四槽 PP **49.04**、两槽 PP **97.26** output token/s。TTFT／TPOT／端到端均值、中位数、P99 见报告，不能只按吞吐认定全面领先。 | [全部请求指标]({METRICS})；[Prefill逐卡计算／通信]({PROFILES})；[来源与哈希]({HERE / 'qwen_ep2_pp2_archive.json'})。含真实模型 profile；GPU毫秒／层不等于请求TPOT。 |",
        "", END, ""])
    section = section.rstrip() + "\n\n"
    if diagnosis.exists():
        section = section.replace(f"[来源与哈希]({HERE / 'qwen_ep2_pp2_archive.json'})。",
            f"[来源与哈希]({HERE / 'qwen_ep2_pp2_archive.json'})；[Attention／槽位详细诊断]({diagnosis})。")
    if BEGIN in text:
        text = text[:text.index(BEGIN)] + section + text[text.index(END) + len(END):].lstrip("\n")
    else:
        marker = "## 9. 以后怎么追踪"
        assert text.count(marker) == 1
        text = text.replace(marker, section + marker)
    readme.write_text(text)

    archive = dict(schema_version=1, id="Q01", date="2026-10-25", no_gpu_launch=True,
                   report=str(REPORT), original_records_unchanged=True,
                   original_guard_status={"r7": "failed_wrapper_SystemExit_0; measurements independently checked; cache and cleanup passed",
                                          "r8": "passed"}, sources=list(sources.values()),
                   artifacts={str(p): sha(p) for p in [REPORT, METRICS, PROFILES]})
    (HERE / "qwen_ep2_pp2_archive.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2) + "\n")
    missing = []
    links = re.findall(r"\]\((/[^)]+)\)", REPORT.read_text() + section)
    for target in links:
        if not Path(target).exists() and Path(target).name != "qwen_ep2_pp2_validation.json":
            missing.append(target)
    assert not missing, missing
    for item in sources.values():
        assert sha(Path(item["path"])) == item["sha256"]
    validation = dict(passed=True, id="Q01", no_gpu_launch=True, metric_records=len(performance),
                      formal_requests_recomputed=6 * 24, prefill_profile_rows=len(profile_rows),
                      source_files_unchanged=len(sources), absolute_links_checked=len(links),
                      missing_links=missing, original_guard_failure_preserved=True,
                      separate_from_eight_gpu_csv=True,
                      report_sha256=sha(REPORT), readme_sha256=sha(readme))
    (HERE / "qwen_ep2_pp2_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
