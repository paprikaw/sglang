"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {data: null, refreshing: false, refreshSeconds: 15, userInterval: false, timer: null, nextRefresh: null, expanded: new Set(), failure: null};
  const statuses = {live: "在线 · 已采样", unreachable: "不可达", unbooked: "历史 / 未预约", probing: "正在采样"};
  const known = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const list = (value) => Array.isArray(value) ? value : [];
  const valueOrDash = (value) => value === null || value === undefined || value === "" ? "—" : String(value);
  const h = (tag, className, text) => {const element = document.createElement(tag); if (className) element.className = className; if (text !== undefined) element.textContent = String(text); return element;};
  const bytes = (value) => {
    if (!known(value)) return "—";
    if (value === 0) return "0 B";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / Math.pow(1024, exponent)).toLocaleString("zh-CN", {maximumFractionDigits: exponent >= 3 ? 2 : 1})} ${units[exponent]}`;
  };
  const time = (epoch) => known(epoch) && epoch > 0 ? new Date(epoch * 1000).toLocaleString("zh-CN", {hour12: false}) : "—";
  const count = (value) => known(value) ? value.toLocaleString("zh-CN") : "—";
  const sumField = (nodes, group, key) => {
    const values = nodes.map((node) => node[group]?.[key]);
    const measured = values.filter(known);
    return {value: measured.length ? measured.reduce((a, b) => a + b, 0) : null, unknown: values.length - measured.length};
  };
  const sumNote = (result) => result.unknown ? `已测总量 · ${result.unknown} 个节点未知` : "已测总量 · 仅在线节点";
  const nodeStatus = (node) => {
    const bookingState = String(node.booking?.state || "").toUpperCase();
    if (["CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "UNBOOKED"].includes(bookingState)) return "unbooked";
    return node.status || "probing";
  };
  const rankId = (rank) => valueOrDash(rank.rank ?? rank.global_rank);
  const layerRange = (rank) => {
    if (rank.layer_start === null || rank.layer_start === undefined || rank.layer_end === null || rank.layer_end === undefined) return "—";
    return `${rank.layer_start}–${rank.layer_end}`;
  };
  const layoutText = (layout) => ["tp", "ep", "pp", "dp"].filter((key) => known(layout?.[key])).map((key) => `${key.toUpperCase()} ${layout[key]}`).join(" · ") || "布局未登记";
  const appendText = (parent, className, text) => parent.append(h("div", className, text));

  function summaryCard(label, value, note, className = "") {
    const card = h("div", `summary-card ${className}`);
    card.append(h("div", "summary-label", label), h("div", "summary-value", value), h("div", "summary-note", note));
    return card;
  }

  function renderSummary(data) {
    const nodes = list(data.nodes);
    const live = nodes.filter((node) => nodeStatus(node) === "live");
    const weight = sumField(live, "weight", "bytes");
    const kvResident = sumField(live, "kv", "resident_bytes");
    const kvLogical = sumField(live, "kv", "logical_bytes");
    const booked = nodes.filter((node) => nodeStatus(node) !== "unbooked").length;
    const unseen = nodes.filter((node) => ["unreachable", "probing"].includes(nodeStatus(node))).length;
    $("summary").replaceChildren(
      summaryCard("在线 / 预约节点", `${live.length} / ${booked}`, unseen ? `${unseen} 个节点尚未验证 · 历史节点不计入` : "本次可达并完成采样的节点"),
      summaryCard("共享 CPU Weight", bytes(weight.value), sumNote(weight), "weight"),
      summaryCard("KV checkpoint · RAM resident", bytes(kvResident.value), sumNote(kvResident), "kv"),
      summaryCard("KV checkpoint · 文件逻辑量", bytes(kvLogical.value), sumNote(kvLogical))
    );
  }

  function metric(label, primary, secondary, hint) {
    const el = h("div", "metric");
    const main = h("div", "metric-main", primary);
    if (secondary) main.append(h("span", "metric-secondary", secondary));
    el.append(h("div", "metric-label", label), main);
    if (hint) el.append(h("div", "metric-hint", hint));
    return el;
  }

  function memoryMetric(node) {
    const memory = node.memory || {};
    const total = memory.total_bytes;
    const available = memory.available_bytes;
    const used = known(total) && known(available) ? Math.max(0, total - available) : null;
    const el = metric("CPU 内存", bytes(used), `/ ${bytes(total)}`, null);
    const bar = h("div", `memory-bar ${known(used) && total > 0 ? "" : "unknown"}`);
    if (known(used) && total > 0) {
      // Verified resident weights and KV are subsets of CPU usage, not extra usage.
      // Daemon RSS is not weight residency: canonical memfds need not be mmap'd.
      // Clamp to physical used memory when independently sampled values differ.
      const weight = Math.min(used, known(node.weight?.resident_bytes) ? node.weight.resident_bytes : 0);
      const kv = Math.min(Math.max(0, used - weight), known(node.kv?.resident_bytes) ? node.kv.resident_bytes : 0);
      for (const [kind, amount] of [["weight", weight], ["kv", kv], ["other", Math.max(0, used - weight - kv)]]) {
        const segment = h("span", `memory-segment ${kind}`);
        segment.style.width = `${Math.min(100, amount / total * 100)}%`;
        bar.append(segment);
      }
      bar.title = `CPU 已用 ${bytes(used)}；Weight resident ${bytes(node.weight?.resident_bytes)}；KV resident ${bytes(node.kv?.resident_bytes)}；分类为独立采样近似值`;
    }
    el.append(bar, h("div", "memory-note", `可用 ${bytes(available)}${known(memory.cgroup_current_bytes) ? ` · 作业 cgroup ${bytes(memory.cgroup_current_bytes)} / ${bytes(memory.cgroup_limit_bytes)}` : ""}`));
    if (!known(node.weight?.resident_bytes) || !known(node.kv?.resident_bytes)) el.append(h("div", "memory-note", `${!known(node.weight?.resident_bytes) ? "Weight resident 未知" : ""}${!known(node.weight?.resident_bytes) && !known(node.kv?.resident_bytes) ? " · " : ""}${!known(node.kv?.resident_bytes) ? "KV resident 未知" : ""}，未着色部分含未分类内存。`));
    return el;
  }

  function renderGpuArea(node) {
    const area = h("div", "gpu-area");
    const label = h("div", "gpu-label-row");
    label.append(h("span", "", "GPU 硬件与并行布局"));
    const layouts = list(node.active_layouts);
    const chips = h("div", "layout-chips");
    const kvDatasets = list(node.kv?.datasets);
    const allLayouts = layouts.length ? layouts : kvDatasets.filter((dataset) => dataset.layout).map((dataset) => ({...dataset.layout, title: dataset.label || dataset.id}));
    for (const layout of allLayouts) {
      const chip = h("span", "layout-chip", layoutText(layout));
      chip.title = `${valueOrDash(layout.title || layout.id)}${layout.state ? ` · ${layout.state}` : ""}${known(layout.node_rank) ? ` · node rank ${layout.node_rank}` : ""}`;
      chips.append(chip);
    }
    if (!allLayouts.length) chips.append(h("span", "not-measured", "布局未登记"));
    label.append(chips);
    area.append(label);
    const inventory = list(node.gpus);
    const gpuGrid = h("div", "gpus");
    const ranks = kvDatasets.flatMap((dataset) => list(dataset.rank_entries).map((rank) => ({...rank, dataset: dataset.label || dataset.id})));
    // Only an explicit gpu_index is an association. Never infer rank % N.
    const hardware = [...inventory].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    for (const gpu of hardware) {
      const card = h("div", "gpu-card");
      const heading = h("div", "gpu-heading");
      const name = h("span", "", `GPU ${valueOrDash(gpu.index)}`);
      name.prepend(h("i", "gpu-icon"));
      heading.append(name, h("span", "gpu-capacity", bytes(gpu.total_bytes)));
      card.append(heading, h("div", "gpu-model", gpu.name || "硬件未采样"));
      const mapped = ranks.filter((rank) => known(rank.gpu_index) && rank.gpu_index === gpu.index);
      if (mapped.length) {
        const seen = new Set();
        const names = mapped.filter((rank) => {const key = `${rankId(rank)}/${rank.pp_rank}/${rank.tp_rank}`; if (seen.has(key)) return false; seen.add(key); return true;}).map((rank) => `rank ${rankId(rank)}${known(rank.pp_rank) ? ` · PP ${rank.pp_rank}` : ""}${known(rank.tp_rank) ? ` · TP ${rank.tp_rank}` : ""}`);
        card.append(h("div", "gpu-ranks", names.join(" / ")));
      } else card.append(h("div", "gpu-ranks muted", "rank 映射未验证"));
      gpuGrid.append(card);
    }
    if (hardware.length) area.append(gpuGrid);
    else area.append(h("div", "logical-ranks", "GPU 硬件未采样，GPU 数量和型号尚未验证。"));
    const unmapped = ranks.filter((rank) => !known(rank.gpu_index));
    if (unmapped.length) {
      const logical = h("div", "logical-ranks");
      logical.append(h("div", "logical-label", "逻辑 KV ranks · 物理 GPU 映射未知"));
      const rankChips = h("div", "rank-chips");
      const seen = new Set();
      for (const rank of unmapped) {
        const text = `rank ${rankId(rank)}${known(rank.pp_rank) ? ` / PP${rank.pp_rank}` : ""}${known(rank.tp_rank) ? ` / TP${rank.tp_rank}` : ""}`;
        if (seen.has(text)) continue;
        seen.add(text);
        const chip = h("span", "rank-chip", text);
        chip.title = `${valueOrDash(rank.dataset)} · 层 ${layerRange(rank)} · ${bytes(rank.logical_bytes)}`;
        rankChips.append(chip);
      }
      logical.append(rankChips);
      area.append(logical);
    }
    if (!ranks.length) area.append(h("div", "layout-empty", "尚无已登记的 KV rank 明细；可展开查看 weight 和布局信息。"));
    return area;
  }

  function table(headers, rows, nameColumns = []) {
    const wrapper = h("div", "table-scroll");
    const el = h("table");
    const head = h("thead");
    const tr = h("tr");
    for (const title of headers) tr.append(h("th", "", title));
    head.append(tr);
    const body = h("tbody");
    for (const row of rows) {
      const tr = h("tr");
      row.forEach((cell, i) => tr.append(h("td", nameColumns.includes(i) ? "table-name" : "", valueOrDash(cell))));
      body.append(tr);
    }
    el.append(head, body);
    wrapper.append(el);
    return wrapper;
  }

  function renderDetails(node) {
    const details = h("details", "details");
    const key = String(node.node || "unknown");
    details.open = state.expanded.has(key);
    details.addEventListener("toggle", () => {if (details.open) state.expanded.add(key); else state.expanded.delete(key);});
    const banks = list(node.weight?.banks);
    const datasets = list(node.kv?.datasets);
    const rankCount = datasets.reduce((sum, dataset) => sum + list(dataset.rank_entries).length, 0);
    details.append(h("summary", "", `缓存与 rank 明细 · ${banks.length} 个 weight bank · ${datasets.length} 个 KV 数据集 · ${rankCount} 个 shards`));
    const content = h("div", "detail-content");
    const weightSection = h("div", "detail-section");
    weightSection.append(h("div", "detail-title", "共享 CPU Weight banks"));
    if (banks.length) weightSection.append(table(["模型 / Bank", "缓存量", "方案", "状态"], banks.map((bank) => [bank.model || bank.id, bytes(bank.bytes), bank.scheme, bank.state]), [0]));
    else weightSection.append(h("div", "zero-caption", "暂无 weight bank 登记。"));
    weightSection.append(h("div", "detail-caption", `当前状态 ${valueOrDash(node.weight?.status)} · 登记预留 ${bytes(node.weight?.reserved_bytes)} · resident ${bytes(node.weight?.resident_bytes)} · 进程 RSS ${bytes(node.weight?.rss_bytes)}。RSS 不是共享模型 weight 的驻留量。`));
    if (node.weight?.error) weightSection.append(h("div", "detail-caption observation-warning", node.weight.error));
    content.append(weightSection);

    if (list(node.active_layouts).length) {
      const layoutSection = h("div", "detail-section");
      layoutSection.append(h("div", "detail-title", "已登记的并行布局"));
      layoutSection.append(table(["布局", "TP / EP / PP / DP", "Node rank", "层范围", "状态"], list(node.active_layouts).map((layout) => [layout.title || layout.id, layoutText(layout), layout.node_rank, typeof layout.layer_ranges === "string" ? layout.layer_ranges : Array.isArray(layout.layer_ranges) ? layout.layer_ranges.map((range) => typeof range === "object" ? `${valueOrDash(range.pp_rank ?? range.rank)}: ${valueOrDash(range.layer_start ?? range.start)}–${valueOrDash(range.layer_end ?? range.end)}` : String(range)).join("; ") : "—", layout.state]), [0, 3]));
      content.append(layoutSection);
    }

    const kvSection = h("div", "detail-section");
    kvSection.append(h("div", "detail-title", "KV 数据集与 shard 布局"));
    if (!datasets.length) kvSection.append(h("div", "zero-caption", "暂无 KV 数据集登记。"));
    for (const dataset of datasets) {
      const block = h("div", "dataset-block");
      const heading = h("div", "dataset-heading", dataset.label || dataset.id || "KV 数据集");
      heading.append(h("span", "muted", `${layoutText(dataset.layout)} · 逻辑 ${bytes(dataset.logical_bytes)} · resident ${bytes(dataset.resident_bytes)}`));
      block.append(heading);
      const datasetState = [`状态 ${valueOrDash(dataset.status)}`];
      if (dataset.registered_boot_matches === false) datasetState.push("节点 boot 与登记不符，checkpoint 可复用性未验证");
      if (dataset.observation_complete === false) datasetState.push("本次统计不完整");
      block.append(h("div", `detail-caption ${dataset.registered_boot_matches === false || dataset.observation_complete === false ? "observation-warning" : ""}`, datasetState.join(" · ")));
      if (dataset.error) block.append(h("div", "detail-caption observation-warning", dataset.error));
      if (dataset.root) block.append(h("div", "dataset-root", dataset.root));
      const ranks = list(dataset.rank_entries);
      if (ranks.length) block.append(table(["逻辑 rank", "GPU", "TP rank", "PP rank", "层范围", "逻辑量", "Resident", "文件", "Tokens"], ranks.map((rank) => [rankId(rank), known(rank.gpu_index) ? rank.gpu_index : "映射未知", rank.tp_rank, rank.pp_rank, layerRange(rank), bytes(rank.logical_bytes), bytes(rank.resident_bytes), Array.isArray(rank.files) ? rank.files.length : count(rank.files), count(rank.tokens)])));
      else block.append(h("div", "zero-caption", "rank / 层范围尚未登记。"));
      kvSection.append(block);
    }
    kvSection.append(h("div", "detail-caption", `活跃引擎 HiCache host pool ${bytes(node.kv?.host_pool_bytes)} · ${node.kv?.host_pool_note || "尚未接入按组件遥测"}`));
    content.append(kvSection);
    details.append(content);
    return details;
  }

  function nodeCard(node) {
    const status = nodeStatus(node);
    const historical = status !== "live" && status !== "probing";
    const card = h("article", `node-card ${historical ? "historical" : ""}`);
    const header = h("div", "node-header");
    const title = h("div", "node-title");
    title.append(h("span", "node-name", node.node || "未知节点"), h("span", `node-status ${status}`, statuses[status] || valueOrDash(status)));
    const jobIds = list(node.booking?.job_ids);
    if (jobIds.length || node.booking?.state) title.append(h("span", "node-booking", `${valueOrDash(node.booking?.state)}${jobIds.length ? ` · Job ${jobIds.join(", ")}` : ""}`));
    header.append(title, h("span", "node-updated", `采样 ${time(node.last_probe_at)}`));
    card.append(header);
    const body = h("div", "node-body");
    if (node.error) body.append(h("div", "node-error", node.error));
    else if (historical) body.append(h("div", "node-error", status === "unbooked" ? "历史登记，仅供查看布局；当前缓存是否仍存在未验证。" : "本次采样不可达，以下为已登记信息，不计入在线总量。"));
    const metrics = h("div", "node-metrics");
    const weightMetric = metric("共享 CPU Weight", bytes(node.weight?.bytes), null, `登记预留 ${bytes(node.weight?.reserved_bytes)} · resident ${bytes(node.weight?.resident_bytes)}`);
    if (node.weight?.error) weightMetric.append(h("div", "metric-hint observation-warning", node.weight.error));
    const kvMetric = metric("CPU KV checkpoint", bytes(node.kv?.resident_bytes), "resident", `文件逻辑量 ${bytes(node.kv?.logical_bytes)} · ${Array.isArray(node.kv?.files) ? node.kv.files.length : count(node.kv?.files)} 个文件`);
    kvMetric.append(h("div", "metric-hint", `活跃 HiCache host pool ${bytes(node.kv?.host_pool_bytes)}`));
    if (node.kv?.host_pool_note) kvMetric.append(h("div", "metric-hint", node.kv.host_pool_note));
    if (node.kv?.error) kvMetric.append(h("div", "metric-hint observation-warning", node.kv.error));
    metrics.append(memoryMetric(node), weightMetric, kvMetric);
    body.append(metrics, renderGpuArea(node));
    card.append(body, renderDetails(node));
    return card;
  }

  function renderAlerts() {
    const alerts = $("alerts");
    alerts.replaceChildren();
    if (state.failure) alerts.append(h("div", "alert", `状态读取失败：${state.failure}。${state.data ? "保留上次采样，当前数值可能已过期。" : "等待下一次刷新。"}`));
    for (const error of list(state.data?.errors)) alerts.append(h("div", "alert", typeof error === "string" ? error : error.message || error.error || "存在采样异常"));
  }

  function render() {
    renderAlerts();
    if (!state.data) return;
    const data = state.data;
    renderSummary(data);
    const nodes = list(data.nodes).filter((node) => $("show-history").checked || nodeStatus(node) !== "unbooked");
    if (nodes.length) $("nodes").replaceChildren(...nodes.map(nodeCard));
    else {
      const empty = h("div", "empty-state");
      empty.append(h("p", "", "当前没有可展示的预约节点。"), h("p", "muted", list(data.nodes).length ? "勾选「显示历史 / 已取消节点」可查看已有登记。" : "发现节点后会自动更新。"));
      $("nodes").replaceChildren(empty);
    }
    $("observed-at").textContent = `数据采样 ${time(data.observed_at)}`;
    $("connection").className = `connection ${state.failure ? "error" : "live"}`;
    $("connection-label").textContent = state.failure ? "数据可能过期" : data.collecting ? "后台正在采样" : "实时监控";
  }

  function scheduleRefresh() {
    clearTimeout(state.timer);
    state.nextRefresh = Date.now() + state.refreshSeconds * 1000;
    state.timer = setTimeout(refresh, state.refreshSeconds * 1000);
    updateCountdown();
  }

  function updateCountdown() {
    $("next-refresh").textContent = state.refreshing ? "读取状态中…" : state.nextRefresh ? `${Math.max(0, Math.ceil((state.nextRefresh - Date.now()) / 1000))} 秒后刷新` : "";
  }

  async function refresh() {
    if (state.refreshing) return;
    clearTimeout(state.timer);
    state.refreshing = true;
    $("refresh").disabled = true;
    updateCountdown();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);
    try {
      const response = await fetch("/api/status", {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data || typeof data !== "object" || !Array.isArray(data.nodes)) throw new Error("返回数据缺少节点列表");
      state.data = data;
      state.failure = null;
      if (!state.userInterval && known(data.refresh_seconds) && data.refresh_seconds > 0) {
        state.refreshSeconds = Math.max(5, data.refresh_seconds);
        if (![...$("refresh-seconds").options].some((option) => Number(option.value) === state.refreshSeconds)) {
          const option = h("option", "", `${state.refreshSeconds} 秒`);
          option.value = String(state.refreshSeconds);
          $("refresh-seconds").append(option);
        }
        $("refresh-seconds").value = String(state.refreshSeconds);
      }
      render();
    } catch (error) {
      state.failure = error.name === "AbortError" ? "请求超时" : String(error.message || error);
      renderAlerts();
      $("connection").className = "connection error";
      $("connection-label").textContent = "连接异常";
      if (!state.data) {
        const empty = h("div", "empty-state");
        empty.append(h("p", "", "暂时无法读取缓存状态。"), h("p", "muted", "页面会自动重试，也可点击「立即刷新」。"));
        $("nodes").replaceChildren(empty);
      }
    } finally {
      clearTimeout(timeout);
      state.refreshing = false;
      $("refresh").disabled = false;
      scheduleRefresh();
    }
  }

  $("refresh").addEventListener("click", refresh);
  $("show-history").addEventListener("change", render);
  $("refresh-seconds").addEventListener("change", (event) => {state.userInterval = true; state.refreshSeconds = Number(event.target.value) || 15; if (!state.refreshing) scheduleRefresh();});
  document.addEventListener("visibilitychange", () => {if (document.visibilityState === "visible" && state.nextRefresh && Date.now() >= state.nextRefresh) refresh();});
  setInterval(updateCountdown, 1000);
  refresh();
})();
