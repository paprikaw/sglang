/**
 * Run with: node tools/cache_monitor/test_frontend.mjs
 * Executes the production frontend against a minimal DOM and /api/status fixture.
 * This checks data rendering and interactions, not browser layout or CSS.
 */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.className = "";
    this._text = "";
    this.value = "";
    this.checked = false;
    this.open = false;
    this.style = {};
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  append(...elements) { this.children.push(...elements); }
  prepend(...elements) { this.children.unshift(...elements); }
  replaceChildren(...elements) { this._text = ""; this.children = [...elements]; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  get options() { return this.children; }
  set innerHTML(_) { throw new Error("Use textContent for API data, never innerHTML"); }
}

function byClass(element, className) {
  const result = element.className.split(" ").includes(className) ? [element] : [];
  return result.concat(element.children.flatMap((child) => byClass(child, className)));
}

const elements = {};
const html = fs.readFileSync(new URL("./static/index.html", import.meta.url), "utf8");
for (const match of html.matchAll(/id="([^"]+)"/g)) elements[match[1]] = new Element("div");
for (const seconds of [15, 30, 60]) {
  const option = new Element("option");
  option.value = String(seconds);
  elements["refresh-seconds"].append(option);
}

const GiB = 1024 ** 3;
const fixture = {
  observed_at: 1791846000,
  refresh_seconds: 12,
  collecting: false,
  errors: [],
  nodes: [
    {
      node: "live-a",
      status: "live",
      booking: {state: "RUNNING", job_ids: [1]},
      last_probe_at: 1791846000,
      memory: {total_bytes: 256 * GiB, available_bytes: 180 * GiB},
      weight: {
        bytes: 64 * GiB,
        reserved_bytes: 80 * GiB,
        resident_bytes: 64 * GiB,
        rss_bytes: 20 * 1024 * 1024,
        banks: [{model: "<img src=x onerror=alert(1)>", id: "bank-a", bytes: 64 * GiB}],
      },
      kv: {
        logical_bytes: 16 * GiB,
        resident_bytes: null,
        files: 2,
        host_pool_bytes: null,
        host_pool_note: "活跃 pool 未接入遥测",
        datasets: [{
          id: "kv-a",
          label: "example_ep12",
          root: "/dev/shm/example_ep12/",
          status: "observed",
          registered_boot_matches: false,
          observation_complete: false,
          error: "scan incomplete",
          // EP is deliberately unknown despite the directory name.
          layout: {tp: 4, pp: 1, ep: null, dp: 1},
          logical_bytes: 16 * GiB,
          resident_bytes: null,
          rank_entries: [{
            rank: 0, tp_rank: 0, pp_rank: 0,
            layer_start: 0, layer_end: 47,
            gpu_index: null,
            logical_bytes: 4 * GiB, resident_bytes: null,
            files: 1, tokens: 32768,
          }],
        }],
      },
      gpus: [0, 1, 2, 3].map((index) => ({index, name: "A100", total_bytes: 80 * GiB})),
    },
    {
      node: "unreachable-b", status: "unreachable",
      booking: {state: "RUNNING"}, error: "SSH timeout",
      weight: {bytes: 999 * GiB},
      kv: {logical_bytes: 999 * GiB, resident_bytes: 999 * GiB},
    },
    {
      // A cancelled booking overrides an old 'live' status.
      node: "cancelled-c", status: "live", booking: {state: "CANCELLED"},
      weight: {bytes: 888 * GiB},
      kv: {logical_bytes: 888 * GiB, resident_bytes: 888 * GiB},
    },
    {
      node: "live-d", status: "live", booking: {state: "RUNNING"},
      weight: {bytes: null, error: "identity mismatch"},
      kv: {logical_bytes: 24 * GiB, resident_bytes: 8 * GiB},
      // Missing hardware inventory must not produce imagined GPU cards.
    },
  ],
};

let failRequest = false;
const requestedPaths = [];
const context = {
  document: {
    getElementById: (id) => elements[id],
    createElement: (tagName) => new Element(tagName),
    addEventListener() {},
    visibilityState: "visible",
  },
  fetch: async (path) => {
    requestedPaths.push(path);
    if (failRequest) throw new Error("offline");
    return {ok: true, json: async () => fixture};
  },
  AbortController, Date, Set, String, Number, Math, Array, Object,
  setTimeout: () => 1,
  clearTimeout() {},
  setInterval: () => 1,
};
vm.runInNewContext(fs.readFileSync(new URL("./static/app.js", import.meta.url), "utf8"), context);
await new Promise(setImmediate);

assert.deepEqual(requestedPaths, ["/api/status"]);
assert.equal(elements.summary.children.length, 4);
assert.match(elements.summary.children[0].textContent, /2 \/ 3/);
assert.match(elements.summary.children[1].textContent, /64 GiB.*1 个节点未知/);
assert.match(elements.summary.children[2].textContent, /8 GiB.*1 个节点未知/);
assert.match(elements.summary.children[3].textContent, /40 GiB/);
assert.equal(elements.nodes.children.length, 3);
assert(!elements.nodes.textContent.includes("cancelled-c"));
assert(elements.nodes.textContent.includes("物理 GPU 映射未知"));
assert(elements.nodes.textContent.includes("<img src=x onerror=alert(1)>"));
assert(elements.nodes.textContent.includes("checkpoint 可复用性未验证"));
assert(elements.nodes.textContent.includes("scan incomplete"));
assert(elements.nodes.textContent.includes("identity mismatch"));
assert(elements.nodes.textContent.includes("活跃 HiCache host pool —"));
assert.equal(byClass(elements.nodes.children[0], "gpu-card").length, 4);
assert.equal(byClass(elements.nodes.children[2], "gpu-card").length, 0);
assert(!byClass(elements.nodes.children[0], "layout-chip").some((chip) => /EP\s*12/.test(chip.textContent)));
const weightSegment = byClass(elements.nodes.children[0], "memory-segment")
  .find((segment) => segment.className.split(" ").includes("weight"));
assert.equal(weightSegment.style.width, "25%", "Use 64 GiB resident, not 20 MiB daemon RSS");
assert.equal(elements["refresh-seconds"].value, "12");

elements["show-history"].checked = true;
elements["show-history"].listeners.change();
assert.equal(elements.nodes.children.length, 4);
assert(elements.nodes.children[2].className.includes("historical"));
const firstDetails = elements.nodes.children[0].children[2];
firstDetails.open = true;
firstDetails.listeners.toggle();
elements["show-history"].listeners.change();
assert.equal(elements.nodes.children[0].children[2].open, true);

failRequest = true;
await elements.refresh.listeners.click();
assert(elements.alerts.textContent.includes("保留上次采样"));
assert.equal(elements.nodes.children.length, 4);
assert(elements.connection.className.includes("error"));

console.log("Frontend fixture passed: live-only totals, unknowns, cancelled/history nodes, GPU mapping, no invented hardware or EP, resident vs RSS, host-pool unknown, identity/boot/partial errors, text-only rendering, interval, expansion, fetch failure.");
