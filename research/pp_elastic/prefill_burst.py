"""Bounded AgentX P-side reactive run. The official client owns arrivals and dependencies."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pd_elastic_smoke as base

TRACE = base.PRIOR
SLO_TTFT_S = 15


def prepare_files(run, cfgs, cache_gate, spec, paths):
    settings = json.loads((TRACE / "config.json").read_text())
    source = TRACE / "harness"
    harness = run / "harness"
    harness.mkdir()
    for f in source.glob("*.py"):
        shutil.copyfile(f, harness / f.name)
    paired = base.ENGINE / "research/pd_reactive_e2e/paired_cache_bust.py"
    shutil.copyfile(paired, harness / paired.name)
    with (harness / "sitecustomize.py").open("a") as f:
        f.write('\nif os.environ.get("AGENTX_PAIRED_NAMESPACE"):\n    from paired_cache_bust import install as freeze_cache_marker\n    freeze_cache_marker(os.environ["AGENTX_PAIRED_NAMESPACE"])\n')
    length = json.loads((cache_gate / "results.json").read_text())
    if not (length["passed"] and length["input_tokens"] == 54949 and length["output_tokens"] == 142):
        raise RuntimeError("Length gate has not passed")
    prior_guard = cache_gate / "dashboard_guard/checks"
    post = list(prior_guard.glob("*-post.json"))
    if not any(json.loads(f.read_text()).get("passed") for f in post):
        raise RuntimeError("Length gate postcheck is missing")
    plan = TRACE / "plan.json"
    data = TRACE / "data/traces.jsonl"
    base.save(run / "protocol.json", dict(plan=str(plan), dataset=str(data), client=settings["agentx_python"],
         model=settings["model_path"], tokenizer=settings["agentx_tokenizer"], tokenizer_cache=settings["tokenizer_cache"],
         namespace=hashlib.sha256(plan.read_bytes()+data.read_bytes()).hexdigest(),
         service_tokens_s=length["input_tokens"] / (length["ttft_ms"]/1000), high_wait_s=SLO_TTFT_S/4,
         sample_s=0.5, high_hold_s=1, low_hold_s=10, cooldown_s=5, base_capacity=65536,
         ttft_s=15, tpot_ms=150, mechanism="inplace", fixed_decode="PP2", gate=str(cache_gate)))
    def endpoint(c):
        return dict(name=c["role"], url=f'http://{c["node"]}:{c["port"]}', attention_dp_size=1,
             owner_weight=1, **(dict(bootstrap_host=c["node"], bootstrap_port=c["bootstrap_port"])
                              if c["role"]=="prefill" else {}))
    ep={c["role"]:[endpoint(c)] for c in cfgs.values()}
    base.save(run/"endpoints.json", dict(mode="pd", require_hicache=True, api_key_file=str(run/"admin.key"),
                                       request_timeout_s=300, backend_force_close=True, **ep))
    paths.extend([run/"protocol.json",run/"endpoints.json",data,plan,cache_gate/"results.json",*post,*sorted(harness.glob("*.py"))])
    spec.update(title="122B AgentX plus BurstGPT, elastic Prefill, fixed PP2 Decode",
            implementation_note="Long-context extension of validated r11 Prefill PP elasticity and fixed Decode. Native AgentX 1.0.7 finite replay with stable cache markers. One observed-queue reactive controller. Cold non-PD-only scheduler recipes are outside this disaggregated path.")


def run_client(run):
    cfg=json.loads((run/"protocol.json").read_text())
    loc=base.node_config(json.loads((run/"config.json").read_text()))
    base.h.authenticate(run,loc,os.environ,local=True)
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_CACHE=cfg["tokenizer_cache"],
             PYTHONPATH=str(run/"harness"),AGENTX_MIXED_CONFIG_PATH=cfg["plan"],
             AGENTX_PAIRED_NAMESPACE=cfg["namespace"],AIPERF_DATASET_WEKA_PARALLEL_WORKERS="2")
    args=[cfg["client"], str(run/"harness/run.py"),"run","--config",str(run/"endpoints.json"),
          "--model",cfg["model"],"--tokenizer",cfg["tokenizer"],"--local-dataset",cfg["dataset"],"--output",str(run/"replay"),
          "--entries","1","--concurrency","6","--duration","720","--profile-grace","150","--warmup-grace","180",
          "--warmup-prefill-limit","2","--wall-timeout","1050","--smoke"]
    os.execve(args[0],args,env)


def measure(run,p,d,key,replay,health):
    cfg=json.loads((run/"protocol.json").read_text())
    base.save(run/"start-client.json",dict(unix=time.time()))
    up=False
    generation=0
    high_since=low_since=None
    phase_start=None
    cooldown=0
    actions=[]
    deadline=time.monotonic()+1200
    def snap(e):
        loads=replay.http(e["host"],e["port"],"/v1/loads",key,timeout=8)["loads"]
        if len(loads)!=1:
            raise ValueError("Expected one attention-DP load snapshot")
        return loads[0]
    with (run/"loads.jsonl").open("w",buffering=1) as log:
        while True:
            health()
            if time.monotonic()>deadline: raise TimeoutError("Bounded AgentX replay exceeded deadline")
            done=run/"client_exit.json"
            if done.is_file() and json.loads(done.read_text())["returncode"]!=0:
                raise RuntimeError("AgentX client exited unsuccessfully")
            if phase_start is None:
                for event_file in (run/"replay/events").glob("*.jsonl"):
                    for row in event_file.read_text().splitlines():
                        if '"phase_start"' in row:
                            phase_start=json.loads(row)["unix_ns"]/1e9
                            break
            if phase_start is None:
                if done.is_file(): raise RuntimeError("No profiling phase")
                time.sleep(.5)
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pf=pool.submit(snap,p)
                df=pool.submit(snap,d)
                ps,ds=pf.result(),df.result()
            stamp=time.time()
            log.write(json.dumps(dict(unix=stamp,P=ps,D=ds))+"\n")
            wait=ps["num_waiting_uncached_tokens"]/cfg["service_tokens_s"]
            high=wait>=cfg["high_wait_s"] and ps["queues"]["waiting"]>0
            quiet=(ps["num_waiting_uncached_tokens"]==0 and ps["num_running_reqs"]==0 and
                   ps["num_waiting_reqs"]==0 and ps["disaggregation"]["prefill_inflight_queue_reqs"]==0)
            high_since=(high_since or stamp) if high else None
            low_since=(low_since or stamp) if quiet else None
            grow=not up and high_since is not None and stamp-high_since>=cfg["high_hold_s"]
            shrink=up and low_since is not None and stamp-low_since>=cfg["low_hold_s"]
            if (grow or shrink) and stamp>=cooldown:
                counts=[16,16,16] if grow else [24,0,24]
                generation+=1
                row=dict(started=stamp,decision=dict(role="prefill",direction="up" if grow else "down"),
                         backlog_s=wait, generation=generation, target=counts)
                actions.append(row);base.save(run/"actions.json",actions)
                try:
                    resp=replay.http(p["host"],p["port"],"/reconfigure_elastic_pp",key,
                                     dict(generation=generation,layer_counts=counts),120)
                    if not resp.get("success") or resp.get("new_layer_counts")!=counts:
                        raise RuntimeError(str(resp)[:450])
                    row["response"]=resp
                except Exception as exc:
                    row["error"]=str(exc)
                    raise
                finally:
                    row["finished"]=time.time();base.save(run/"actions.json",actions)
                cooldown=row["finished"]+cfg["cooldown_s"]; up=grow
                high_since=low_since=None
            if done.is_file() and not up: break
            time.sleep(cfg["sample_s"])
    rows=[json.loads(row) for row in (run/"replay/agentx/profile_export.jsonl").read_text().splitlines()]
    formal=[row for row in rows if row["metadata"]["benchmark_phase"]=="profiling"]
    routes={r["x_request_id"]:r for r in (json.loads(line) for line in (run/"replay/routing.jsonl").read_text().splitlines())}
    passed=len(formal)==97 and all(routes.get(row["metadata"]["x_request_id"],{}).get("success")
                                    and not row["metadata"].get("was_cancelled") for row in formal)
    return dict(passed=passed,formal_requests=len(formal),expected=97,actions=len(actions),phase_start=phase_start)
