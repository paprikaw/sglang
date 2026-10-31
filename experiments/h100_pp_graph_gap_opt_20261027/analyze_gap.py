"""Offline reconstruction of complete, Guard-qualified PP scheduling runs."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
NAMES = {0:"Shared GEMM",1:"Early activation",2:"Early + local feedback",
         3:"Early + local + schema",5:"Early + typed channels",
         6:"Early + typed channels + activation stream",7:"Early next-step metadata",8:"PP lookahead",9:"PP lookahead, wider safe window"}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def table(path,rows):
    if not rows:return
    with path.open("w") as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def analyze(run,out,drop_graphs=4,selected_count=16):
    assert drop_graphs>=4 and selected_count>0
    stop=drop_graphs+selected_count
    out.mkdir(parents=True,exist_ok=True)
    time_tools=load("gap_timing_tools",ROOT.parent/"h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/analyze.py")
    profile_loader=load("gap_profile_loader",ROOT.parent/"h100_pp_expert_fragmentation_20261026/analyze.py")
    tools=profile_loader.load_analysis()
    previous=tools.graph_bucket
    tools.graph_bucket=lambda name:"expert.layout_adapter" if name=="_metadata" or name.startswith("_metadata<") else previous(name)
    visibility=load("gap_visibility",Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py"))
    cfg=read(run/"config.json");progress=read(run/"task/progress.json")
    assert progress["stage"]=="complete",progress
    checks=[read(f) for f in (run/"dashboard_guard/checks").glob("*-post.json")]
    assert len(checks)==1 and checks[0]["passed"] and checks[0]["cleanup_verified"]
    assert read(run/"task/cleanup.json")["passed"]
    for entry in read(run/"dashboard_spec.json")["frozen_inputs"]:
        assert sha(entry["path"])==entry["sha256"],entry
    metrics,cycles,profiles,components,sources,receipts,audits=[],[],[],[],[],[],[]
    for arm in cfg["arm_order"]:
        assert read(run/arm/"completed.json")["passed"]
        assert read(run/arm/"cleanup.json")["cleanup_verified"]
        agreement=read(run/arm/"output_agreement.json")
        if arm=="pp2":assert agreement["all_modes_equal"],agreement
        policies=cfg["gap_policies"] if arm=="pp2" else [0]
        for policy in policies:
            path=run/arm/f"metrics.decode.p{policy}.json"
            data,receipt=visibility.wait_for_stable_json(path,timeout_s=15);receipts.append(receipt)
            groups=data["rounds"];rows=[r for g in groups for r in g["requests"]]
            assert len(groups)==cfg.get("samples_per_policy",3)
            assert all(len(g["requests"])==8 for g in groups)
            assert all(r["cached_tokens"]==4096 and r["output_tokens"]==128 and len(r["output_ids"])==128 for r in rows)
            audit=time_tools.formal_stream_audit(path);audits.append(dict(arm=arm,policy=policy,**audit))
            assert audit["repeated_completion_count_events"]==0
            assert abs(audit["reported_tpot_ms"]-audit["unique_token_tpot_ms"])<1e-8
            if arm=="pp2":assert read(run/arm/f"qualification.p{policy}.json")["passed"]
            means=[st.mean(r["tpot_ms"] for r in g["requests"]) for g in groups]
            rate=[1024/g["makespan_s"] for g in groups]
            metrics.append(dict(arm=arm,policy=policy,name=NAMES[policy],cohorts=len(groups),
                tpot_ms=st.mean(r["tpot_ms"] for r in rows),cohort_tpot_min_ms=min(means),cohort_tpot_max_ms=max(means),
                output_tok_s=st.mean(rate),output_tok_s_stdev=st.stdev(rate),
                ttft_ms=1000*st.mean(r["ttft_s"] for r in rows),e2e_ms=1000*st.mean(r["e2e_s"] for r in rows),
                source=str(path),sha256=sha(path)))
            manifest,receipt=visibility.wait_for_stable_json(run/arm/"traces"/f"p{policy}"/"manifest.json",timeout_s=15)
            receipts.append(receipt);entries=sorted(manifest["files"],key=lambda x:x["path"])
            traces=[time_tools.read_trace(Path(e["path"]),e["sha256"]) for e in entries]
            sources+=entries
            assert len(traces)==2 and traces[0]["host"]==traces[1]["host"] and traces[0]["base_time"]==traces[1]["base_time"]
            local=[]
            for rank,(entry,trace) in enumerate(zip(entries,traces)):
                value,detail=tools.trace_components(Path(entry["path"]),arm,"native","H100",stop-4)
                offset=drop_graphs-4
                selected=detail["samples"][offset:]
                assert len(selected)==selected_count
                value.update({k:st.mean(row.get(k,0) for row in selected) for k in {k for row in selected for k in row}})
                value.update(selected_forwards=selected_count,dropped_initial_decode_forwards=drop_graphs,
                             actual_layer_invocations=selected_count*value["layers_per_rank"])
                if drop_graphs!=4:
                    # The legacy boundary-residency scope covers the complete
                    # prefix returned by its parser, not this later window.
                    for key in list(value):
                        if key.startswith("pp_boundary_"):value.pop(key)
                assert [g["launch"] for g in trace["graphs"][drop_graphs:stop]]==[g["cpu_start_us"] for g in detail["actual_graphs"][offset:]]
                components.append({**value,"arm":arm,"policy":policy,"rank":rank})
            if arm=="pp2":
                a,b=[t["graphs"] for t in traces];remote={g["key"]:g for g in b}
                for i in range(drop_graphs,stop):
                    g,n=a[i],a[i+2];r=remote[g["key"]]
                    assert n["key"]==(g["key"][0]+4,g["key"][1])
                    assert g["end"]<=r["start"]<r["end"]<=n["start"]
                    row=dict(stage0_graph_ms=(g["end"]-g["start"])/1000,
                        forward_handoff_ms=(r["start"]-g["end"])/1000,
                        stage1_graph_ms=(r["end"]-r["start"])/1000,
                        feedback_ms=(n["start"]-r["end"])/1000)
                    total=(n["start"]-g["start"])/1000
                    assert abs(sum(row.values())-total)<1e-8
                    local.append(dict(arm=arm,policy=policy,sequence_sum=g["key"][0],occurrence=g["key"][1],**row,cycle_ms=total))
            else:
                for rank,trace in enumerate(traces):
                    a=trace["graphs"]
                    for i in range(drop_graphs,stop):
                        g,n=a[i],a[i+1]
                        assert n["key"][0]==g["key"][0]+4
                        local.append(dict(arm=arm,policy=policy,rank=rank,sequence_sum=g["key"][0],
                            model_graphs_ms=(g["end"]-g["start"])/1000,
                            between_graphs_ms=(n["start"]-g["end"])/1000,cycle_ms=(n["start"]-g["start"])/1000))
            record=dict(arm=arm,policy=policy,name=NAMES[policy],samples=len(local))
            record.update({k:st.mean(r[k] for r in local) for k in local[0] if k.endswith("_ms")})
            if arm=="pp2":
                record["model_graphs_ms"]=record["stage0_graph_ms"]+record["stage1_graph_ms"]
                record["between_graphs_ms"]=record["forward_handoff_ms"]+record["feedback_ms"]
                record["conditional_compute_bound_ms"]=2*max(record["stage0_graph_ms"],record["stage1_graph_ms"])
            for rank,trace in enumerate(traces):
                a=trace["graphs"];lo,hi=a[drop_graphs]["start"],a[stop]["start"]
                record[f"rank{rank}_graph_fraction"]=time_tools.union(time_tools.clipped([(g["start"],g["end"]) for g in a],lo,hi))/(hi-lo)
            cycles+=local;profiles.append(record)
    result=dict(run=str(run),model=cfg["model_path"],node=cfg["node"],physical_gpus=[0,1],
        pp_ring_slots=cfg["pp_loop_slots"],formal_metrics=metrics,independent_profile=profiles,
        profile_window=dict(drop_initial_graphs=drop_graphs,selected_forwards_per_rank=selected_count),
        components=components,stream_audits=audits,trace_sources=sources,visibility=receipts,
        scope="Formal profiler-off metrics and separately profiled token cycles; profile gaps are not pure link transfer times",
        checks=dict(frozen_inputs=True,guard_postcheck=True,owned_cleanup=True,pp_output_tokens_equal=True),
        analyzer=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    save(out/"results.json",result);table(out/"metrics.csv",metrics);table(out/"profile_cycles.csv",cycles)
    table(out/"profile_summary.csv",profiles);table(out/"components.csv",components)
    print(json.dumps(dict(run=run.name,metrics=metrics,profile=profiles),indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path)
    parser.add_argument("--drop-graphs",type=int,default=4);parser.add_argument("--forwards",type=int,default=16)
    args=parser.parse_args();analyze(args.run.resolve(),args.out or ROOT/"analysis"/args.run.name,args.drop_graphs,args.forwards)
