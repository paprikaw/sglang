"""Audit phase tradeoffs and the SLO grid frozen in each AgentX replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from completed_episode_files import completed_formals, gate_sha256


def write_csv(path, rows):
    if rows:
        with path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    elif path.exists():
        path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    trials, slo_trials, ready_slo_trials, provenance = [], [], [], []
    files, pending = completed_formals(run)
    for p in files:
        arm, case, policy = p.relative_to(run).parts[1:4]
        f = json.loads(p.read_text()); rows = f["requests"]
        after_first = sum(r["e2e_ms"] - r["ttft_ms"] for r in rows)
        generated_intervals = sum(r["output_tokens"] - 1 for r in rows)
        identity = dict(arm=arm, case=case, policy=policy, repeat=p.stem.split(".")[-1])
        trials.append(dict(**identity, requests=len(rows),
            ttft_ms=st.mean(r["ttft_ms"] for r in rows),
            request_mean_tpot_ms=st.mean(r["tpot_ms"] for r in rows if r["tpot_ms"] is not None),
            interval_weighted_tpot_ms=after_first/generated_intervals,
            mean_after_first_ms=after_first/len(rows),
            e2e_ms=st.mean(r["e2e_ms"] for r in rows),
            client_dispatch_lag_ms=st.mean(r["client_dispatch_lag_ms"] for r in rows),
            ready_ttft_ms=st.mean(r["ttft_ms"] + r["client_dispatch_lag_ms"] for r in rows),
            ready_e2e_ms=st.mean(r["e2e_ms"] + r["client_dispatch_lag_ms"] for r in rows),
            wall_s=f["wall_s"]))
        for s in f["slo"]:
            n = sum(r["ttft_ms"] <= s["ttft_ms"] and
                (r["tpot_ms"] is None or r["tpot_ms"] <= s["tpot_ms"]) for r in rows)
            assert n == s["passed_requests"]
            assert abs(n/f["wall_s"] - s["goodput_rps"]) < 1e-8
            slo_trials.append(dict(**identity, ttft_limit_ms=s["ttft_ms"],
                tpot_limit_ms=s["tpot_ms"], passed_requests=n,
                fraction_passed=n/len(rows), goodput_rps=n/f["wall_s"]))
            ready_n = sum(r["ttft_ms"] + r["client_dispatch_lag_ms"] <= s["ttft_ms"] and
                (r["tpot_ms"] is None or r["tpot_ms"] <= s["tpot_ms"]) for r in rows)
            assert ready_n <= n
            ready_slo_trials.append(dict(**identity, ttft_limit_ms=s["ttft_ms"],
                tpot_limit_ms=s["tpot_ms"], passed_requests=ready_n,
                fraction_passed=ready_n/len(rows), goodput_rps=ready_n/f["wall_s"]))
        provenance.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    def aggregate(rows, fields, values):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[k] for k in fields), []).append(r)
        return [dict(**dict(zip(fields, key)), repeats=len(g), complete_point=len(g)==cfg["repeats"],
                **{v:st.mean(r[v] for r in g) for v in values}) for key,g in groups.items()]
    means = aggregate(trials, ["arm","case","policy"],
        ["ttft_ms","request_mean_tpot_ms","interval_weighted_tpot_ms","mean_after_first_ms","e2e_ms",
         "client_dispatch_lag_ms","ready_ttft_ms","ready_e2e_ms","wall_s"])
    slo_means = aggregate(slo_trials, ["arm","case","policy","ttft_limit_ms","tpot_limit_ms"],
        ["passed_requests","fraction_passed","goodput_rps"])
    ready_slo_means = aggregate(ready_slo_trials, ["arm","case","policy","ttft_limit_ms","tpot_limit_ms"],
        ["passed_requests","fraction_passed","goodput_rps"])
    out = run/"analysis"; out.mkdir(exist_ok=True)
    for name, rows in [("phase_trials.csv",trials),("phase_means.csv",means),
                       ("slo_trials.csv",slo_trials),("slo_means.csv",slo_means),
                       ("ready_slo_trials.csv",ready_slo_trials),("ready_slo_means.csv",ready_slo_means)]:
        write_csv(out/name,rows)
    text = ["# AgentX phase and latency-target tradeoffs", "",
        "Only episodes with passed completion markers enter this table. Every such repeat is retained. Only rows marked complete have all planned repeats; whole-run qualification and cleanup remain separate. "
        "The SLO grid is the existing grid frozen in the replay; it is not an official AgentX score.", "",
        "| Arm | Case | Repeats | TTFT ms | Request mean TPOT ms/token | Interval-weighted TPOT ms/token | Mean after-first-token ms | E2E ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in means:
        text.append(f"| {m['arm']} | {m['case']} | {m['repeats']} | {m['ttft_ms']:.2f} | {m['request_mean_tpot_ms']:.3f} | {m['interval_weighted_tpot_ms']:.3f} | {m['mean_after_first_ms']:.2f} | {m['e2e_ms']:.2f} |")
    text += ["", "A request-mean TPOT gives a short output and a long output equal weight. "
        "The interval-weighted value equals total after-first-token request time divided by total generated-token intervals; "
        "neither is a device kernel time. Mean E2E equals mean TTFT plus mean after-first-token time.", "",
        "## Frozen SLO grid", "",
        "| Arm | Case | Repeats | TTFT limit ms | TPOT limit ms/token | Mean pass fraction | Mean achieved requests/s |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for m in slo_means:
        text.append(f"| {m['arm']} | {m['case']} | {m['repeats']} | {m['ttft_limit_ms']} | {m['tpot_limit_ms']} | {m['fraction_passed']:.4f} | {m['goodput_rps']:.4f} |")
    text += ["", "Achieved requests/s includes tool waits in this finite closed-loop replay; it is not a saturation-capacity estimate. "
        "A TPOT target here applies to a request's average TPOT, not every inter-token gap. "
        "The initial pilot includes first-use effects and must not be pooled with explicitly warmed runs."]
    text += ["", "## Request-ready boundary", "",
        "The replay has a finite client thread pool. The table below also includes recorded delay from DAG readiness "
        "to HTTP worker start, including client-side preparation/queueing. It excludes authored tool waits before readiness. "
        "This is a separate boundary, not a redefinition of the original HTTP TTFT/E2E above.", "",
        "| Arm | Case | Repeats | Client dispatch lag ms | Ready-to-first-token ms | Ready-to-completion ms |",
        "|---|---|---:|---:|---:|---:|"]
    for m in means:
        text.append(f"| {m['arm']} | {m['case']} | {m['repeats']} | {m['client_dispatch_lag_ms']:.2f} | {m['ready_ttft_ms']:.2f} | {m['ready_e2e_ms']:.2f} |")
    text += ["", "[Same target grid including client dispatch lag](ready_slo_means.csv). "
        "Both definitions are retained so a smaller HTTP queue cannot hide a larger replay-side queue."]
    (out/"TRADEOFFS.md").write_text("\n".join(text)+"\n")
    (out/"tradeoff_audit.json").write_text(json.dumps(dict(slo_recomputed=True, ready_boundary_also_reported=True,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=provenance,
        excluded_unmarked_formals=pending, completion_gate_sha256=gate_sha256(),
        episodes=len(trials), complete_points=sum(m["complete_point"] for m in means)),indent=2)+"\n")
    print(json.dumps(dict(episodes=len(trials), complete_points=sum(m["complete_point"] for m in means))))


if __name__ == "__main__":
    main()
