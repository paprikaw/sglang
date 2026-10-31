"""Recompute timing statistics and native Decode groups without model imports."""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def audit(run):
    result = {"passed": True, "arms": {}}
    for layout in ("ep8", "pp2ep4"):
        point = json.loads((run / layout / "p81920_decode_results.json").read_text())
        tails = []
        for request in point["requests"]:
            events = request["token_events"]
            assert [e[0] for e in events] == list(range(1, 129))
            assert all(a[1] <= b[1] for a, b in zip(events, events[1:]))
            close(events[0][1], request["ttft_s"])
            close((events[-1][1] - events[0][1]) / 127, request["tpot_s"])
            # E2E includes reading DONE and closing the response after token 128.
            tail = request["end_to_end_s"] - events[-1][1]
            assert 0 <= tail < 0.05
            tails.append(tail)
            assert request["meta"]["cached_tokens_details"] == {"device": 0, "host": 81920}
            assert request["meta"]["num_retractions"] == 0
        for metric in ("ttft_s", "tpot_s", "end_to_end_s"):
            values = [r[metric] for r in point["requests"]]
            for name, value in (("mean", statistics.mean(values)), ("median", statistics.median(values)), ("p99", max(values))):
                close(value, point[metric][name])
        close(1024 / point["wall_s"], point["throughput_tok_s"])
        ranks = []
        for rank in range(8):
            receipt = json.loads((run / layout / "receipts" / f"p81920_decode_after_rank{rank}.json").read_text())
            assert receipt["passed"]
            batches = [b for b in receipt["batches"] if str(b["mode"]) == "2"]
            assert all(b["batch_size"] == len(b["request_ids"]) == 4 for b in batches)
            groups = collections.Counter(tuple(sorted(b["request_ids"])) for b in batches)
            assert len(groups) == (1 if layout == "ep8" else 2)
            assert set(groups.values()) == {127}
            assert sum(groups.values()) == (127 if layout == "ep8" else 254)
            flat = [rid for group in groups for rid in group]
            assert len(flat) == len(set(flat))
            ranks.append({"rank": rank, "decode_launches": len(batches), "batch_size": 4,
                          "groups": [{"request_ids": list(k), "decode_steps": v} for k, v in groups.items()]})
        result["arms"][layout] = {"requests": 8, "output_token_events": 1024,
                                  "zero_retractions": True, "response_close_tail_max_s": max(tails),
                                  "actual_decode_groups": ranks}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    if args.output:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps({"passed": result["passed"], "arms": {k: {"requests": v["requests"], "token_events": v["output_token_events"], "close_tail_max_s": v["response_close_tail_max_s"]} for k, v in result["arms"].items()}}))
