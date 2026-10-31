"""Common model-runner load metric, independent of loader-specific counters."""
import math
import re

END = re.compile(r"\b(?:PP(?P<pp>\d+)\s+)?TP(?P<rank>\d+)(?:\s|\]).*?Load weight end\. elapsed=(?P<elapsed>[0-9.]+) s")
BEGIN = re.compile(r"\b(?:PP(?P<pp>\d+)\s+)?TP(?P<rank>\d+)(?:\s|\]).*?Load weight begin\.")


def extract_weight_load(text, expected_ranks):
    begins, ends = {}, {}
    for line in text.splitlines():
        match = BEGIN.search(line)
        if match:
            rank = int(match["rank"]) + 4*int(match["pp"] or 0)
            begins[rank] = begins.get(rank, 0) + 1
        match = END.search(line)
        if match:
            rank = int(match["rank"]) + 4*int(match["pp"] or 0)
            if rank in ends:
                raise ValueError("Duplicate weight-load completion")
            value = float(match["elapsed"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid load duration")
            ends[rank] = value
    expected = set(expected_ranks)
    if set(ends) != expected or set(begins) != expected or any(v != 1 for v in begins.values()):
        raise ValueError("Load timing lacks exact expected rank coverage")
    return {"metric": "model_runner_load_weight_elapsed_s",
            "boundary": "Load weight begin to Load weight end; model allocation, loader and parameter copies included",
            "per_rank_s": {str(rank): ends[rank] for rank in sorted(ends)},
            "slowest_rank_s": max(ends.values()), "rank_count": len(ends),
            "durations_summed_across_ranks": False}
