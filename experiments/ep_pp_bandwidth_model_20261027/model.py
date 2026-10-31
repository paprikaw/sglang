"""Two-GPU Expert/communication service model; SI bytes and seconds throughout.

PP2 has two balanced one-GPU stages and two microbatches of b tokens.
All times are per full-model layer equivalent in a steady B=2b wave.
This is a memory-bound model, not an end-to-end latency predictor.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Hardware:
    name: str
    hbm_bytes_s: float


A100 = Hardware("A100 80GB PCIe", 1.935e12)
H100 = Hardware("H100 80GB SXM", 3.35e12)


def expert_weight_bytes(hidden, intermediate, bytes_per_weight):
    """SwiGLU gate, up and down; excludes quantization scales/shared Experts."""
    return 3 * hidden * intermediate * bytes_per_weight


def uniform_activation(experts, top_k, tokens_per_microbatch):
    """Independent tokens; top-k selects distinct Experts within each token."""
    a = -math.expm1(tokens_per_microbatch * math.log1p(-top_k / experts))
    return {"microbatch_fraction": a, "union_fraction": 2*a-a*a,
            "overlap_fraction": a*a}


def communication_time(payload_bytes, one_way_bytes_s, fixed_seconds=0):
    """D+C serialized send bytes/rank; full-duplex receive is not counted twice."""
    return fixed_seconds + payload_bytes / one_way_bytes_s


def layer_times(weight_bytes, a1, a2, union_fraction, hbm_bytes_s, dc_seconds):
    overlap = a1 + a2 - union_fraction
    if not (0 <= a1 <= 1 and 0 <= a2 <= 1 and
            max(a1, a2)-1e-12 <= union_fraction <= min(1, a1+a2)+1e-12):
        raise ValueError("Invalid active Expert sets")
    ep = weight_bytes * union_fraction / (2*hbm_bytes_s) + dc_seconds
    pp = weight_bytes * (a1+a2) / (2*hbm_bytes_s)
    return {"ep_seconds": ep, "pp_seconds": pp,
            "extra_pp_weight_seconds": weight_bytes * max(0, overlap)/(2*hbm_bytes_s),
            "pp_advantage_seconds": ep-pp}


def critical_weight_bytes(hbm_bytes_s, dc_seconds, overlap_fraction):
    return (2*hbm_bytes_s*dc_seconds / overlap_fraction
            if overlap_fraction > 0 else math.inf)


def critical_expert_count(hbm_bytes_s, dc_seconds, bytes_per_expert):
    """Maximum shared active Expert count R at the PP/EP break-even."""
    return 2*hbm_bytes_s*dc_seconds / bytes_per_expert


def critical_network_bytes_s(weight_bytes, overlap_fraction, hbm_bytes_s,
                             payload_bytes, fixed_seconds=0):
    weight_penalty = weight_bytes*overlap_fraction/(2*hbm_bytes_s)
    return (math.inf if weight_penalty <= fixed_seconds
            else payload_bytes/(weight_penalty-fixed_seconds))
