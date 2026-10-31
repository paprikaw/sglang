"""Use Humming's cp.async path for grouped AWQ zero points on H100.

The installed grouped zero-point TMA launcher passes a rank-three tensor with
only two box dimensions. Change this supported tuning option before compilation;
leave the quantized weights and all other tuning choices unchanged.
"""
import json


def zero_point_tma_fallback(configs):
    result = dict(configs)
    for name in ("w13_tuning_config", "w2_tuning_config"):
        source = configs[name]
        if isinstance(source, dict):
            tuning = dict(source, use_tma_bzp=False)
        else:
            # Native heuristics return (min_M, max_M, tuning_dict) entries.
            tuning = [[lo, hi, dict(entry, use_tma_bzp=False)]
                      for lo, hi, entry in source]
        result[name] = tuning
        result[name + "_str"] = json.dumps(tuning)
    return result
