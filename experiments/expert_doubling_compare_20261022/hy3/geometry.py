"""Actual received rows; balanced active-expert counts, no TP duplication."""
import math
import statistics

HIDDEN=4096
INTERMEDIATE=1536
TOKENS=tuple(2**p for p in range(3,14))
EXPERTS=tuple(2**p for p in range(6))
BYTES_PER_EXPERT=3*HIDDEN*INTERMEDIATE*2


def counts(total, active, resident=48):
    assert 0 < active <= min(total,resident)
    ids=[i*resident//active for i in range(active)]
    result=[0]*resident
    for i,e in enumerate(ids):result[e]=total//active+(i<total%active)
    assert sum(result)==total
    return result


def grid():
    return [(t,e) for t in TOKENS for e in EXPERTS if e<=t]


def model_features(total, active, block_m):
    cs=counts(total,active)
    tiles=sum(math.ceil(n/block_m) for n in cs if n)
    return dict(useful_rows=total, active_experts=active, m_tiles=tiles,
                padded_rows=tiles*block_m, padding_ratio=tiles*block_m/total,
                gemm_flops=6*HIDDEN*INTERMEDIATE*total,
                active_weight_bytes=active*BYTES_PER_EXPERT)


def summary(values):
    x=sorted(values)
    def q(p):
        z=(len(x)-1)*p;l=int(z);h=min(l+1,len(x)-1)
        return x[l]+(x[h]-x[l])*(z-l)
    return dict(n=len(x),mean_ms=statistics.mean(x),median_ms=statistics.median(x),
                p95_ms=q(.95),p99_ms=q(.99),min_ms=x[0],max_ms=x[-1])
