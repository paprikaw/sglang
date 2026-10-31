"""Normalize saved KV capacity to independent full-model prefix-cache pools."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
SOURCE = WORK/'experiments/agentx_pd_dynamic_20261030/matrix_analysis/cache_capacities.csv'


def main():
    with SOURCE.open() as stream:
        all_rows = list(csv.DictReader(stream))
    results, evidence = [], []
    for arm in ('col_single','col_pp2','col_ep4'):
        rows = [r for r in all_rows if r['arm']==arm]
        assert len(rows)==8
        pools = collections.defaultdict(list)
        for row in rows:
            key = row['node'],row['instance']
            if arm=='col_ep4':
                key += (row['rank_label'].split()[0],)
            pools[key].append(row)
        capacities = [min(int(r['tokens']) for r in pool) for pool in pools.values()]
        slots = set()
        for source in sorted({r['source'] for r in rows}):
            path = Path(source)
            content = path.read_bytes()
            text = content.decode()
            found = re.findall(r'Mamba Cache is allocated\. max_mamba_cache_size: (\d+)',text)
            assert found,source
            slots.update(int(x) for x in found)
            evidence.append(dict(path=source,sha256=hashlib.sha256(content).hexdigest(),
                allocated_gdn_slots=[int(x) for x in found]))
        assert len(slots)==1
        results.append(dict(arm=arm,gpus=8,independent_cache_pools=len(pools),
            per_pool_gpu_token_capacity_min=min(capacities),
            per_pool_gpu_token_capacity_max=max(capacities),
            total_gpu_logical_token_capacity=sum(capacities),
            gdn_slots_per_owner=next(iter(slots)),
            gdn_slot_scope='Each PP stage stores its own layers' if arm=='col_pp2' else 'Full-model recurrent state per Attention owner',
            c128_root_families_per_pool=128//len(pools)))
    with (HERE/'cache_layout_resources.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    result=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        pools=results,allocation_log_evidence=evidence,
        normalization='PP stage token slots describe the same requests; take the smaller stage capacity and count once per instance. EP4 Attention-DP ranks are independent pools.',
        causal_limit='Capacities, GDN slots and owner placement are observed; no controlled decomposition of cache misses into eviction, state loss and scheduling was performed.')
    (HERE/'cache_layout_resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':
    main()
