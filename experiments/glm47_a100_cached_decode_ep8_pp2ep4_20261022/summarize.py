"""Accept only complete paired results plus Guard's scientific/cleanup postcheck."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def is_decode_mode(mode):
    # ForwardMode is an IntEnum: receipts serialize DECODE as "2".
    return str(mode) in {'2','DECODE','ForwardMode.DECODE'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='accepted');a=p.parse_args();run=a.run.resolve()
    assert Path(a.output_name).name==a.output_name and a.output_name not in {'.','..'}
    result=load(run/'task/results.json');assert result['passed'] and len(result['arms'])==2
    checks=[load(x) for x in (run/'dashboard_guard/checks').glob('*-post.json')]
    assert len(checks)==1 and checks[0]['passed'] and checks[0]['scientific_success'] and checks[0]['cleanup_verified']
    assert checks[0]['model_cache']['passed']
    output=run/a.output_name;output.mkdir(exist_ok=False)
    points=[];requests=[];evidence={}
    for arm in result['arms']:
        assert arm['passed'] and arm['cleanup_verified'];row=arm['results'][0];layout=arm['layout']
        assert row['workload']=='decode' and row['request_count']==8 and row['prefix_tokens_per_request']==81920
        assert row['output_tokens_per_request']==128 and row['hicache_h2d_timed'] and not row['weights_loading_timed']
        for r in row['requests']:
            assert r['completion_tokens']==128 and r['meta']['cached_tokens']==81920
            assert r['meta']['cached_tokens_details']=={'device':0,'host':81920}
            requests.append(dict(layout=layout,request_index=r['request_index'],ttft_s=r['ttft_s'],tpot_ms=1000*r['tpot_s'],end_to_end_s=r['end_to_end_s'],num_retractions=r['meta'].get('num_retractions')))
        caps=[load(run/layout/'capacity'/f'rank{r}.json') for r in range(8)]
        actual=[]
        for r in range(8):
            assert load(run/layout/f'weights_rank{r}.json')['passed'] and load(run/layout/f'route_rank{r}.json')['passed']
            proof=load(run/layout/'receipts'/f'p81920_decode_cold_rank{r}.json')
            assert proof['passed'] and proof['native_hicache_quiescent']
            assert all(m['gpu_tokens']==0 and m['cpu_tokens']==81920 for m in proof['matches'])
            after=load(run/layout/'receipts'/f'p81920_decode_after_rank{r}.json')
            batches=after['batches'];dec=[b for b in batches if is_decode_mode(b['mode'])]
            assert len(dec)==(127 if layout=='ep8' else 254)
            assert all(b['batch_size']==4 for b in dec)
            actual.append(dict(rank=r,capacity=caps[r]['token_capacity'],decode_batch_sizes=sorted(set(b['batch_size'] for b in dec)),decode_batch_count=len(dec)))
        evidence[layout]=dict(capacities=caps,actual_batches=actual,weight_load=arm['weights_load'])
        point=dict(layout=layout,output_tok_s=row['throughput_tok_s'],wall_s=row['wall_s'])
        for metric,label,scale in [('ttft_s','ttft_s',1),('tpot_s','tpot_ms',1000),('end_to_end_s','end_to_end_s',1)]:
            for stat in ['mean','median','p99']:point[label+'_'+stat]=row[metric][stat]*scale
        points.append(point)
    for name,rows in [('metrics.csv',points),('requests.csv',requests)]:
        with (output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(output/'validation.json',dict(passed=True,request_rows=len(requests),guard_postcheck=checks[0],evidence=evidence,protocol=load(run/'config.json')))
    ep,pp=points
    summary=f'''# GLM-4.7 A100 matched cached Decode\n\n065/167, 8 A100; both ConnectX-6 Dx 100 Gb/s RoCE, PCIe4 x8.\nB8; P=81920/request; Q=1; 128 output tokens.\nStart: full native CPU HiCache, target GPU prefix empty. H2D and admission included.\nGraph OFF; overlap OFF; Triton Attention; Humming AWQ INT4; DeepEP pair NVLink + RDMA outside pairs, threshold0.\nSynthetic KV; performance only, not generation-quality validation. One cohort/configuration; P99 is nearest-rank over N=8.\n\n|Layout|Output tok/s|TTFT mean s|TPOT mean ms|E2E mean s|\n|---|---:|---:|---:|---:|\n'''
    for x in points:summary+=f"|{x['layout']}|{x['output_tok_s']:.3f}|{x['ttft_s_mean']:.3f}|{x['tpot_ms_mean']:.3f}|{x['end_to_end_s_mean']:.3f}|\n"
    summary+=f"\nPP/EP throughput: {pp['output_tok_s']/ep['output_tok_s']:.4f}x.\n"
    (output/'summary.md').write_text(summary)
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    write(output/'checksums.json',hashes);print(summary)
if __name__=='__main__':main()
