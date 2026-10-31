"""Untimed atomic B4/DP or B8/PP probability probes."""
from concurrent.futures import ThreadPoolExecutor


def atomic_validation(base,cfg,arm,key,tokens):
    def request(indices,rank=None):
        payload=dict(input_ids=[tokens[i]+[198] for i in indices],
            sampling_params=dict(temperature=0,max_new_tokens=8,ignore_eos=True),
            stream=False,return_logprob=True,logprob_start_len=cfg['prefix_tokens']+1,
            top_logprobs_num=10)
        if rank is not None:payload['data_parallel_rank']=rank
        value=base.h.http_json(cfg['port'],'/generate',key,payload,timeout=180)
        if not isinstance(value,list) or len(value)!=len(indices):
            raise RuntimeError('Atomic validation response does not cover its request batch')
        rows=[]
        for index,result in zip(indices,value):
            meta=result['meta_info']
            if meta['completion_tokens']!=8 or meta['cached_tokens']!=cfg['prefix_tokens']:
                raise RuntimeError('Atomic validation changed output length or GPU prefix state')
            rows.append((index,dict(text=result.get('text'),output_tokens=8,
                validation_logits={k:meta[k] for k in ('output_token_logprobs','output_top_logprobs')})))
        return rows
    if arm=='ep2':
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(request,list(range(rank,cfg['requests'],2)),rank) for rank in range(2)]
            rows=[row for future in futures for row in future.result()]
    else:rows=request(list(range(cfg['requests'])))
    return dict(requests=[row for _,row in sorted(rows)],performance_sample=False,
                protocol='Atomic B4 request batch per EP Attention-DP replica; atomic B8 split into B4 PP slots')
