"""One atomic HTTP batch with indexed SSE; CPU-only parser, no engine patches."""
import json
import time
import urllib.request


def make_payload(seqs, query_fn, cohort_base, output_tokens):
    if len(seqs)!=8 or output_tokens<1:
        raise ValueError('This experiment requires exactly eight ordered requests')
    return dict(input_ids=[seq+query_fn(i,1,False) for i,seq in enumerate(seqs)],
        rid=[cohort_base+'-'+str(i) for i in range(8)],
        sampling_params=dict(temperature=0,ignore_eos=True,max_new_tokens=output_tokens),
        stream=True)


def parse_sse(lines,rids,output_tokens,started,*,clock=time.perf_counter):
    """Each row uses the one batch submission timestamp and its own last token."""
    events=[[] for _ in rids];last=[None for _ in rids];done=False
    for line in lines:
        if not line.startswith(b'data:'):continue
        data=line[5:].strip()
        if data==b'[DONE]':done=True;break
        obj=json.loads(data);index=obj.get('index')
        if type(index) is not int or not 0<=index<len(rids):
            raise RuntimeError('Batched stream requires an exact in-range output index')
        meta=obj.get('meta_info',{})
        if meta.get('id')!=rids[index]:
            raise RuntimeError('SSE index and request ID disagree')
        count=meta.get('completion_tokens',0)
        if type(count) is not int or not 0<=count<=output_tokens:
            raise RuntimeError('Invalid cumulative completion count')
        if events[index] and count<events[index][-1][0]:
            raise RuntimeError('Completion count moved backwards')
        now=clock();last[index]=obj
        if count and (not events[index] or count>events[index][-1][0]):
            events[index].append((count,now-started))
    if not done:raise RuntimeError('Batched SSE stream did not terminate with DONE')
    rows=[]
    for index in range(len(rids)):
        ev=events[index];obj=last[index]
        if not ev or ev[-1][0]!=output_tokens:
            raise RuntimeError('Incomplete per-index streamed output')
        if output_tokens>1 and ev[0][0]!=1:
            raise RuntimeError('Buffered first chunk cannot establish true TTFT/TPOT')
        ids=obj.get('output_ids')
        if not isinstance(ids,list) or len(ids)!=output_tokens:
            raise RuntimeError('Complete generated token IDs required')
        rows.append(dict(request_index=index,rid=rids[index],completion_tokens=output_tokens,
            ttft_s=ev[0][1],tpot_s=(ev[-1][1]-ev[0][1])/(output_tokens-1) if output_tokens>1 else None,
            end_to_end_s=ev[-1][1],token_events=ev,meta=obj['meta_info'],
            first_stream_chunk_tokens=ev[0][0],output_ids=ids,output_text=obj.get('text'),
            latency_boundary='Single atomic batch HTTP submission to this request first/last token received'))
    return rows


def generate_batch(url,payload,timeout=1200):
    req=urllib.request.Request(url+'/generate',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    started=time.perf_counter()
    with urllib.request.urlopen(req,timeout=timeout) as response:
        rows=parse_sse(response,payload['rid'],payload['sampling_params']['max_new_tokens'],started)
    return rows,time.perf_counter()-started


def output_parity(left,right):
    def by_id(rows):
        result={row['request_index']:row for row in rows}
        if len(rows)!=8 or set(result)!=set(range(8)):
            raise ValueError('Exact eight request indices required for parity')
        return result
    a,b=by_id(left),by_id(right);rows=[]
    for index in range(8):
        x,y=a[index],b[index]
        tokens=bool(x.get('output_ids')) and x.get('output_ids')==y.get('output_ids')
        text=isinstance(x.get('output_text'),str) and x.get('output_text')==y.get('output_text')
        rows.append(dict(request_index=index,token_ids_equal=tokens,text_equal=text,
            first_output_ids=x.get('output_ids'),second_output_ids=y.get('output_ids'),
            first_text=x.get('output_text'),second_text=y.get('output_text')))
    return dict(passed=all(row['token_ids_equal'] and row['text_equal'] for row in rows),requests=rows,
        criterion='Exact greedy token IDs and text for matched semantic requests')
