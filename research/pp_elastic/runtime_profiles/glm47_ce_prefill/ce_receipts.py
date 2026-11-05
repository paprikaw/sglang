"""Post-formal CE counter requests via owned local Unix sockets; no signals."""
import json
import math
from pathlib import Path
import time
import uuid
from common import save, sha, visible_json
from counter_socket import exchange


def fetch_owned_ranks(output,node_rank,owned,request,record_lookup,socket_exchange=exchange):
    """Validate all four owners, then fetch only matching post-formal counters."""
    identities=[]
    for rank in range(node_rank*4,node_rank*4+4):
        identity=json.loads((Path(output)/f'ce_config_rank{rank}.json').read_text())
        pid,ticks=identity['pid'],identity['process_start_ticks'];record=record_lookup(pid)
        if (identity['rank']!=rank or owned.get(pid)!=ticks or record is None
                or record['ticks']!=ticks or record.get('state')=='Z' or not identity.get('counter_socket')):
            raise RuntimeError('CE counter owner identity changed')
        identities.append(identity)
    responses=[]
    for identity in identities:
        response=socket_exchange(identity,request)
        receipt=response.get('receipt',{})
        if (not receipt.get('passed') or receipt.get('request_id')!=request['request_id']
                or any(receipt.get(key)!=identity[key] for key in ('rank','pid','process_start_ticks'))
                or response.get('receipt_sha256')!=sha(Path(output)/f"ce_final_rank{identity['rank']}.json")):
            raise RuntimeError('CE socket response identity/nonce/hash mismatch')
        if receipt!=validate_final(output,identity['rank'],request):
            raise RuntimeError('CE socket response differs from its saved receipt')
        responses.append(dict(rank=identity['rank'],pid=identity['pid'],process_start_ticks=identity['process_start_ticks'],
            request_id=request['request_id'],counter_socket=identity['counter_socket'],receipt_sha256=response['receipt_sha256']))
    return dict(passed=True,node_rank=node_rank,request_id=request['request_id'],completed_at=time.time(),
                responses=responses,transport='unix_socket',cpu_only=True,timed=False)


def validate_final(output,rank,request):
    output=Path(output);identity=json.loads((output/f'ce_config_rank{rank}.json').read_text())
    row=json.loads((output/f'ce_final_rank{rank}.json').read_text())
    if (not row.get('passed') or not row.get('after_formal_requests') or not row.get('cpu_only')
            or row.get('timed') is not False
            or row.get('source')!='unix_socket_listener' or row.get('request_id')!=request['request_id']
            or any(row.get(key)!=identity[key] for key in ('rank','pid','process_start_ticks'))):
        raise RuntimeError('CE final counter receipt identity/request mismatch')
    if (any(not isinstance(row.get(key),(int,float)) or not math.isfinite(row[key]) or row[key]<=0
            for key in ('received_at','completed_at')) or row['completed_at']<row['received_at']):
        raise RuntimeError('CE final counter receipt local timestamps are invalid')
    return row


def collect(run,arm,selected,read_visible=visible_json):
    """All model requests have finished; fetch from two supervisors before STOP."""
    run=Path(run);output=run/arm;receipts=run/'artifact_visibility/ce_counters'
    request=dict(passed=True,arm=arm,request_id=uuid.uuid4().hex,created_at=time.time(),
                 after_formal_requests=True,no_other_model_requests=True,timed=False)
    save(output/'CE_SNAPSHOT_REQUEST.json',request)
    nodes=[]
    for node_rank in range(len(selected['nodes'])):
        row=read_visible(output/f'ce_socket_node{node_rank}.json',receipts,60)
        if (not row.get('passed') or row.get('transport')!='unix_socket' or row.get('request_id')!=request['request_id']
                or not row.get('cpu_only') or row.get('timed') is not False or row.get('node_rank')!=node_rank
                or [ack.get('rank') for ack in row.get('responses',[])]!=list(range(node_rank*4,node_rank*4+4))):
            raise RuntimeError('CE socket counters were not received for this request')
        nodes.append(row)
    finals=[]
    for rank in range(selected['world']):
        read_visible(output/f'ce_final_rank{rank}.json',receipts,30)
        final=validate_final(output,rank,request);node=nodes[rank//4];ack=node['responses'][rank%4]
        identity=json.loads((output/f'ce_config_rank{rank}.json').read_text())
        if (any(ack.get(key)!=identity[key] for key in ('rank','pid','process_start_ticks','counter_socket'))
                or ack.get('request_id')!=request['request_id']
                or ack.get('receipt_sha256')!=sha(output/f'ce_final_rank{rank}.json')):
            raise RuntimeError('CE node acknowledgement identity/hash mismatch')
        finals.append(final)
    save(output/'ce_counter_receipts.json',dict(passed=True,request=request,node_socket_receipts=nodes,
        ranks=[row['rank'] for row in finals],all_workers_received=True,transport='unix_socket',cpu_only=True,timed=False))
    return finals
