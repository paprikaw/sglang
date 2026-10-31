"""Versioned lossless byte packet for tiny PP Verify control tensors."""
import math
META='__pp_verify_layout_v1__';DATA='__pp_verify_bytes_v1__'
EXPECTED={'next_token_ids','spec_accept_lens','spec_new_seq_lens','spec_bonus_tokens',
          'spec_next_chain','spec_next_parents','spec_next_top_scores'}
SIZES={'torch.int64':8,'torch.int32':4}

def layout(specs):
    if set(specs)!=EXPECTED:raise RuntimeError('Unexpected Verify feedback fields')
    records=[];offset=0
    # Descending element size keeps every view correctly aligned without padding.
    for key,(dtype,shape) in sorted(specs.items(),key=lambda item:(-SIZES.get(item[1][0],0),item[0])):
        if dtype not in SIZES or any(not isinstance(n,int) or n<0 for n in shape):
            raise RuntimeError('Unsupported Verify packet dtype/shape')
        size=SIZES[dtype];count=math.prod(shape);nbytes=count*size
        if offset%size:raise RuntimeError('Unaligned Verify packet')
        records.append((key,dtype,list(shape),offset,nbytes));offset+=nbytes
    if offset>16384:raise RuntimeError('Verify packet exceeds qualified B4 bound')
    return records,offset


def pack(payload):
    import torch
    specs={key:(str(value.dtype),list(value.shape)) for key,value in payload.items()}
    records,total=layout(specs)
    pieces=[payload[key].contiguous().view(torch.uint8).reshape(-1) for key,*_ in records]
    buffer=torch.cat(pieces)
    if buffer.numel()!=total:raise RuntimeError('Verify packet byte count mismatch')
    return {META:records,DATA:buffer}


def unpack(payload):
    if META not in payload:return payload
    import torch
    allowed={META,DATA,'__msg_type__','__pp_generation__'}
    if not set(payload)<=allowed or DATA not in payload:raise RuntimeError('Malformed Verify packet keys')
    records=payload[META];specs={row[0]:(row[1],row[2]) for row in records}
    expected,total=layout(specs)
    if records!=expected or len(records)!=len(specs):raise RuntimeError('Malformed Verify packet layout')
    buffer=payload[DATA]
    if buffer.dtype!=torch.uint8 or buffer.ndim!=1 or buffer.numel()!=total or not buffer.is_contiguous():
        raise RuntimeError('Malformed Verify packet storage')
    dtypes={'torch.int64':torch.int64,'torch.int32':torch.int32}
    result={key:payload[key] for key in ('__msg_type__','__pp_generation__') if key in payload}
    for key,dtype,shape,start,nbytes in records:
        result[key]=buffer[start:start+nbytes].view(dtypes[dtype]).reshape(shape)
    return result


def cpu_qualification():
    """Called once inside admitted GPU workers, before model loading/timing."""
    import torch
    for bs in (1,3,4):
        original={key:torch.arange(bs*(3 if 'chain' in key or key=='next_token_ids' else 2 if key in ('spec_next_parents','spec_next_top_scores') else 1),dtype=torch.int32 if key in ('spec_accept_lens','spec_new_seq_lens') else torch.int64).reshape(bs,-1) for key in EXPECTED}
        packet=pack(original);packet['__msg_type__']='output'
        # Recreate the native TP4 flat split/gather before viewing the fields.
        packet[DATA]=torch.cat(packet[DATA].chunk(4))
        restored=unpack(packet)
        if restored['__msg_type__']!='output' or any(not torch.equal(restored[key],original[key]) or restored[key].dtype!=original[key].dtype for key in EXPECTED):
            raise RuntimeError('CPU lossless Verify codec qualification failed')
    return dict(passed=True,batches=[1,3,4],bit_exact=True,zero_padding=True,one_gpu_tensor=True)
