"""Dynamic CE setup and a separately labelled contiguous-transfer calibration."""
import ctypes
from common import HIDDEN,EXPERTS,CAPACITY,save
from pathlib import Path

def geometry(world,capacity,experts=EXPERTS,hidden=HIDDEN):
    assert world in (3,4,8) and experts%world==0
    assert capacity>0 and capacity<=2048 and 0<experts//world<=256
    header=16+((experts//world*4+15)//16)*16
    frame=header+capacity*(experts//world)*(hidden*2+32)
    extra=((world*frame+2*world*128+127)//128)*128
    return dict(header_bytes=header,frame_bytes=frame,extra_bytes=extra,local_experts=experts//world)

def setup(torch,native,view,ptrs,rdma_bytes,rank,world,capacity,dedup,matched=False,skip_drain=False,local_notify=False,experts=EXPERTS,hidden=HIDDEN):
    g=geometry(world,capacity,experts,hidden);frame=g['frame_bytes'];extra=g['extra_bytes']
    view[rdma_bytes:rdma_bytes+extra].zero_()
    tensors=[torch.empty(world*frame,device='cuda',dtype=torch.uint8),
        torch.empty(world,device='cuda',dtype=torch.int32),
        torch.empty(experts,device='cuda',dtype=torch.int32),
        torch.empty(world,dtype=torch.int32,pin_memory=True)]
    api=ctypes.CDLL(str(native))
    api.deepep_ce_configure_v2.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    api.deepep_ce_configure_v2.restype=ctypes.c_int
    api.deepep_ce_snapshot.argtypes=[ctypes.POINTER(ctypes.c_uint64)];api.deepep_ce_snapshot.restype=ctypes.c_int
    api.deepep_ce_set_dispatch_dedup.argtypes=[ctypes.c_int];api.deepep_ce_set_dispatch_dedup.restype=ctypes.c_int
    peers=(ctypes.c_uint64*world)(*[x+rdma_bytes if x else 0 for x in ptrs])
    torch.cuda.synchronize()
    assert api.deepep_ce_configure_v2(tensors[0].data_ptr(),view.data_ptr()+rdma_bytes,peers,
        tensors[1].data_ptr(),tensors[2].data_ptr(),tensors[3].data_ptr(),capacity*g['local_experts'],rank,world,capacity)==0
    assert api.deepep_ce_set_dispatch_dedup(int(dedup))==0
    api.deepep_ce_set_matched_peer_order.argtypes=[ctypes.c_int];api.deepep_ce_set_matched_peer_order.restype=ctypes.c_int
    assert api.deepep_ce_set_matched_peer_order(int(matched))==0
    api.deepep_ce_set_skip_local_drain.argtypes=[ctypes.c_int];api.deepep_ce_set_skip_local_drain.restype=ctypes.c_int
    assert api.deepep_ce_set_skip_local_drain(int(skip_drain))==0
    api.deepep_ce_set_local_notify.argtypes=[ctypes.c_int];api.deepep_ce_set_local_notify.restype=ctypes.c_int
    assert api.deepep_ce_set_local_notify(int(local_notify))==0
    return api,tensors

def snapshot(api):
    result=(ctypes.c_uint64*6)();assert api.deepep_ce_snapshot(result)==result[0]
    return list(result)

def validate_metrics(delta):
    # EP4 and EP8 each have two mapped cross-pair peers on the source node.
    # Dispatch/combine each issue two CE memcpys; remote traffic stays RDMA.
    assert len(delta)==6 and delta[0]==2 and delta[1]==4, delta
    assert delta[2]>0 and delta[3]>0 and delta[4:]==[1,1], delta
    return True
