"""CPU-only routing/shape/ownership regression for the PP message adapter."""
from collections import defaultdict,deque
from dataclasses import dataclass
from types import SimpleNamespace
import json
from pathlib import Path
import torch as cpu
import dynamic_pp_schema as adapter

queues=defaultdict(deque)
class Tensor:
    def __init__(self,shape,dtype,device,value):self.shape=tuple(shape);self.dtype=dtype;self.device=device;self.value=value;self.is_cuda=True
    def is_contiguous(self):return True
    def clone(self):return Tensor(self.shape,self.dtype,self.device,self.value)
class Work:
    def __init__(self,fn=lambda:None):self.fn=fn
    def wait(self):return self.fn()
    def is_completed(self):return True
class Dist:
    count=0
    def new_group(self,**kw):
        group='header'+str(self.count%3);self.count+=1;return group
    def barrier(self,**kw):pass
    def isend(self,tensor,peer,group):queues[group].append(tensor.clone());return Work()
    def irecv(self,tensor,src,group):
        def receive():
            source=queues[group].popleft()
            if isinstance(tensor,Tensor):
                assert tensor.shape==source.shape and tensor.dtype==source.dtype
                tensor.value=source.value
            else:tensor.copy_(source)
        return Work(receive)
dist=Dist()
torch=SimpleNamespace(distributed=dist,uint8=cpu.uint8,bfloat16=cpu.bfloat16,int64=cpu.int64,
    frombuffer=cpu.frombuffer,is_tensor=lambda x:isinstance(x,Tensor) or cpu.is_tensor(x),
    device=lambda kind,index:(kind,index),cuda=SimpleNamespace(current_device=lambda:1))
def empty(shape,dtype,device):return Tensor(shape,dtype,device,None) if device!='cpu' else cpu.empty(shape,dtype=dtype,device='cpu')
torch.empty=empty
@dataclass
class P2PWork:work:object;payload:object
class Pipeline:
    def __init__(self,last):
        self.pp_group=SimpleNamespace(ranks=[0,1],rank_in_group=int(last),is_last_rank=last)
        self._gap_groups=['data0','data1','data2'];self._pp_tensor_dict_inbox={}
    def init_pp_loop_state(self):pass
    def _pp_send_dict_to_next_stage(self,*a,**kw):return 'native-send'
    def _pp_recv_typed_dict(self,*a,**kw):return 'native-receive'
mode=[9]
adapter.install_schema(Pipeline,torch,lambda:mode[0],P2PWork)
head,tail=Pipeline(False),Pipeline(True)
head.init_pp_loop_state();tail.init_pp_loop_state()
checks=0
for sender,receiver,kind in [(head,tail,'proxy'),(tail,head,'output'),(head,tail,'output')]:
    for n in [1,2,4,8,16,512]:
        payload={'hidden':Tensor((n,2048),cpu.bfloat16,('cuda',0),n),'nullable':None}
        works=sender._pp_send_dict_to_next_stage(payload,msg_type=kind)
        assert len(works)==1 and sender._gap_header_pending[-1][1].device.type=='cpu'
        assert all(isinstance(w.payload,Tensor) for w in works)
        result=receiver._pp_recv_typed_dict(expected_kind=kind)
        assert result['hidden'].shape==(n,2048) and result['hidden'].value==n
        assert result['hidden'].device==('cuda',1) and result['nullable'] is None and result['__msg_type__']==kind
        checks+=1
assert all(not q for q in queues.values())
mode[0]=0
assert head._pp_send_dict_to_next_stage({})=='native-send'
assert tail._pp_recv_typed_dict()=='native-receive'
try:adapter.decode_schema(bytes(5));raise AssertionError('Invalid header accepted')
except RuntimeError:pass
try:adapter.encode_schema([['x',[1],'int64']],{'huge':'x'*20000});raise AssertionError('Oversized header accepted')
except RuntimeError:pass
result=dict(passed=True,shape_channel_ownership_cases=checks,native_fallback=True,invalid_headers_rejected=True,cpu_only=True)
Path(__file__).with_name('dynamic_schema_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
