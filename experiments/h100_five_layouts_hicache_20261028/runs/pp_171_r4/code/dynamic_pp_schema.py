"""Shape-announced typed PP2 channels; asynchronous CPU header ownership.

The fixed-schema optimization cannot process request admission/drain or HiCache
loadback batches. Send each actual shape on a distinct Gloo channel, retaining
the header until its native P2PWork is retired. CUDA payloads use the existing
three NCCL channels and native producer dependencies. No inferred batch shape.
"""
import json,struct
HEADER_BYTES=16384

def encode_schema(tensors,values):
    raw=json.dumps(dict(tensors=tensors,values=values),separators=(',',':')).encode()
    if len(raw)>HEADER_BYTES-4:raise RuntimeError('Typed PP header exceeds fixed buffer')
    return bytearray(struct.pack('<I',len(raw))+raw+bytes(HEADER_BYTES-4-len(raw)))

def decode_schema(raw):
    if len(raw)!=HEADER_BYTES:raise RuntimeError('Typed PP header length mismatch')
    length=struct.unpack('<I',raw[:4])[0]
    if length>HEADER_BYTES-4:raise RuntimeError('Typed PP header payload length invalid')
    result=json.loads(raw[4:4+length])
    if not result['tensors']:raise RuntimeError('Typed PP message has no tensor')
    return result

def install_schema(cls,torch,read_mode,P2PWork):
    original_send=cls._pp_send_dict_to_next_stage
    original_receive=cls._pp_recv_typed_dict
    original_init=cls.init_pp_loop_state

    def init(self,*args,**kwargs):
        result=original_init(self,*args,**kwargs)
        self._gap_header_groups=[]
        for _ in range(3):
            group=torch.distributed.new_group(ranks=self.pp_group.ranks,backend='gloo')
            torch.distributed.barrier(group=group)
            self._gap_header_groups.append(group)
        self._gap_fast_sent=self._gap_fast_received=0
        return result

    def channel_index(self,kind,sending):
        if kind=='proxy':return 0
        if kind!='output':raise RuntimeError('Unexpected typed PP message kind')
        return (1 if self.pp_group.is_last_rank else 2) if sending else (2 if self.pp_group.is_last_rank else 1)

    def send(self,payload,async_send=True,msg_type='default',dst=None):
        if read_mode() not in (5,6,7,8,9,10):
            return original_send(self,payload,async_send=async_send,msg_type=msg_type,dst=dst)
        if dst is not None or not async_send:raise RuntimeError('Typed PP2 requires native one-way async messages')
        payload['__msg_type__']=msg_type
        tensors=[];values={}
        for key,value in payload.items():
            if torch.is_tensor(value):
                if not value.is_cuda or not value.is_contiguous():raise RuntimeError('Typed PP payload must retain contiguous native CUDA tensor')
                tensors.append([key,list(value.shape),str(value.dtype).removeprefix('torch.')])
            else:values[key]=value
        raw=encode_schema(tensors,values)
        header=torch.frombuffer(raw,dtype=torch.uint8)
        index=channel_index(self,msg_type,True)
        peer=self.pp_group.ranks[1-self.pp_group.rank_in_group]
        work=[P2PWork(torch.distributed.isend(header,peer,group=self._gap_header_groups[index]),header)]
        for key,shape,dtype in tensors:
            tensor=payload[key]
            work.append(P2PWork(torch.distributed.isend(tensor,peer,group=self._gap_groups[index]),tensor))
        self._gap_fast_sent+=1
        return work

    def receive(self,expected_kind='default',all_gather_group=None,src=None):
        if read_mode() not in (5,6,7,8,9,10):
            return original_receive(self,expected_kind=expected_kind,all_gather_group=all_gather_group,src=src)
        if src is not None or getattr(all_gather_group,'world_size',1)!=1:raise RuntimeError('Typed PP receive requires static PP2 x Attention TP1')
        if any(self._pp_tensor_dict_inbox.values()):raise RuntimeError('Typed PP cannot bypass queued native messages')
        index=channel_index(self,expected_kind,False)
        peer=self.pp_group.ranks[1-self.pp_group.rank_in_group]
        header=torch.empty(HEADER_BYTES,dtype=torch.uint8,device='cpu')
        torch.distributed.irecv(header,src=peer,group=self._gap_header_groups[index]).wait()
        schema=decode_schema(bytes(header.numpy()))
        if schema['values'].get('__msg_type__')!=expected_kind:raise RuntimeError('Typed PP message kind differs from expected channel')
        result={}
        for key,shape,dtype in schema['tensors']:
            tensor=torch.empty(shape,dtype=getattr(torch,dtype),device=torch.device('cuda',torch.cuda.current_device()))
            torch.distributed.irecv(tensor,src=peer,group=self._gap_groups[index]).wait()
            result[key]=tensor
        result.update(schema['values'])
        self._gap_fast_received+=1
        return result

    cls.init_pp_loop_state=init
    cls._pp_send_dict_to_next_stage=send
    cls._pp_recv_typed_dict=receive
