"""Private PP2 scheduling candidates, switched only at a drained cohort fence.

0: shared GEMM baseline; 1: original activation send submitted early;
2: additionally retire the tail's own result, eliminating its head echo;
3: additionally reuse warmed message schemas for fixed B4/B8/B16 tensor messages;
5: early activation and cached schemas on separate typed NCCL channels;
6: additionally submit activation on a dedicated producer-dependent stream,
retaining the original output ring; 7: prepare next Decode metadata early;
8: additionally launch the next slot before CPU retirement;
9: widen the qualified lookahead window while retaining boundary guards.
Policy4 is rejected: a shared channel
cannot safely infer message type when asynchronous PP changes arrival order.
Payloads retain the native producer-event and P2PWork ownership dependencies.
Policies 5-9 use distinct NCCL groups for activation and the two token relays.
"""
from dataclasses import dataclass
from functools import wraps
import json
import os
from pathlib import Path

from pp_schedule_opt import open_policy

_INSTALLED = False
_POLICY = None


def mode():
    value = _POLICY[0] - ord("0")
    if value not in (0, 1, 2, 3, 5, 6, 7, 8, 9):
        raise RuntimeError("Unknown PP gap policy")
    return value


@dataclass
class Pending:
    event: object
    payload: object
    rids: tuple


class Slots:
    def __init__(self):
        self.pending = {}
        self.produced = self.retired = 0

    def put(self, slot, event, payload, rids):
        if slot not in (0, 1) or slot in self.pending:
            raise RuntimeError("Local feedback slot overwrite")
        self.pending[slot] = Pending(event, payload, tuple(rids))
        self.produced += 1

    def pop(self, slot, rids):
        pending = self.pending.pop(slot)
        if pending.rids != tuple(rids):
            raise RuntimeError("Local feedback request ancestry mismatch")
        self.retired += 1
        return pending


def install_early(cls, read_mode):
    launch = cls._pp_launch_or_skip_batch
    send = cls._pp_send_launched_proxy

    def launch_wrapped(self, slot, batch, proxy, metadata, queue):
        result, event = launch(self, slot, batch, proxy, metadata, queue)
        if read_mode() >= 1 and batch is not None and not self._pp_is_last_rank():
            if self.ps.pp_size != 2 or self._pp_topology() is not None:
                raise RuntimeError("Early activation requires static PP2")
            # The caller normally assigns this event after launch returns.
            # Publish exactly that event before submitting the same native send.
            self.launch_event = event
            if read_mode() >= 6:
                with self._gap_proxy_stream_ctx:
                    send(self, batch, result)
            else:
                send(self, batch, result)
            if not hasattr(self,"_gap_proxy_sent"):
                self._gap_proxy_sent=[]
            self._gap_proxy_sent.append((batch,result))
        return result, event

    def send_wrapped(self, batch, result):
        pending=getattr(self,"_gap_proxy_sent",[])
        for index,(b,r) in enumerate(pending):
            if b is batch and r is result:
                del pending[index]
                return None
        return send(self, batch, result)

    cls._pp_launch_or_skip_batch = launch_wrapped
    cls._pp_send_launched_proxy = send_wrapped


def install_local(cls, read_mode, can_skip):
    init = cls.init_pp_loop_state
    send = cls._pp_send_output_to_next_stage
    receive = cls._pp_recv_dict_from_prev_stage
    exchange = cls._pp_send_recv_and_preprocess_output_tensors

    def init_wrapped(self, *args, **kwargs):
        result = init(self, *args, **kwargs)
        if self.ps.pp_size != 2 or self._pp_topology() is not None:
            raise RuntimeError("Candidate requires static PP2")
        self._gap_slots = Slots()
        self._gap_retiring = None
        self._gap_echo_skipped = 0
        self._gap_prepared = self._gap_prepared_consumed = 0
        self._gap_ahead={}
        self._gap_ahead_produced=self._gap_ahead_consumed=0
        self._gap_groups = []
        if os.environ.get("PP_GAP_SEPARATE_CHANNELS") == "1":
            import torch
            if self.pp_group.is_first_rank:
                self._gap_proxy_stream = torch.cuda.Stream()
                self._gap_proxy_stream_ctx = torch.cuda.stream(self._gap_proxy_stream)
            for _ in range(3):
                group = torch.distributed.new_group(ranks=self.pp_group.ranks, backend="nccl")
                # Initialize all channels collectively before the scheduler
                # starts. Lazy initialization at a differently ordered P2P
                # receive would reintroduce a cross-channel wait cycle.
                torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
                self._gap_groups.append(group)
        return result

    def send_wrapped(self, first_id, batches, queue, previous):
        if read_mode() not in (2, 3):
            return send(self, first_id, batches, queue, previous)
        if self.pp_loop_size != 2:
            raise RuntimeError("Local feedback requires exactly two slots")
        if not self.pp_group.is_last_rank:
            if previous is not None:
                self._gap_echo_skipped += 1
            return []
        target = batches[first_id]
        if target is None:
            return []
        event, payload = queue.popleft()
        if target.forward_mode.is_prebuilt() or can_skip(target):
            return []
        self._gap_slots.put(first_id, event, payload, [r.rid for r in target.reqs])
        self.device_module.current_stream().wait_event(event)
        return self._pp_send_dict_to_next_stage(payload.tensors, async_send=True, msg_type="output")

    def receive_wrapped(self):
        if read_mode() not in (2, 3) or not self.pp_group.is_last_rank:
            return receive(self)
        if self._gap_retiring is None:
            raise RuntimeError("Local feedback outside retirement")
        slot, rids = self._gap_retiring
        pending = self._gap_slots.pop(slot, rids)
        self.device_module.current_stream().wait_event(pending.event)
        return pending.payload.tensors

    def exchange_wrapped(self, first_id, next_id, batches, metadata, queue, previous):
        if read_mode() not in (2, 3):
            return exchange(self, first_id, next_id, batches, metadata, queue, previous)
        if self._pp_requires_immediate_output_relay():
            raise RuntimeError("Local feedback does not cover elastic PP")
        target = batches[next_id]
        self._gap_retiring = (next_id, tuple(r.rid for r in target.reqs)) if target is not None else None
        try:
            return exchange(self, first_id, next_id, batches, metadata, queue, previous)
        finally:
            self._gap_retiring = None

    cls.init_pp_loop_state = init_wrapped
    cls._pp_send_output_to_next_stage = send_wrapped
    cls._pp_recv_dict_from_prev_stage = receive_wrapped
    cls._pp_send_recv_and_preprocess_output_tensors = exchange_wrapped


def can_prepare(batch, interval, min_outputs=8, finish_margin=8):
    if batch is None or not batch.forward_mode.is_decode() or len(batch.reqs) not in (2, 4, 8, 16):
        return False
    if not batch.spec_algorithm.is_none() or batch.return_logprob or batch.has_grammar:
        return False
    if batch.sampling_info.penalizer_orchestrator.is_required:
        return False
    if getattr(batch,"_gap_next_prepared",False) or not batch.check_decode_mem():
        return False
    # Avoid changing GDN ping-pong ownership while a boundary is retiring.
    if any(int(n) % interval in (0,interval-1) for n in batch.seq_lens_cpu.tolist()):
        return False
    for req in batch.reqs:
        params=req.sampling_params
        if not params.ignore_eos or not (min_outputs <= len(req.output_ids) < params.max_new_tokens-finish_margin):
            return False
        if req.finished() or req.is_retracted or getattr(req,"to_finish",None) is not None or req.grammar is not None:
            return False
        if getattr(params,"stop_strs",None) or getattr(params,"stop_token_ids",None):
            return False
    return True


def install_prepare(cls, batch_cls, read_mode, interval_of):
    exchange=cls._pp_send_recv_and_preprocess_output_tensors
    prepare=batch_cls.prepare_for_decode

    def prepare_wrapped(batch):
        owner=getattr(batch,"_gap_next_prepared",None)
        if owner is not None and owner is not False:
            batch._gap_next_prepared=False
            owner._gap_prepared_consumed+=1
            return None
        return prepare(batch)

    def exchange_wrapped(self,first,next_id,batches,metadata,queue,previous):
        result=exchange(self,first,next_id,batches,metadata,queue,previous)
        if read_mode() not in (7,8,9):
            return result
        target=batches[next_id]
        if target is not self.running_mbs[next_id]:
            return result
        window=(2,2) if read_mode()==9 else (8,8)
        if not can_prepare(target,interval_of(self),*window):
            return result
        # The previous actual GPU token has already been stashed in future_map.
        # Keep the original result-processing snapshot; allocate only predictable
        # next-step KV/sequence metadata before the caller waits for D2H.
        snapshot=target.copy()
        prepare(target)
        if snapshot.mamba_track_mask_cpu is not None:
            snapshot.mamba_track_mask_next_cpu=target.mamba_track_mask_cpu[:]
        target._gap_next_prepared=self
        batches[next_id]=snapshot
        self._gap_prepared+=1
        if read_mode()>=8:
            self._gap_launch_ahead(next_id,target,metadata,queue)
        return result

    batch_cls.prepare_for_decode=prepare_wrapped
    cls._pp_send_recv_and_preprocess_output_tensors=exchange_wrapped


@dataclass
class Ahead:
    batch:object
    proxy:object
    result:object
    event:object
    rids:tuple
    received:bool=False


def install_lookahead(cls,read_mode):
    receive=cls._pp_recv_proxy_tensors
    launch=cls._pp_launch_or_skip_batch

    def launch_ahead(self,slot,batch,metadata,queue):
        if slot in self._gap_ahead:
            raise RuntimeError("PP lookahead slot already occupied")
        old_running,old_debug=self.running_batch,self.cur_batch_for_debug
        try:
            self.running_batch=self.cur_batch_for_debug=batch
            # Retain the native write-after-send dependency before reusing a
            # CUDA Graph's activation buffers. Fast typed channels contain only
            # GPU work, so this inserts a stream dependency without CPU waits.
            self._pp_commit_comm_work(self.send_proxy_work)
            proxy=receive(self)
            result,event=launch(self,slot,batch,proxy,metadata,queue)
            self._gap_ahead[slot]=Ahead(batch,proxy,result,event,tuple(r.rid for r in batch.reqs))
            self._gap_ahead_produced+=1
        finally:
            self.running_batch,self.cur_batch_for_debug=old_running,old_debug

    def receive_wrapped(self):
        for pending in self._gap_ahead.values():
            if pending.batch is self.cur_batch_for_debug:
                if read_mode() not in (8,9) or pending.received:
                    raise RuntimeError("PP lookahead proxy consumed twice")
                pending.received=True
                return pending.proxy
        return receive(self)

    def launch_wrapped(self,slot,batch,proxy,metadata,queue):
        pending=self._gap_ahead.pop(slot,None)
        if pending is None:
            return launch(self,slot,batch,proxy,metadata,queue)
        if (read_mode() not in (8,9) or pending.batch is not batch or pending.proxy is not proxy
                or not pending.received or pending.rids!=tuple(r.rid for r in batch.reqs)):
            raise RuntimeError("PP lookahead launch ancestry mismatch")
        self._gap_ahead_consumed+=1
        return pending.result,pending.event

    cls._gap_launch_ahead=launch_ahead
    cls._pp_recv_proxy_tensors=receive_wrapped
    cls._pp_launch_or_skip_batch=launch_wrapped


def describe(torch, payload):
    tensors, values = [], []
    for key, value in payload.items():
        if torch.is_tensor(value):
            if not value.is_cuda or not value.is_contiguous():
                return None
            if not value.shape or value.shape[0] not in (2, 4, 8, 16):
                return None
            tensors.append((key, tuple(value.shape), value.dtype, value.device))
        else:
            values.append((key, value))
    return (tuple(tensors), tuple(values)) if tensors else None


def install_schema(cls, torch, read_mode, P2PWork):
    original_send = cls._pp_send_dict_to_next_stage
    original_receive = cls._pp_recv_typed_dict

    def cache(self):
        if not hasattr(self, "_gap_schemas"):
            self._gap_schemas = {}
            self._gap_fast_sent = self._gap_fast_received = 0
        return self._gap_schemas

    def send(self, payload, async_send=True, msg_type="default", dst=None):
        remembered = cache(self)
        if read_mode() not in (3, 5, 6, 7, 8, 9):
            result = original_send(self, payload, async_send=async_send, msg_type=msg_type, dst=dst)
            schema = describe(torch, payload)
            if schema is not None:
                remembered["send", msg_type] = schema
            return result
        expected = ("output",) if self.pp_group.is_last_rank else (("proxy", "output") if read_mode() >= 5 else ("proxy",))
        if msg_type not in expected or dst is not None or not async_send:
            raise RuntimeError("Fast path requires one-way static PP2 messages")
        payload["__msg_type__"] = msg_type
        schema = describe(torch, payload)
        if schema is None or schema != remembered.get(("send", msg_type)):
            raise RuntimeError("Fast PP send schema differs from warmed native message")
        group = self.pp_group
        peer = group.ranks[(group.rank_in_group + 1) % 2]
        channel = group.device_group
        if read_mode() >= 5:
            index = 0 if msg_type == "proxy" else (1 if group.is_last_rank else 2)
            channel = self._gap_groups[index]
        work = []
        for key, shape, dtype, device in schema[0]:
            tensor = payload[key]
            # This is the native payload operation, including payload retention.
            handle = torch.distributed.isend(tensor, peer, group=channel)
            work.append(P2PWork(handle, tensor))
        self._gap_fast_sent += 1
        return work

    def receive(self, expected_kind="default", all_gather_group=None, src=None):
        remembered = cache(self)
        if read_mode() not in (3, 5, 6, 7, 8, 9):
            result = original_receive(self, expected_kind=expected_kind, all_gather_group=all_gather_group, src=src)
            schema = describe(torch, result)
            if schema is not None:
                remembered["recv", expected_kind] = schema
            return result
        expected = (("proxy", "output") if read_mode() >= 5 else ("proxy",)) if self.pp_group.is_last_rank else ("output",)
        if expected_kind not in expected or src is not None or getattr(all_gather_group, "world_size", 1) != 1:
            raise RuntimeError("Fast PP receive requires static PP2 x TP1")
        schema = remembered.get(("recv", expected_kind))
        if schema is None:
            raise RuntimeError("Fast PP receive has no warmed native schema")
        if any(self._pp_tensor_dict_inbox.values()):
            raise RuntimeError("Fast PP receive cannot bypass a queued native message")
        group = self.pp_group
        peer = group.ranks[(group.rank_in_group - 1) % 2]
        channel = group.device_group
        if read_mode() >= 5:
            index = 0 if expected_kind == "proxy" else (2 if group.is_last_rank else 1)
            channel = self._gap_groups[index]
        result = {}
        for key, shape, dtype, device in schema[0]:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            handle = torch.distributed.irecv(tensor, src=peer, group=channel)
            handle.wait()  # Same stream dependency as native recv_tensor_dict.
            result[key] = tensor
        result.update(schema[1])
        self._gap_fast_received += 1
        return result

    cls._pp_send_dict_to_next_stage = send
    cls._pp_recv_typed_dict = receive


def install():
    global _INSTALLED, _POLICY
    if _INSTALLED or os.environ["EP2_PP2_ARM"] != "pp2":
        return
    import torch
    from sglang.srt.distributed.parallel_state import P2PWork
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.runtime_context import mamba_track_grid
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin, _pp_can_skip_output_comm
    _POLICY = open_policy(os.environ["PP_GAP_POLICY_PATH"])
    install_early(SchedulerPPMixin, mode)
    install_local(SchedulerPPMixin, mode, _pp_can_skip_output_comm)
    install_schema(SchedulerPPMixin, torch, mode, P2PWork)
    install_lookahead(SchedulerPPMixin,mode)
    install_prepare(SchedulerPPMixin, ScheduleBatch, mode,
                    lambda s:mamba_track_grid(s.tree_cache.page_size))
    internal = Scheduler.get_internal_state

    @wraps(internal)
    def state(self, *args, **kwargs):
        result = internal(self, *args, **kwargs)
        slots = getattr(self, "_gap_slots", None)
        result.internal_state["experiment_gap_state"] = dict(
            policy=mode(), produced=slots.produced if slots else 0,
            retired=slots.retired if slots else 0,
            pending=list(slots.pending) if slots else [],
            echoes_skipped=getattr(self, "_gap_echo_skipped", 0),
            fast_sent=getattr(self, "_gap_fast_sent", 0),
            fast_received=getattr(self, "_gap_fast_received", 0),
            typed_channels=len(getattr(self,"_gap_groups",[])),
            dedicated_proxy_stream=hasattr(self,"_gap_proxy_stream"),
            actual_ring_slots=self.pp_loop_size)
        result.internal_state["experiment_gap_state"].update(
            prepared_next=self._gap_prepared,prepared_consumed=self._gap_prepared_consumed,
            ahead_pending=list(self._gap_ahead),ahead_produced=self._gap_ahead_produced,
            ahead_consumed=self._gap_ahead_consumed)
        directory = Path(os.environ["EP2_PP2_RUN"]) / "pp2" / "gap_receipts"
        directory.mkdir(exist_ok=True)
        # Control requests run outside measured cohorts. Keep a bounded latest
        # receipt per policy/rank, never add telemetry to the per-token path.
        (directory / f"policy{mode()}.pp{self.ps.pp_rank}.json").write_text(
            json.dumps(result.internal_state["experiment_gap_state"], indent=2) + "\n")
        return result
    Scheduler.get_internal_state = state
    _INSTALLED = True
