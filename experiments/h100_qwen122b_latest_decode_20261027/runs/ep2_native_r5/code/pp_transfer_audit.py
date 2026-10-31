"""Untimed, bounded PP2 payload and microbatch audit; never a speed sample.

The native send/recv and producer-event dependencies are unchanged. Capture
clones on their existing CUDA stream, then hash only after an idle control
request. No tensor contents are written to disk; receipts contain hashes.
"""
from collections import Counter
from functools import wraps
import hashlib
import json
import os
from pathlib import Path

from pp_schedule_opt import open_policy

_INSTALLED = False
_CONTROL = None


def compare_transfers(stages):
    """Exact, ordered comparison in both PP directions, independently per kind."""
    if len(stages) != 2 or {s['pp_rank'] for s in stages} != {0, 1}:
        raise ValueError('Exactly two distinct PP rank receipts are required')
    by_rank = {s['pp_rank']: s for s in stages}
    if by_rank[0]['epoch'] != by_rank[1]['epoch']:
        raise ValueError('Capture epochs differ')
    channels = []
    for source in (0, 1):
        sent = [r for r in by_rank[source]['transfers'] if r['direction'] == 'send']
        received = [r for r in by_rank[1-source]['transfers'] if r['direction'] == 'recv']
        kinds = sorted({r['kind'] for r in sent + received})
        for kind in kinds:
            left = [r for r in sent if r['kind'] == kind]
            right = [r for r in received if r['kind'] == kind]
            def canonical(rows):
                return [{key: row[key] for key in ('sequence', 'tensors')}
                        for row in rows]
            equal = canonical(left) == canonical(right)
            channels.append(dict(source=source, destination=1-source, kind=kind,
                sent=len(left), received=len(right), equal=equal))
    passed = bool(channels) and all(c['sent'] > 0 and c['equal'] for c in channels)
    return dict(passed=passed, channels=channels,
                matched_messages=sum(c['sent'] for c in channels if c['equal']))


def instrument(cls, torch, read_control, root):
    send = cls._pp_send_dict_to_next_stage
    recv = cls._pp_recv_typed_dict
    internal = cls.get_internal_state
    run = cls.run_batch

    def state(self):
        epoch = read_control()
        if epoch == ord('0'):
            return None
        if epoch not in range(ord('1'), ord('9')+1):
            raise RuntimeError('Invalid private capture epoch')
        if self.ps.pp_size != 2 or self.ps.tp_size != 1 or self._pp_topology() is not None:
            raise RuntimeError('Capture is qualified only for static PP2 x TP1')
        current = getattr(self, '_private_transfer_audit', None)
        if current is None:
            current = dict(epoch=chr(epoch), counters=Counter(), tensors=[], batches=[], bytes=0)
            self._private_transfer_audit = current
        if current['epoch'] != chr(epoch):
            raise RuntimeError('A prior epoch was not flushed')
        return current

    def capture(self, direction, kind, tensors):
        current = state(self)
        if current is None:
            return
        key = (direction, kind)
        sequence = current['counters'][key]
        current['counters'][key] += 1
        row = dict(direction=direction, kind=kind, sequence=sequence, tensors={})
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                # Clone at the native stream position, before reusable graph
                # buffers can be overwritten. No CPU fence in this hot path.
                clone = tensor.detach().clone(memory_format=torch.contiguous_format)
                current['bytes'] += clone.numel() * clone.element_size()
                if current['bytes'] > 64 * 1024**2:
                    raise RuntimeError('Capture exceeded its fixed 64 MiB bound')
                row['tensors'][name] = clone
        current['tensors'].append(row)

    @wraps(send)
    def send_wrapped(self, tensor_dict, async_send=True, msg_type='default', dst=None):
        capture(self, 'send', msg_type, tensor_dict)
        return send(self, tensor_dict, async_send=async_send, msg_type=msg_type, dst=dst)

    @wraps(recv)
    def recv_wrapped(self, expected_kind='default', all_gather_group=None, src=None):
        result = recv(self, expected_kind=expected_kind,
                      all_gather_group=all_gather_group, src=src)
        # The typed receive returns stashed messages once, in consumption order.
        capture(self, 'recv', expected_kind, result)
        return result

    @wraps(run)
    def run_wrapped(self, batch, *args, **kwargs):
        current = state(self)
        if current is not None:
            current['batches'].append(dict(mode=str(batch.forward_mode),
                requests=[dict(marker=int(r.origin_input_ids[0]),
                               generated=len(r.output_ids)) for r in batch.reqs]))
        return run(self, batch, *args, **kwargs)

    @wraps(internal)
    def internal_wrapped(self, *args, **kwargs):
        current = getattr(self, '_private_transfer_audit', None)
        if current is not None and read_control() == ord('0'):
            if not self.is_fully_idle():
                # Retry a normal control request after all native slots retire;
                # never force-drain, discard feedback, or fake an empty queue.
                return internal(self, *args, **kwargs)
            torch.cuda.synchronize()
            records = []
            for row in current['tensors']:
                tensors = {}
                for name, tensor in row['tensors'].items():
                    raw = tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                    tensors[name] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                                         sha256=hashlib.sha256(raw).hexdigest())
                records.append(dict(row, tensors=tensors))
            data = dict(pp_rank=self.ps.pp_rank, epoch=current['epoch'],
                        captured_bytes=current['bytes'], transfers=records,
                        batches=current['batches'], performance_sample=False,
                        flushed_when_idle=True)
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = root / f'epoch{current["epoch"]}.pp{self.ps.pp_rank}.json'
            # Never replace an earlier receipt or follow a symlink.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(data, stream, sort_keys=True)
                stream.write('\n')
            self._private_transfer_audit = None
        return internal(self, *args, **kwargs)

    cls._pp_send_dict_to_next_stage = send_wrapped
    cls._pp_recv_typed_dict = recv_wrapped
    cls.get_internal_state = internal_wrapped
    cls.run_batch = run_wrapped


def install():
    global _INSTALLED, _CONTROL
    if _INSTALLED:
        return
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    _CONTROL = open_policy(os.environ['PP_TRANSFER_AUDIT_CONTROL'])
    root = Path(os.environ['EP2_PP2_RUN']) / 'pp2' / 'transfer_audit'
    instrument(Scheduler, torch, lambda: _CONTROL[0], root)
    _INSTALLED = True
