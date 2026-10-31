"""CUDA-only Decode activity with two tiny GPU markers per native forward.

No inner Attention/Expert/communication wrappers or CPU operator recording.
Preallocated whole-forward events independently check marker durations. Stop,
readout and export happen after client completion. CPU-importable.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time

_PROBE = None


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def validate_marker(value, run, arm, attribution, finish=False):
    if (value.get('schema_version') != 1 or value.get('run_id') != Path(run).name
            or value.get('arm') != arm or value.get('token') != attribution['run_token']):
        raise ValueError('Attribution marker identity/token mismatch')
    if finish:
        if value.get('finished') is not True:
            raise ValueError('Finish marker requires finished=true')
    elif (value.get('enabled') is not True or value.get('detailed_forward_ordinals')
            != attribution['detailed_ordinals']):
        raise ValueError('Attribution marker detail protocol mismatch')
    return value


def cpu_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if getattr(getattr(value, 'device', None), 'type', None) == 'cpu':
        return value.tolist()
    return None


def metadata(batch):
    data = {key: cpu_list(getattr(batch, key, None)) for key in
            ('seq_lens_cpu', 'global_num_tokens_cpu', 'original_global_num_tokens_cpu',
             'extend_prefix_lens_cpu', 'extend_seq_lens_cpu')}
    data.update(batch_size=getattr(batch, 'batch_size', None),
                extend_num_tokens=getattr(batch, 'extend_num_tokens', None),
                input_ids_shape=list(getattr(getattr(batch, 'input_ids', None), 'shape', ())),
                forward_mode=str(getattr(batch, 'forward_mode', 'unknown')))
    return data


def shapes(args, kwargs):
    data = []
    def visit(name, value, depth=0):
        if len(data) >= 12 or depth > 2:
            return
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            data.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                             device=str(value.device)))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value[:8]):
                visit(f'{name}.{i}', item, depth + 1)
        elif isinstance(value, dict):
            for key, item in list(value.items())[:8]:
                visit(f'{name}.{key}', item, depth + 1)
    for i, value in enumerate(args):
        visit(f'arg{i}', value)
    for key, value in kwargs.items():
        visit(str(key), value)
    return data


def union(intervals):
    total, end = 0., None
    for low, high in sorted(intervals):
        if high < low:
            raise ValueError('Backwards interval')
        if end is None or low > end:
            total += high - low
            end = high
        elif high > end:
            total += high - end
            end = high
    return total


def summarize(events):
    """Per-stream exclusive child union. Never add across streams/ranks."""
    result = {}
    for row in events:
        if row['start_stream'] != row['end_stream']:
            continue
        children = [c for c in events if c['parent'] == row['index']
                    and c['start_stream'] == c['end_stream'] == row['start_stream']]
        covered = union([(max(row['start_ms'], c['start_ms']), min(row['end_ms'], c['end_ms']))
                         for c in children if min(row['end_ms'], c['end_ms']) > max(row['start_ms'], c['start_ms'])])
        row['exclusive_same_stream_ms'] = max(0., row['duration_ms'] - covered)
        key = f"stream{row['start_stream']}:{row['category']}"
        item = result.setdefault(key, dict(calls=0, inclusive_ms=0., exclusive_ms=0.))
        item['calls'] += 1
        item['inclusive_ms'] += row['duration_ms']
        item['exclusive_ms'] += row['exclusive_same_stream_ms']
    return result


def route_counts(ids, source, world, experts=160):
    """Assignment accounting, not physical NIC wire bytes or unique-rank puts."""
    owners = [0] * world
    invalid = 0
    for expert in ids:
        if expert < 0:
            invalid += 1
        elif expert >= experts:
            raise ValueError('Expert id outside configured 160 experts')
        else:
            owners[expert // (experts // world)] += 1
    buckets = dict(self_gpu=0, nvlink_pair=0, same_node_rdma=0, cross_node_rdma=0)
    for destination, count in enumerate(owners):
        key = ('self_gpu' if destination == source else
               'nvlink_pair' if destination == (source ^ 1) else
               'same_node_rdma' if destination // 4 == source // 4 else 'cross_node_rdma')
        buckets[key] += count
    return dict(source_ep_rank=source, destination_assignment_counts=owners,
                assignment_counts=buckets, padding_assignments=invalid,
                dispatch_payload_bytes={key: value * 10240 for key, value in buckets.items()},
                dispatch_payload_plus_token_record_bytes={key: value * 10256 for key, value in buckets.items()},
                combine_payload_bytes={key: value * 10240 for key, value in buckets.items()},
                semantics='Per-expert BF16 H5120 assignments: dispatch data 10240B + 16B token record; combine data 10240B. No headers/control/retransmissions counted. These are logical bytes, not measured wire bandwidth.')


class Probe:
    def __init__(self, run, arm, config):
        self.run, self.arm, self.config = Path(run).resolve(), arm, config
        self.attribution = dict(config['attribution'], detailed_ordinals=config['attribution']['by_arm'][arm])
        self.out = self.run / arm
        self.rank = None
        self.torch = None
        self.ready = self.armed = self.started = self.finished = self.done = False
        self.detail = False
        self.forward_index = self.extend_index = 0
        self.forward_meta = None
        self.events, self.cpu, self.forwards, self.stack = [], [], [], []
        self.route_references = []
        self.installation, self.labels = [], {}
        self.pool, self.pool_index = [], 0
        self.profiler = None
        self.origins = {}
        self.error = None
        self.selected_seen = []
        self._watch_pid = None
        self._busy = 0
        self._finishing = False

    def prepare(self, runner):
        from sglang.srt.runtime_context import get_parallel
        p = get_parallel()
        self.rank = p.pp_rank * 4 + p.tp_rank if self.arm == 'pp2ep4' else p.tp_rank
        self.labels = {id(module): name for name, module in runner.model.named_modules()}
        for name, module in runner.model.named_modules():
            if not name.endswith('.mlp.experts'):
                continue
            for inner in getattr(getattr(module, 'dispatcher', None), '_inners', ()):
                inner._attribution_layer = getattr(module, 'layer_id', name)
                low_latency = getattr(inner, '_low_latency_dispatcher', None)
                if low_latency is not None:
                    low_latency._attribution_layer = inner._attribution_layer
        torch = self.torch
        self.cuda_device = int(torch.cuda.current_device())
        if not torch.profiler.kineto_available():
            raise RuntimeError('Kineto is required for true GPU activity/idle attribution')
        count = int(self.attribution['event_pool_size'])
        self.pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(count)]
        # Materialize lazy native event handles outside all request cohorts.
        for low, high in self.pool:
            low.record()
            high.record()
        self.pool[-1][1].synchronize()  # startup only, not a profile/forward boundary
        # Built-in two tiny GPU markers delimit each actual forward in Kineto's
        # device clock. Identify their exact kernel name at startup, no guessing.
        marker=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False)
        marker.start()
        torch.cuda._sleep(1)
        torch.cuda.synchronize()  # startup only
        marker.stop()
        marker_path=self.out / f'marker_trace_rank{self.rank}.json'
        marker.export_chrome_trace(str(marker_path))
        marker_events=[e for e in json.loads(marker_path.read_text())['traceEvents'] if e.get('cat')=='kernel']
        if len(marker_events)!=1:
            raise RuntimeError('Cannot uniquely identify startup GPU marker')
        self.marker_name=marker_events[0]['name']
        save(self.out / f'marker_signature_rank{self.rank}.json',dict(passed=True,rank=self.rank,
             kernel_name=self.marker_name,cuda_device=self.cuda_device,startup_only=True))
        self.ready = True
        self._watch_pid = os.getpid()
        def watch():
            while not self.done:
                try:
                    if not self.armed:
                        raw = (self.out / 'ATTRIBUTION_CONTROL.json').read_bytes()
                        value = validate_marker(json.loads(raw), self.run, self.arm, self.attribution)
                        self.control_sha = hashlib.sha256(raw).hexdigest()
                        self.control = value
                        self.armed = True
                        save(self.out / f'attribution_armed_rank{self.rank}.json',
                             dict(passed=True, rank=self.rank, pid=os.getpid(),
                                  control_sha256=self.control_sha, events_preallocated=count,
                                  profiler_backend='Kineto CUDA activities only; exact GPU markers',
                                  timestamp=time.time(), startup_event_warmup_complete=True))
                    if self.armed and not self.finished:
                        raw = (self.out / 'ATTRIBUTION_FINISH.json').read_bytes()
                        validate_marker(json.loads(raw), self.run, self.arm, self.attribution, finish=True)
                        self.finished = True
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    self.error = repr(exc)
                time.sleep(.1)
        threading.Thread(target=watch, name='attribution-file-control', daemon=True).start()

    def start(self):
        if self.started or not self.armed or self.finished:
            return
        torch = self.torch
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False)
        self.profiler.start()
        self.started = True
        self.host_origin_ns, self.wall_origin_ns = time.perf_counter_ns(), time.time_ns()
        self._origin(torch.cuda.current_stream())

    def _origin(self, stream):
        key = int(stream.cuda_stream)
        if key not in self.origins:
            # Preallocated pair; origins on independent streams are intentionally
            # not synchronized/aligned. Kineto supplies the device time axis.
            if self.pool_index >= len(self.pool):
                raise RuntimeError('Attribution event pool exhausted')
            event, unused = self.pool[self.pool_index]
            self.pool_index += 1
            event.record(stream)
            self.origins[key] = event
        return key

    def call(self, function, receiver, args, kwargs, category, label=None, thin=False, cpu_only=False):
        if category != 'model_forward':
            return function(*args, **kwargs)
        active = self.started and not self.finished
        if not active:
            started = time.perf_counter_ns()
            output = function(*args, **kwargs)
            # A rank may already be blocked in request recv when CPU ack is
            # emitted; retain a CPU-only prebody interval after it wakes.
            if thin and self.armed and not self.finished and not self.started:
                self.cpu.append(dict(category=category, cpu_start_ns=started,
                                     cpu_end_ns=time.perf_counter_ns(), prebody_before_trace=True))
            return output
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Attribution requires production eager Decode')
        selected = label(receiver, args, kwargs) if callable(label) else label
        selected = str(self.labels.get(id(receiver), type(receiver).__name__) if selected is None else selected)
        parent = self.stack[-1] if self.stack else None
        layer = next((self.events[i]['label'] for i in reversed(self.stack)
                      if self.events[i]['category'] == 'layer'), None)
        row = dict(index=len(self.events), parent=parent, layer=layer, category=category,
                   label=selected, forward_index=self.forward_index, extend_index=self.extend_index,
                   input_shapes={}, cpu_start_ns=time.perf_counter_ns(), cpu_only=cpu_only)
        if not cpu_only:
            stream = self.torch.cuda.current_stream()
            row['start_stream'] = self._origin(stream)
            if self.pool_index >= len(self.pool):
                raise RuntimeError('Attribution event pool exhausted')
            row['start_event'], row['end_event'] = self.pool[self.pool_index]
            self.pool_index += 1
            row['start_event'].record(stream)
        self.events.append(row)
        self.stack.append(row['index'])
        self._busy += 1
        try:
            self.torch.cuda._sleep(1)
            try:
                return function(*args, **kwargs)
            finally:
                self.torch.cuda._sleep(1)
        finally:
            row['cpu_end_ns'] = time.perf_counter_ns()
            if not cpu_only:
                ending = self.torch.cuda.current_stream()
                row['end_stream'] = self._origin(ending)
                row['end_event'].record(ending)
            self.stack.pop()
            self._busy -= 1

    def patch(self, cls, name, category, label=None, thin=False, cpu_only=False, optional=False):
        if not hasattr(cls, name):
            if optional:
                self.installation.append(dict(class_name=cls.__name__, method=name, missing=True))
                return
            raise AttributeError(f'Required attribution hook missing: {cls.__name__}.{name}')
        descriptor = inspect.getattr_static(cls, name)
        original = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
        if isinstance(descriptor, staticmethod):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                return self.call(original, None, args, kwargs, category, label, thin, cpu_only)
            replacement = staticmethod(wrapped)
        else:
            @functools.wraps(original)
            def wrapped(receiver, *args, **kwargs):
                return self.call(lambda *a, **k: original(receiver, *a, **k), receiver,
                                 args, kwargs, category, label, thin, cpu_only)
            replacement = classmethod(wrapped) if isinstance(descriptor, classmethod) else wrapped
        setattr(cls, name, replacement)
        self.installation.append(dict(class_name=cls.__name__, method=name, category=category,
                                      thin=thin, cpu_only=cpu_only))

    def finish(self):
        if not self.finished or self.done or not self.started or self._busy or self._finishing:
            return
        self._finishing = True
        try:
            # Stop/export is a single post-client boundary, never inside a
            # forward, P2P interval, or formal request window.
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.out / f'attribution_trace_rank{self.rank}.json'))
            trace_data = json.loads((self.out / f'attribution_trace_rank{self.rank}.json').read_bytes())
            kernel_events = [row for row in trace_data.get('traceEvents', []) if row.get('cat') == 'kernel']
            kernel_count = len(kernel_events)
            kernel_devices = sorted({str(row.get('args', {}).get('device', row.get('pid')))
                                     for row in kernel_events})
            kernel_device_matches_rank = kernel_devices == [str(self.cuda_device)]
            del trace_data, kernel_events
            route_rows = []
            world = int(self.config['cases'][self.arm]['ep'])
            source = self.rank % 4 if self.arm == 'pp2ep4' else self.rank
            for route in self.route_references:
                tensor = route.pop('tensor')
                values = tensor.detach().cpu().reshape(-1).tolist()
                route_rows.append(dict(**route, **route_counts(values, source, world)))
            self.route_references.clear()
            rows, cpu_rows, unresolved = [], [], []
            for raw in self.events:
                row = {key: value for key, value in raw.items() if not key.endswith('_event')}
                if row['cpu_only']:
                    cpu_rows.append(row)
                    continue
                if not raw['start_event'].query() or not raw['end_event'].query():
                    unresolved.append(row['index'])
                    continue
                if row['start_stream'] == row['end_stream']:
                    origin = self.origins[row['start_stream']]
                    row.update(start_ms=origin.elapsed_time(raw['start_event']),
                               end_ms=origin.elapsed_time(raw['end_event']),
                               duration_ms=raw['start_event'].elapsed_time(raw['end_event']))
                else:
                    row.update(start_ms=self.origins[row['start_stream']].elapsed_time(raw['start_event']),
                               end_ms=self.origins[row['end_stream']].elapsed_time(raw['end_event']),
                               duration_ms=None, excluded_from_additive_summary=True)
                rows.append(row)
            same_stream = [r for r in rows if r['start_stream'] == r['end_stream']]
            result = dict(schema_version=1, passed=self.error is None and not unresolved and bool(rows) and kernel_count > 0 and kernel_device_matches_rank,
                          arm=self.arm, rank=self.rank, node_rank=self.rank//4, local_rank=self.rank%4,
                          hostname=os.uname().nodename, pid=os.getpid(), control_sha256=self.control_sha,
                          detailed_forward_ordinals=self.attribution['detailed_ordinals'],
                          detail_ordinals_seen=self.selected_seen, events=rows,
                          cpu=self.cpu + cpu_rows, forwards=self.forwards,
                          per_stream_summary=summarize(same_stream), installation=self.installation,
                          event_pool_size=len(self.pool), event_pairs_used=self.pool_index,
                          unresolved_event_indices=unresolved, error=self.error,
                          trace_filename=f'attribution_trace_rank{self.rank}.json',
                          profiling_method='cuda_activity_exact_name_calibration',marker_kernel_name=self.marker_name,
                          gpu_kernel_trace_available=kernel_count > 0, gpu_kernel_event_count=kernel_count,
                          cuda_device=self.cuda_device, gpu_kernel_devices=kernel_devices,
                          gpu_kernel_device_matches_rank=kernel_device_matches_rank,
                          routes=route_rows,
                          route_capture='Disabled for CUDA-only activity measurement; no routing tensors retained.',
                          host_origin_ns=self.host_origin_ns, wall_origin_ns=self.wall_origin_ns,
                          added_segment_synchronizations=0, added_segment_barriers=0,
                          readout_after_client_completion=True,
                          timing_semantics='CUDA activities only, without CPU operator traces or inner phase events. Startup-identified built-in GPU markers bound forwards. Exact kernel names must pass independent CPU-scope calibration and held-out checks. Per-device GPU timeline partition includes idle/overlap/unknown explicitly; never sum independent rank clocks.',
                          cross_host_alignment='Host wall-clock anchors recorded; no global GPU clock synchronization assumed. Render host/rank lanes separately unless offset uncertainty is independently established.')
            traces = []
            for row in rows:
                if row['start_stream'] != row['end_stream']:
                    continue
                traces.append(dict(name=row['category'] + ':' + row['label'], cat='CUDA_event_API_interval',
                                   ph='X', pid=self.rank, tid=f"cuda_stream_{row['start_stream']}",
                                   ts=row['start_ms']*1000, dur=row['duration_ms']*1000,
                                   args={key: row[key] for key in ('layer','forward_index','extend_index','input_shapes')}))
            for row in self.cpu + cpu_rows + self.forwards:
                traces.append(dict(name=row.get('category','model_forward'), cat='CPU_host_interval', ph='X',
                                   pid=self.rank, tid='host', ts=(row['cpu_start_ns']-self.host_origin_ns)/1000,
                                   dur=(row['cpu_end_ns']-row['cpu_start_ns'])/1000,
                                   args={key:value for key,value in row.items() if key not in ('cpu_start_ns','cpu_end_ns')}))
            save(self.out / f'timeline_rank{self.rank}.json', dict(traceEvents=traces,
                 displayTimeUnit='ms', timing_warning=result['timing_semantics']))
            save(self.out / f'attribution_rank{self.rank}.json', result)
            self.done = True
        except Exception as exc:
            save(self.out / f'attribution_rank{self.rank}.json', dict(passed=False, rank=self.rank,
                 error=repr(exc), traceback_location='post-client finish/export', added_segment_synchronizations=0))
            self.done = True
        finally:
            self._finishing = False


def layer_label(obj, args, kwargs):
    return getattr(obj, 'layer_id', getattr(getattr(obj, 'attn', None), 'layer_id', 'unknown'))


def linear_label(obj, args, kwargs):
    layer = kwargs.get('layer', args[0] if args else None)
    return str(getattr(layer, 'prefix', type(layer).__name__))


def gemm_label(obj, args, kwargs):
    return str(kwargs.get('sublayer_name', 'linear')) + ':' + linear_label(obj, args, kwargs)


def group_label(obj, args, kwargs):
    return f"size{obj.world_size}:ranks={list(obj.ranks)}"


def install():
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    run, arm = Path(os.environ['GLM_RUN_DIR']).resolve(), os.environ['ARM']
    probe = Probe(run, arm, json.loads((run / 'config.json').read_bytes()))
    import torch
    from humming.layer import HummingMethod
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplLowLatency
    from sglang.srt.layers.quantization.humming import HummingLinearMethod
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.glm4_moe import Glm4MoeMLP, Glm4MoeAttention, Glm4MoeDecoderLayer, Glm4MoeGate, Glm4MoeSparseMoeBlock
    probe.torch = torch
    original_load = ModelRunner.load_model
    @functools.wraps(original_load)
    def load(runner, *args, **kwargs):
        result = original_load(runner,*args,**kwargs)
        probe.prepare(runner)
        return result
    ModelRunner.load_model = load

    original_forward = ModelRunner.forward
    @functools.wraps(original_forward)
    def forward(runner, forward_batch, *args, **kwargs):
        if not probe.armed or probe.finished:
            return original_forward(runner,forward_batch,*args,**kwargs)
        probe.start()
        probe.forward_index += 1
        meta = metadata(forward_batch)
        qualified = forward_batch.forward_mode.is_decode() and forward_batch.batch_size == 4
        if qualified:
            probe.extend_index += 1
        probe.detail = qualified and probe.extend_index in probe.attribution['detailed_ordinals']
        if probe.detail:
            probe.selected_seen.append(probe.extend_index)
        probe.forward_meta = meta
        entry = dict(category='model_forward', forward_index=probe.forward_index,
                     extend_index=probe.extend_index, detailed=probe.detail,
                     metadata=meta, cpu_start_ns=time.perf_counter_ns())
        try:
            return probe.call(lambda *a,**k: original_forward(runner,*a,**k),runner,
                              (forward_batch,*args),kwargs,'model_forward',thin=True)
        finally:
            entry['cpu_end_ns'] = time.perf_counter_ns()
            probe.forwards.append(entry)
            probe.detail = False
    ModelRunner.forward = forward

    original_boundary = Scheduler.get_next_batch_to_run
    @functools.wraps(original_boundary)
    def boundary(scheduler,*args,**kwargs):
        probe.finish()
        return probe.call(lambda *a,**k: original_boundary(scheduler,*a,**k),scheduler,
                          args,kwargs,'scheduler_get_next_batch',thin=True,cpu_only=True)
    Scheduler.get_next_batch_to_run = boundary
    # Both PP stages pass this boundary; only the first PP stage schedules its
    # own get_next_batch_to_run. Finish is checked before blocking receive and
    # again after wakeup so the client control ping reaches every rank.
    original_receive = SchedulerRequestReceiver.recv_requests
    @functools.wraps(original_receive)
    def receive(receiver,*args,**kwargs):
        probe.finish()
        result = probe.call(lambda *a,**k: original_receive(receiver,*a,**k),receiver,
                            args,kwargs,'scheduler_receive_requests',thin=True,cpu_only=True)
        probe.finish()
        return result
    SchedulerRequestReceiver.recv_requests = receive
    probe.installation.extend([dict(class_name='ModelRunner',method='forward',category='model_forward'),
                               dict(class_name='Scheduler',method='get_next_batch_to_run',category='scheduler_boundary_finish'),
                               dict(class_name='SchedulerRequestReceiver',method='recv_requests',category='all_rank_prebody_finish')])
    _PROBE = probe
    return probe
