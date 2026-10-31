"""CPU-only metadata and legacy saved-trace parser; no collector.
"""
MARKER_PROOF_SCOPES = tuple(f'stage-marker-proof:{i}' for i in range(3))


def marker_signature(events, expected_device):
    """Resolve actual marker kernels by CPU scope/runtime correlation only.

    CUPTI can omit the first tiny GPU activity after earlier profiler sessions,
    even when its runtime launch is present. Require two independent positive
    GPU witnesses, preserve missing witnesses, and never infer a kernel name.
    This CPU-only helper deliberately does not classify by name substrings.
    """
    def argument(event, *names):
        values = {str(k).lower().replace('_', ' '): str(v)
                  for k, v in event.get('args', {}).items()}
        return next((values[n] for n in names if n in values), None)

    def inside(event, scope):
        return (event.get('pid') == scope.get('pid') and event.get('tid') == scope.get('tid')
                and float(scope['ts']) <= float(event['ts'])
                and float(event['ts']) + float(event.get('dur', 0))
                <= float(scope['ts']) + float(scope.get('dur', 0)) + .001)

    launches = [e for e in events if e.get('ph') == 'X'
                and e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    kernels = [e for e in events if e.get('ph') == 'X'
               and e.get('cat') in ('kernel', 'cuda_kernel')]
    proofs = []
    observed_names = set()
    for label in MARKER_PROOF_SCOPES:
        # CUPTI can mirror record_function as gpu_user_annotation with the
        # same name. Only the real CPU scope can own a runtime launch.
        scopes = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                  and e.get('name') == label]
        if len(scopes) != 1:
            raise RuntimeError('Marker proof requires one actual CPU scope: ' + label)
        correlations = {argument(e, 'correlation', 'correlation id')
                        for e in launches if inside(e, scopes[0])}
        correlations.discard(None)
        linked = [e for e in kernels
                  if argument(e, 'correlation', 'correlation id') in correlations]
        if len(linked) > 1:
            raise RuntimeError('Marker proof scope launched multiple observed GPU kernels: ' + label)
        evidence = dict(scope=label, runtime_correlations=sorted(correlations),
                        actual_gpu_kernel_count=len(linked))
        if linked:
            kernel = linked[0]
            device = argument(kernel, 'device')
            if device != str(expected_device):
                raise RuntimeError('Marker proof is on the wrong/unknown device: ' + str(device))
            observed_names.add(kernel['name'])
            evidence.update(name=kernel['name'], device=int(device),
                            stream=argument(kernel, 'stream'),
                            correlation=argument(kernel, 'correlation', 'correlation id'),
                            duration_us=float(kernel['dur']))
        proofs.append(evidence)
    observed = sum(p['actual_gpu_kernel_count'] for p in proofs)
    if observed < 2 or len(observed_names) != 1:
        raise RuntimeError('Marker requires at least two independently scoped actual GPU kernels with one exact name; observed=' + str(observed))
    return dict(name=next(iter(observed_names)), observed_proof_count=observed,
                missing_proof_count=len(proofs) - observed, proofs=proofs,
                identification='CPU proof scope -> actual runtime correlation -> GPU kernel; at least two independent positive witnesses; no guessed names')


def batch_metadata(batch, frame=None):
    """Read CPU identities/shapes only; never materialize a CUDA tensor."""
    requests = getattr(batch, 'reqs', None)
    ids = [r.rid for r in requests] if requests is not None else getattr(batch, 'rids', None)
    if not isinstance(ids, (tuple, list)) or not all(isinstance(x, str) for x in ids):
        ids = (frame or {}).get('request_ids', [])
    size = len(requests) if requests is not None else getattr(batch, 'batch_size', None)
    if type(size) is not int:
        size = (frame or {}).get('batch_size')
    inputs = getattr(batch, 'input_ids', None)
    shape = list(inputs.shape) if inputs is not None and hasattr(inputs, 'shape') else None
    lengths = getattr(batch, 'seq_lens_cpu', None)
    # A CPU Tensor conversion is deliberately excluded as well: receipts keep
    # only already-Python metadata so this helper is safe on every code path.
    lengths = list(lengths) if isinstance(lengths, (list, tuple)) else None
    return dict(request_ids=list(ids), batch_size=size, input_shape=shape,
                mode=getattr(getattr(batch, 'forward_mode', None), 'name', None),
                seq_lens_cpu=lengths, pp_microstep=(frame or {}).get('pp_microstep'))


def target_phase(batch, is_draft_worker=False):
    if is_draft_worker:
        return 'draft_model'
    mode = getattr(batch, 'forward_mode', None)
    if mode is not None and mode.is_target_verify():
        return 'target_verify'
    if mode is not None and mode.is_decode():
        return 'target_decode'
    return None


