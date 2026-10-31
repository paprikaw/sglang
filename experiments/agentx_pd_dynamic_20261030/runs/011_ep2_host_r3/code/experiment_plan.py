"""Capacity-relative, equal-GPU Qwen comparisons on A100; CPU-only planning.

Only real allocator capacities measured with the frozen state/Graph/transport
budgets are accepted. EP1 calibration also records the single-GPU baseline;
the formal small-model comparison spends two GPUs in both arms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Layout:
    id: str
    replicas: int
    tp: int
    ep: int
    dp: int = 1
    pp: int = 1

    def __post_init__(self):
        if (type(self.replicas) is not int or self.replicas < 1
                or type(self.tp) is not int or self.tp < 1
                or self.ep != self.tp or self.dp != 1 or self.pp != 1):
            raise ValueError('Positive replicas/TP, Attention TP=EP, DP1 and PP1 are required')

    @property
    def gpu_count(self):
        return self.replicas * self.tp

    @property
    def gpu_groups(self):
        return [list(range(i * self.tp, (i + 1) * self.tp))
                for i in range(self.replicas)]


LAYOUTS = {
    'small': (Layout('2xep1', 2, 1, 1), Layout('ep2', 1, 2, 2)),
    'large': (Layout('2xep2', 2, 2, 2), Layout('ep4', 1, 4, 4)),
}
MODELS = {
    'small': dict(repo_id='Qwen/Qwen3.6-35B-A3B', dtype='bfloat16', quantization=None),
    'large': dict(repo_id='Qwen/Qwen3.5-122B-A10B-FP8', dtype='bfloat16', quantization='fp8'),
}
WORKLOADS = {
    'prefill': dict(query_tokens=512, output_tokens=1),
    'decode': dict(query_tokens=1, output_tokens=128),
}
PROTOCOL = dict(
    prefix_source='finite_synthetic_full_kv_and_gdn_endpoint',
    initial_state='native_cpu_hicache_hot_target_gpu_prefix_cold',
    timed_source_bank_reads=False, timed_native_h2d=True,
    timed_admission_wait=True, timed_output=True,
    timed_client_payload_preparation=False,
    between_requests_prefix_sharing=False,
    capacity_source='real_allocator_after_workspace_gdn_and_graph_allocation',
    context_length=327680, page_size=64, base_request_count=32,
    single_request_cached_context_goal=300000,
    capacity_queueing='native_admission_waves_not_single_request_streaming',
)


def state_slot_budget(layout, request_count):
    """Freeze native extra-buffer/overlap state slots per locally owned request."""
    if (type(request_count) is not int or request_count <= 0
            or request_count % layout.replicas):
        raise ValueError('Requests must split evenly across replicas')
    local = request_count // layout.replicas
    return dict(max_running_requests=local, max_mamba_cache_size=5 * local + 4,
                request_count=request_count, mandatory_allocator_recalibration=True)


def validate_capacities(cohort, capacities):
    layouts = LAYOUTS[cohort]
    if set(capacities) != {layout.id for layout in layouts}:
        raise ValueError('Capacity observations must cover exactly this model cohort')
    if len({layout.gpu_count for layout in layouts}) != 1:
        raise ValueError('Every formal arm in a model cohort must use the same GPU budget')
    for layout in layouts:
        row = capacities[layout.id]
        actual = row.get('per_replica_tokens')
        if (not isinstance(actual, list) or len(actual) != layout.replicas
                or any(type(value) is not int or value <= 0 for value in actual)
                or row.get('source') != 'actual_allocator'
                or row.get('hybrid_restore_smoke_passed') is not True):
            raise ValueError('Only measured allocator capacities with proven hybrid restore are accepted')


def make_matrix(cohort, capacities, *, maximum_context=327680, page_size=64):
    validate_capacities(cohort, capacities)
    if (type(maximum_context) is not int or maximum_context <= 0
            or type(page_size) is not int or page_size <= 0):
        raise ValueError('Positive integer context and page sizes are required')
    layouts = LAYOUTS[cohort]
    totals = {layout.id: sum(capacities[layout.id]['per_replica_tokens'])
              for layout in layouts}
    references = [('shared_low', min(totals.values()), .5)]
    references += [(layout.id + '_near_full', totals[layout.id], .95)
                   for layout in layouts]
    request_count = PROTOCOL['base_request_count']
    reserve = max(row['query_tokens'] + row['output_tokens']
                  for row in WORKLOADS.values())
    smallest_replica = min(min(row['per_replica_tokens']) for row in capacities.values())
    per_request_limit = min(maximum_context - reserve, smallest_replica - reserve)
    if per_request_limit < page_size:
        raise ValueError('Even one request cannot fit the smallest replica with new-token reserve')
    points = []
    for name, total, fraction in references:
        requested_total = int(total * fraction)
        prefix = requested_total // (request_count * page_size) * page_size
        if prefix < page_size:
            raise ValueError('Capacity point cannot supply one complete prefix page per request')
        if prefix > per_request_limit:
            raise ValueError('A single request exceeds a device/context limit at fixed N32; '
                             'change and recalibrate the request/state budget explicitly, '
                             'because HiCache cannot stream one oversized request')
        points.append(dict(id=name, requested_total_context=requested_total,
                           total_context=prefix * request_count, requests=request_count,
                           per_request_cached_tokens=prefix))
    cases = []
    for layout in layouts:
        for point in points:
            for workload, tokens in WORKLOADS.items():
                case = dict(**point, layout=asdict(layout), gpu_groups=layout.gpu_groups,
                    workload=workload, **tokens,
                    request_indices_by_replica=[list(range(index, request_count, layout.replicas))
                                               for index in range(layout.replicas)],
                    aggregate_residency_overflow=(point['total_context'] + request_count *
                        (tokens['query_tokens'] + tokens['output_tokens']) > totals[layout.id]))
                case['point_id'] = point['id']
                case['id'] = f'{layout.id}.{point["id"]}.{workload}'
                cases.append(case)
    return dict(cohort=cohort, model=MODELS[cohort], protocol=PROTOCOL,
                gpu_budget=layouts[0].gpu_count,
                capacities=capacities, points=points, cases=cases,
                expected_valid_conditions=len(cases), performance_measured=False)
