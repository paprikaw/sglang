"""CPU-only proof from initialized rank state; not a numerical guarantee.

Native deterministic inference is a correctness control, not a promise that the
third-party Humming or DeepEP kernels are deterministic. Strict generated-token
repeatability and cross-arm parity remain necessary.
"""
import os


def capture(scheduler):
    from sglang.srt.runtime_context import get_exec, get_server_args
    args = get_server_args()
    backend = scheduler.tp_worker.model_runner.attn_backend
    return dict(
        source='initialized_scheduler_rank',
        resolved_enabled=bool(get_exec().deterministic.enable_deterministic_inference),
        declared_enabled=bool(args.enable_deterministic_inference),
        resolved_attention_backend=args.attention_backend,
        actual_target_backend=type(backend).__name__,
        actual_target_backend_module=type(backend).__module__,
        actual_target_deterministic=getattr(backend, 'enable_deterministic', None),
        nccl_algo=os.environ.get('NCCL_ALGO'),
        nccl_min_nchannels=os.environ.get('NCCL_MIN_NCHANNELS'),
        nccl_max_nchannels=os.environ.get('NCCL_MAX_NCHANNELS'),
        actual_truncation_align_size=scheduler.truncation_align_size,
        actual_chunked_prefill_size=scheduler.chunked_prefill_size,
        effective_max_prefill_tokens=args.max_prefill_tokens,
        configured_truncation_alignment=os.environ.get('SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE'),
        disable_custom_all_reduce=bool(args.disable_custom_all_reduce),
        enable_torch_symm_mem=bool(args.enable_torch_symm_mem),
        numerical_determinism_guaranteed=False,
    )


def validate(rows):
    if len(rows) != 8 or sorted(row['rank'] for row in rows) != list(range(8)):
        raise RuntimeError('Deterministic proof requires all eight initialized ranks')
    proofs=[]
    for row in rows:
        p=row.get('deterministic', {})
        if (p.get('source') != 'initialized_scheduler_rank'
                or p.get('resolved_enabled') is not True
                or p.get('declared_enabled') is not True
                or p.get('resolved_attention_backend') != 'triton'
                or p.get('actual_target_backend') != 'TritonAttnBackend'
                or p.get('actual_target_backend_module') != 'sglang.srt.layers.attention.triton_backend'
                or p.get('actual_target_deterministic') is not True
                or p.get('nccl_algo') != 'allreduce:tree'
                or p.get('nccl_min_nchannels') != '8'
                or p.get('nccl_max_nchannels') != '8'
                or p.get('actual_truncation_align_size') != 1024
                or p.get('actual_chunked_prefill_size') != 1024
                or p.get('effective_max_prefill_tokens') != 1024
                or p.get('configured_truncation_alignment') != '1024'
                or p.get('disable_custom_all_reduce') is not True
                or p.get('enable_torch_symm_mem') is not False):
            raise RuntimeError('Actual deterministic runtime proof mismatch at rank '+str(row['rank'])+': '+str(p))
        proofs.append(dict(rank=row['rank'],**p))
    return dict(passed=True,ranks=proofs,numerical_determinism_guaranteed=False,
                strict_repeat_and_cross_arm_parity_still_required=True)
