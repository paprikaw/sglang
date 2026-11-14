"""Fail before allocating GPUs for a known deterministic Prefill starvation case."""
def validate_prefill_alignment(args):
    chunk=args.chunked_prefill_size
    if args.enable_deterministic_inference and args.attention_backend=='triton':
        if chunk is not None and chunk>0 and (chunk<4096 or chunk%4096):
            raise ValueError('Deterministic Triton Prefill requires a positive chunk aligned to 4096; smaller chunks cannot admit long requests')
