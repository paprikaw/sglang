# Elastic PP manual validation

## Qwen3.5 hybrid attention

The Qwen3.5 adapter migrates decoder weights, full-attention token KV, and GDN
convolution/recurrent state. GDN state uses request slots, not token KV slots.
Preparation allocates destination storage while the source generation remains
live; state transfer and publication happen at the drained scheduler fence.
Retained GDN layers are copied at that fence, not during background preparation.
The resident scheduler ring keeps its default concurrency: output forwarding is
aligned with the slot being retired, and the autoscaler counts unfinished
requests across every ring slot, including in-flight prefills.
Post-commit cleanup removes retired parameters' loader reference cycles so old
weights can be reclaimed even after SGLang freezes GC at server warmup.

The current supported path has these constraints:

- Tensor parallel size 1 with a fixed resident PP process group. Logical stages
  can become inactive; this does not release the Slurm booking or GPU process.
- Unquantized weights or `moe_wna16` for the official
  `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` checkpoint. Pass
  `--disable-shared-experts-fusion` for this checkpoint's BF16 shared experts.
- Fixed positive `--max-running-requests`, identical request/state capacities on
  every rank, and `--disable-radix-cache`.
- No speculative decoding, ReplaySSM, int8 Mamba checkpoints, LoRA, hierarchical
  cache, unified memory, page-major layout, or quantized KV cache.
- The manual recipe below uses text-only serving and disables CUDA graphs.
  Multimodal input and other backend combinations need separate validation.

Allocator identities are checked collectively before any migration. A mismatch
aborts the transition; the implementation does not guess request-slot remapping.

### Two-node, eight-GPU correctness recipe

Use only idle, already-booked nodes after the workspace's experiment preflight.
Run the following on each node, setting its rank to 0 or 1 and using the same
leader address. The two partitions have four and eight **logical** stages over
the same eight resident ranks. Each node needs four GPUs.

```bash
MODEL=/path/to/Qwen3.5-397B-A17B-GPTQ-Int4
PP_NODE_RANK=0
PP_DIST_INIT_ADDR=leader-hostname:29844

export CUDA_VISIBLE_DEVICES=0,1,2,3
export KVCACHED_AUTOPATCH=0
export KVCACHED_LAYER_STACKING=true
export KVCACHED_CONTIGUOUS_LAYOUT=false
export KVCACHED_PAGE_PREALLOC_ENABLED=false
export KVCACHED_PAGE_SIZE_MB=2
export KVCACHED_PHYSICAL_BLOCK_SIZE_KB=2048
export KVCACHED_MAX_CACHED_TOKENS=0
export KVCACHED_NUM_PP_STAGES=8
export KVCACHED_IPC_NAME=qwen35_elastic_validation

python -m sglang.launch_server \
  --model-path "$MODEL" --language-model-only \
  --host 0.0.0.0 --port 29843 \
  --tp-size 1 --pp-size 8 --nnodes 2 \
  --node-rank "$PP_NODE_RANK" --dist-init-addr "$PP_DIST_INIT_ADDR" \
  --dtype bfloat16 --quantization moe_wna16 \
  --disable-shared-experts-fusion \
  --attention-backend triton --mamba-backend triton \
  --moe-runner-backend triton --sampling-backend pytorch \
  --context-length 2048 --chunked-prefill-size 128 --page-size 1 \
  --max-total-tokens 4096 --max-running-requests 8 --max-mamba-cache-size 8 \
  --mem-fraction-static 0.90 --disable-radix-cache --disable-cuda-graph \
  --random-seed 42 --pp-async-batch-depth 0 \
  --enable-elastic-pp --elastic-pp-kv-backend kvcached \
  --elastic-pp-initial-partition 15,0,15,0,15,0,15,0 \
  --enable-elastic-pp-autoscaling \
  --elastic-pp-autoscale-partitions '15,0,15,0,15,0,15,0;8,7,8,7,8,7,8,7' \
  --elastic-pp-autoscale-up-requests 3 --elastic-pp-autoscale-down-requests 1 \
  --elastic-pp-autoscale-up-stable-polls 2 \
  --elastic-pp-autoscale-down-stable-polls 4 \
  --elastic-pp-autoscale-poll-interval-s 0.25 \
  --elastic-pp-autoscale-min-residence-s 5 \
  --elastic-pp-autoscale-failure-backoff-s 600 \
  --elastic-pp-autoscale-max-operations 2 \
  --watchdog-timeout 600 --dist-timeout 1800
```

The 60-layer model has only 15 full-attention layers. KVCacheD's backing pool
therefore uses 15 attention ordinals, while the model and GDN maps retain global
decoder IDs. The 2 MiB page/block settings above give layer-group granularity
one, allowing every required attention boundary. `pp_async_batch_depth=0` keeps
the default physical-ring concurrency; it does not force one in-flight batch.

After the server is ready, run once against the leader:

```bash
python test/manual/elastic_pp/test_qwen35_autoscaling.py \
  --base-url http://leader-hostname:29843 \
  --output /path/to/run/results.json
```

The client first obtains four single-request, fixed-topology controls, then
sends the same four requests concurrently. It requires automatic PP4 -> PP8 ->
PP4 transitions, a live request across each transition, exact output-token
agreement with every control, and matching output after the complete cycle.
It never calls the manual reconfiguration API. Its short contexts and small
request capacity are for correctness validation, not a serving-performance or
overlap benchmark. Keep server logs and the exact launch config beside the
result; a successful process launch alone is not a passing result.
