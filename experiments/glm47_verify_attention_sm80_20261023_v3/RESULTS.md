# A100 Verify Attention qualification

One booked A100 80GB PCIe on spartan-gpgpu065, physical GPU0. No model or Expert computation. GLM TP4-local shapes: 24 Q heads, 2 KV heads, head dimension128, BF16, page64. Each latency is the median of3 CUDA-Graph blocks,32 replays/block. Timings below use the actual131072-token context-cap scratch/launch capacity and include metadata rebuilding in every layer.

| B | Cached P | Verify Q | Original unified (ms) | Integrated causal Decode (ms) | Speedup | Same-cap ordinary Q1 Decode (ms) |
|---:|---:|---:|---:|---:|---:|---:|
|4|4096|3|0.604544|0.068960|8.77x|0.047584|
|4|16384|3|2.391264|0.226496|10.56x|0.089600|
|4|4096|1|0.601184|0.059104|10.17x|0.047552|

## Source diagnosis

The original four-way run enabled deterministic inference. TritonAttnBackend.forward_extend returns via _forward_extend_unified before the ordinary gfx95 Verify dispatcher. Actual Verify therefore uses a query tileM128,N128,8warps with a serial KV loop. Normal Decode uses grouped Split-KV and deterministic256-token split sizing. The previous description of ordinary extend_attention_fwd fallback was incorrect for these frozen arguments.

## Candidate

Top-k-one Verify is a causal chain. Treat each of its3 queries as a Decode query with prefix lengthsP+1,P+2,P+3 and reuse the existing grouped Decode and reduction kernels. The metadata expansion and all kernel launches are timed. The prototype rebuilds metadata per layer to avoid stale-cache and CUDA-Graph replay assumptions. It preserves physical KV slot IDs, including non-contiguous index views.

## Correctness and limits

All3 cases pass an independent FP32 attention reference at atol0.002/rtol0.02. The integrated output matches independent ordinary Decode bitwise. A single captured Graph also matches after in-place changes to ragged per-request prefix lengths, including zero, then after restoring the prefixes. This establishes these tested kernel shapes, not corpus-wide lossless MTP. Original unified and Decode have different floating-point reduction orders; bitwise equality to the original unified output is not claimed.

The serving adapter is opt-in and qualified only for SM80, deterministic Triton, BF16, page64,24Q/2KV heads,D128,topk1,Q3,full causal attention without SWA/sinks/logit-cap/score modifiers. All other modes retain the original path. Full-model TPOT must be measured separately; a kernel speedup is not a full-model speedup.

## Audit

r1 isolated kernels all passed. r2 found an actual non-contiguous ExtendIDs stride bug in the extraQ1 test; its failed evidence is retained. r3 fixes index strides and passes3/3. Guard postcheck and owned cleanup pass; global lease released. No weights or KV Bank were modified. Raw results are in run_v1/worker_summary.json.
