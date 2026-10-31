# DeepEP Graph profiler isolation

Same native/wrapper and pair-NVLink + nonpair-RDMA CPU-host-memory NIC-handler policy as GLM four-way v7. Exact booked nodes 065/167, 8 A100, no model or expert/attention compute.

Each of three modes uses fresh processes: CPU+CUDA; default CUDA-only (still runtime/driver callbacks); CUDA concurrent-kernel-only (no runtime/driver callback correlation). Local 12 BF16 rows, H5120, top8, E160, capacity256, 20 QPs/rank. Every token visits exactly one expert on each rank; each rank receives 96 assignments and each expert receives 4 or 5.

Order: eager correctness and warm; capture one dispatch+combine Graph retaining pointers; two native replays and exact payload/source/count check; first and only profiler; idle fill and three scoped sleep proofs; one eager exchange; three scoped Graph replays; stop/export; exact identity check; changed payload Graph replay check. Actual loaded-library maps are saved before capture but CUPTI version is queried only after the collector starts.

This is a crash/isolation diagnostic, not a latency benchmark. Raw trace names/counts and exact changed-payload identity are retained. Missing GPU events remain invalid. Filtered mode cannot provide scoped runtime correlation and is never used for formal Attention/Expert attribution. Each worker mode has a 60-second deadline; any crash must receive exact owned-process and post-exit GPU cleanup before the next fresh mode.

Only root may launch the frozen spec through canonical Experiment Guard. No full model retries are part of this diagnostic.
