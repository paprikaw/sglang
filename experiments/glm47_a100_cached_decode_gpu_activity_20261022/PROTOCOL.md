# GPU activity confirmation

Same GLM-4.7-AWQ B8/P81920/Q1,32-output hot-prefix Decode and exact065/167 nodes, engine, native DeepEP and layout as prior arms.

Earlier CPU+CUDA traces and all-layer/sparse phase events were observer-effect diagnostics: sparse whole-step overhead was small but independent layer envelope controls still measured20% local phase-scope overhead. Do not report those phase percentages as the uninstrumented baseline.

This version removes all inner phase wrappers and route tensor capture. Collect CUDA activities only. At startup capture exactly one built-in torch.cuda._sleep(1) kernel to identify its exact name; two such tiny markers bound each native model_forward on the actual GPU clock. Keep only one preallocated event pair per forward for an independent duration cross-check. No mid-step barriers, synchronizations, CPU tensor reads or CPU operator traces.

Classify exact GPU kernel names using the prior same-engine CPU-correlated full trace: training and held-out detailed forwards on ranks0 and4 for both layouts. Every eligible name must have one broad phase across direct observations and held-out witnesses; conflicting/unseen names remain unknown. This is calibrated attribution, not kernel-name substring guessing.

Partition detailed stable B4 Decode marker windows into Attention, Expert/FFN, communication, other/unknown, overlap and idle. Communication kernel activity includes genuine polling/waits; it is not pure wire time. PP boundary kernels within windows are classified separately; do not align independent GPU/host clocks or sum concurrent stages. Compare CUDA-only cohort TPOT and forward event duration against unprofiled controls and the envelope-only observations.
