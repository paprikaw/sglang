# Conditional PP47/45 Decode Graph candidate

Status: **NOT_RUN**. CPU tests passed; no GPU or Guard launch. The matched
46/46 Graph comparison has already met the TPOT objective according to the
parent run's result, so this conditional candidate will not be launched.
No performance or GPU-correctness result is claimed for PP47/45.

The same complete GLM-4.7-AWQ model, pinned engine/native, CPU weight bank,
B8/P81920/Q1/128-output CPU-cold HiCache protocol and Full B4 Decode Graph
remain unchanged. EP8 receives no layer-partition environment override. PP2
alone receives `SGLANG_PP_LAYER_PARTITION=47,45`: stage0 owns layers0–46
(3 dense,44 MoE), stage1 layers47–91 (45 MoE plus final output processing).

Startup validates actual `model.start_layer/end_layer` and real MoE counts.
The validated interval is held in process memory and reused by native host-KV
construction. Stage1 synthetic KV starts at global layer47, not46; host pools
must report exactly47/45 layers. No file polling is added inside Decode.

`prepare_measure.py` defaults to `balance_r1`, has a distinct experiment ID,
and retains the original20261022 `runs/preload_r2` full CPU-bank references.
Preparing this candidate does not authorize or perform a GPU launch.
