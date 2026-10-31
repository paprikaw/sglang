# EP / PP deployment: communication-domain page

Five English slides. The fifth page compares only two deployment cases: two GPUs/NVLink
(Qwen122B FP8, B8) and eight GPUs/two nodes (GLM4.7 INT4, B4). The eight-GPU B16 row is
omitted. Prefill and Decode yield four within-cohort throughput ratios. The nine standalone
Dispatch+Combine medians retain both local nodes and observed min/max whiskers. Original
source tables remain in domain_data; plotted_throughput.csv identifies the four visible bars.

The paired deployments are EP2/PP2 on two GPUs and EP8/PP2xEP4 on eight GPUs. The Expert
execution versus repeated cross-domain exchange trade-off is the page's decision rule.
Models, cache starts, workloads and timing scopes are detailed in the English speaker notes.
The single-GPU card is conceptual. The original four page previews are preserved exactly.

Rebuild from the workspace experiment root: build_english.py --with-domains.
