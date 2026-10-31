# EP / PP deployment: communication-domain page

Five English slides. The fifth page adds the conceptual GPU-local -> NVLink -> cross-node
progression and redraws both supplied figure datasets. Six within-cohort throughput ratios
and nine standalone Dispatch+Combine medians are shown, retaining both local nodes and
the observed min/max whiskers. Twelve complete domain request-metric rows are included.

The paired deployments are EP2/PP2 on two GPUs and EP8/PP2xEP4 on eight GPUs. The Expert
execution versus repeated cross-domain exchange trade-off is the page's decision rule.
Models, cache starts, workloads and timing scopes are detailed in the English speaker notes.
The single-GPU card is conceptual. The original four page previews are preserved exactly.

Rebuild from the workspace experiment root: build_english.py --with-domains.
