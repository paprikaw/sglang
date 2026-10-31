# H100 historical EP/PP communication-domain comparison

Offline analysis only; no new GPU samples. Two-GPU Qwen122B FP8 NVLink versus eight-GPU GLM INT4 cross-node deployments, each comparing EP and PP on its own saved workload. Different models, caches, request counts, backends and chunks prohibit network-only E2E causal attribution. Same-native EP4/EP8 communication provides a separate mechanism check.

[Complete report and both scientific figures](REPORT.md) · [12 formal metric rows](full_metrics.csv) · [source hashes](source_manifest.json) · [validation](validation.json).
