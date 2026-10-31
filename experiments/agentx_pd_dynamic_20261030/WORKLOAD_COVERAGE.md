# AgentX workload coverage

The current replay is a bounded, explicitly selected AgentX-derived cohort. It is not an unbiased sample or an official score.

| Quantity | Full 256k source | Selected complete graphs |
|---|---:|---:|
| Root sessions | 393.0000 | 8.0000 |
| Calls | 68266.0000 | 171.0000 |
| Mean input tokens | 100946.7211 | 84856.1404 |
| Mean output tokens | 860.2937 | 435.7193 |
| Nested calls | 39822.0000 | 45.0000 |
| Nested call fraction | 0.5833 | 0.2632 |
| Largest input tokens | 255808.0000 | 208320.0000 |

Eligibility: Whole sessions, 4..80 calls, <=16000 output tokens, <=256000 total context; evenly spaced across max context; upstream orphan policy retained. There were 34 eligible whole sessions, 24 exported candidates, and eight fixed before the eight-GPU outcomes.

The selected requests have shorter outputs and fewer nested-agent calls than the full source. This can alter Decode occupancy and fan-out. C128 uses sixteen cache-isolated copies, so it increases load and memory pressure without repairing this coverage gap. A conclusion about the current layouts on this cohort must not be generalized to all AgentX traffic. Broader claims would need longer-output and stronger-fan-out complete families. No extra GPU cohort is silently added here.

[Source-derived values and provenance](workload_coverage.json)
