# EP / PP deployment: English presentation

Four slides: vertical EP2/PP2 placement; GPU platform; model; Prefill versus Decode.
All slide text, charts, footers and speaker notes are in English.

Decode comparisons include mean TTFT in milliseconds. Prefill comparisons include input
throughput in k tokens/s. The mixed-phase slide shows Prefill input throughput and Decode
TTFT in their respective columns. The existing end-to-end latency and separated Expert,
dispatch/combine and PP communication values are retained.

Eight unique metric cells were recomputed from 24 requests each across three saved rounds.
Prefill throughput is fresh input tokens divided by each cohort's makespan, averaged across
rounds. Decode TTFT starts from the recorded cached prefix. Detailed protocols and provenance
are in the speaker notes, sources.json and request_metric_validation.json.

Rebuild from the workspace experiment root with build_english.py --out <output_directory>.
