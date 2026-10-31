# EP / PP deployment: Decode throughput revision

The second slide now compares mean full-request E2E latency and Decode output throughput,
replacing its TTFT charts. Throughput is output tokens divided by cohort makespan, averaged
over three saved rounds, with 8 requests and 128 output tokens per request.

A100: EP2 72.1 versus PP2 97.3 output tokens/s. H100: EP2 636.6 versus PP2 509.8 output tokens/s.
The other three slide previews and both original numeric tables are preserved. All four
slides remain in English. Raw request records were checked again; no GPU experiment ran.

Rebuild from the workspace experiment root: build_english.py --gpu-throughput.
