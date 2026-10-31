"""Update the existing small result snapshot without copying runtime secrets."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / HERE.name
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((HERE / 'analysis/fixed_groups/summary.json').read_text())
    if not summary['pp_output_equality_passed'] or summary['matched_messages'] != 3072:
        raise RuntimeError('Final scoped qualification missing')
    files = sorted(path.relative_to(HERE).as_posix() for path in HERE.glob('*.py'))
    files += [
        'REPORT.md', 'FIXED_GROUPS_REPORT.md', 'manifest.json',
        'analysis/audit.json', 'analysis/metrics.csv', 'analysis/profiles.csv',
        'analysis/35b_transfer_qualification.json',
        'analysis/fixed_groups/audit.json', 'analysis/fixed_groups/metrics.csv',
        'analysis/fixed_groups/profiles.csv', 'analysis/fixed_groups/summary.json',
        'analysis/fixed_groups/scheduling_absolute.png',
        'analysis/fixed_groups/scheduling_absolute.svg',
    ]
    files = sorted(files)
    for relative in files:
        src = HERE / relative
        dst = SNAPSHOT / relative
        if not src.is_file() or src.stat().st_size > 4 * 1024 * 1024:
            raise RuntimeError(f'Not a small source/report artifact: {relative}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(src) != digest(dst):
            raise RuntimeError(f'Snapshot differs: {relative}')
    checksums = ''.join(f'{digest(HERE / relative)}  {relative}\n' for relative in files)
    (HERE / 'checksums.sha256').write_text(checksums)
    (SNAPSHOT / 'checksums.sha256').write_text(checksums)
    index = ROOT / 'experiments/experiment_summary/component_reports/Q09.md'
    shutil.copy2(index, SNAPSHOT / 'index_Q09.md')
    (SNAPSHOT / 'summary.md').write_text(
        '# H100 Decode PP scheduling optimization\n\n'
        'Final fixed-group confirmation: same H100178 NVLink GPUs0/1, '
        '35B BF16 / 122B FP8; EP2 versus two-slot PP2. '
        'Eight requests, each real GPUcached4096 + new1 -> output128; '
        'three unprofiled rounds per mode. PP early send improves throughput '
        '3.98% / 1.35%, but EP2 remains faster. All six PP output cohorts '
        'are identical per model; 3072 transmitted messages match; '
        'cleanup and Guard postchecks passed. Candidate is scoped/private, '
        'not a shared engine default.\n\n'
        '[Final report, all absolute components](FIXED_GROUPS_REPORT.md) · '
        '[Historical pilot, retained separately](REPORT.md) · '
        '[Index entry](index_Q09.md).\n\n'
        'Only small scripts, tests, reports, CSV/JSON summaries and figures '
        'are copied. No weights, KV payloads, runtime keys, raw logs or traces. '
        'Original paths and hashes are retained in analysis/audit files.\n')
    entry = '''## h100_decode_pp_opt_20261026

- Status: final fixed-group confirmation complete; candidate validated only for the tested models/cache/B8/two-slot scope; shared engine unchanged, not a production default.
- Same H100178 NVLink GPUs0/1; Qwen3.6-35B-A3B BF16 / Qwen3.5-122B-A10B FP8. EP2 native reference versus PP2 original/early-send, Attention TP1; real GPUcached4096/new1/output128/B8, two fixed PP slots of four requests. Complete CPU weight banks reused.
- Three unprofiled rounds per mode: PP throughput 685.379 -> 712.640 (+3.98%) / 521.764 -> 528.832 (+1.35%) output tok/s; TPOT 11.069 -> 10.628 / 14.370 -> 14.177 ms. EP2 TPOT 7.946 / 11.138 ms. Absolute PP-minus-EP gap 2.683 / 3.040 ms; relative gap 33.77% / 27.29%, not a size-only universal trend.
- Fixed-group PP A/A and A/B generated tokens all equal; 3072 stage messages byte-hash/shape/dtype/order matched, complete stage batch sequences matched. Original native random-group output variability and failed/interrupted attempts retained. All 21 CPU tests and final Guard/cache/cleanup checks passed, GPUs released.
- Independent actual-Graph profiles: first-stage routed GEMM 2.076 -> 1.731 / 2.733 -> 2.111 ms, with NCCL residency overlap 1.800 / 2.255 -> 0 ms. Tail unchanged; Graph-external dependencies remain. Interval overlap is not SM-counter evidence or additive request latency.
- Historical four-policy pilot and +3.61%/+1.85% random-group cohort retained separately, not merged into the fixed-group results. Six final metric rows, twelve rank profiles, absolute figures and source hashes archived.
- Snapshot: `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_decode_pp_opt_20261026`
- Final report: `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_decode_pp_opt_20261026/FIXED_GROUPS_REPORT.md`
- Project entry: `/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q09.md`
'''
    original = REGISTRY.read_text()
    pattern = r'(?ms)^## h100_decode_pp_opt_20261026\n.*?(?=^## |\Z)'
    if len(re.findall(pattern, original)) != 1:
        raise RuntimeError('Expected one existing registry entry')
    updated = re.sub(pattern, lambda _: entry + '\n', original)
    if updated != original:
        REGISTRY.write_text(updated)
    # Verify the registry replacement itself is idempotent.
    if re.sub(pattern, lambda _: entry + '\n', updated) != updated:
        raise RuntimeError('Registry rewrite is not idempotent')
    result = dict(files=len(files), snapshot=str(SNAPSHOT), byte_verified=True,
                  registry_idempotent=True, weight_kv_or_keys_copied=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
