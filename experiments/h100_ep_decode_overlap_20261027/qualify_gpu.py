"""Qualify the pinned implementation from completed, audited GPU evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TAG = 'experiment/sglang/qwen35b-decode-ep-overlap-20261027-h100-qualified'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    records = [read(ROOT/'analysis'/r.name/'results.json') for r in args.runs]
    assert {r['config']['variant'] for r in records} == {'native','serial','overlap','pp'}
    commits = {r['config']['engine_commit'] for r in records}
    assert len(commits) == 1
    commit = commits.pop()
    source = Path(records[0]['config']['engine_root'])
    implementation = source/'python/sglang/srt/layers/moe/decode_overlap.py'
    evidence = []
    max_l2 = 0.
    layer_records = 0
    profiles = 0
    for record in records:
        assert record['audit']['guard_and_cleanup_verified']
        assert record['audit']['frozen_hashes_verified'] and not record['profiles_skipped']
        run = Path(record['run'])
        cpu = read(run/'decode_overlap_cpu_qualification.json')
        assert cpu['passed'] and cpu['tests'] == 23 and cpu['source_sha256'] == sha(implementation)
        expected = 1 if record['config']['variant'] == 'serial' else 4
        assert len(record['metrics']) == expected
        assert len(record['components']) == expected * 2
        for part in record['components']:
            assert part['selected_graphs'] == 16
            if part['arm'] == 'ep2':
                chunks = 1 if part['variant'] == 'native' else 2
                counts = part['kernel_count_per_layer']
                # Native uses recv-hook send/receive phases (two kernels).
                # Each new chunk uses an async combined send/receive kernel;
                # two chunks therefore also have two kernels, not four.
                assert counts['communication.dispatch'] == 2
                assert counts['communication.combine'] == 2
                assert counts['expert.routed_gemm'] == chunks * 2
                assert counts['expert.layout_adapter'] == chunks
            profiles += 1
        if record['config']['variant'] != 'pp':
            for metric in record['metrics']:
                paths = sorted((run/'ep2/workloads'/metric['workload']/'overlap_qualification').glob('rank*.json'))
                assert len(paths) == 2
                for path in paths:
                    layers = read(path)
                    assert len(layers) == 40
                    for row in layers.values():
                        assert row['passed'] and row['serial_vs_overlap']['exact']
                        max_l2 = max(max_l2,row['native_vs_serial']['relative_l2'])
                        layer_records += 1
        else:
            for metric in record['metrics']:
                folder = run/'pp2/workloads'/metric['workload']
                assert read(folder/'native_transport_warmup.json')['policy'] == 0
                assert read(folder/'output_agreement.json')['all_formal_sequences_equal']
                groups = read(Path(metric['source']))['rounds']
                for previous,current in zip(groups,groups[1:]):
                    a = previous['scheduling_receipt']['states']
                    b = current['scheduling_receipt']['states']
                    assert all(y['ahead_produced'] > x['ahead_produced'] for x,y in zip(a,b))
        path = ROOT/'analysis'/run.name/'results.json'
        evidence.append(dict(run=str(run),analysis=str(path),sha256=sha(path),points=expected))
    assert layer_records == 720 and profiles == 26 and max_l2 <= .015
    command = [sys.executable,'/home/bxb1/.codex/skills/spartan-model-serving/scripts/tag_engine.py',
        '--repo',str(source),'--engine','sglang','--model-slug','qwen36-35b-a3b',
        '--model-revision',records[0]['config']['model_revision'],'--commit',commit,'--tag',TAG,
        '--validation','23 CPU ownership, transport boundary, graph capture and synchronized-rank admission tests.',
        '--validation',f'H100178 NVLink GPU0/1: 13 matched Decode workload/configuration points, five profiler-off cohorts per point; 26 independent rank profiles.',
        '--validation',f'{layer_records} real-weight MoE layer comparisons: native/split max relative L2 {max_l2}; serial/overlap bit exact.',
        '--validation','Actual CUDA Graph replay and per-layer kernel counts verified: native recv-hook two-phase communication versus two async combined chunks; two versus four routed GEMMs. PP policy9 lookahead advanced between formal cohorts.',
        '--validation','Frozen input hashes, canonical CPU weight-cache postchecks, owned-process cleanup and Guard lease release audited.',
        '--validation','Functional experimental qualification only. Two-chunk Decode overlap regresses all four measured workloads; opt-in policy remains default-off. Whole-model TBO OFF; no Prefill DeepGEMM SM reservation.']
    subprocess.run(command,check=True)
    receipt = dict(passed=True,engine_commit=commit,source_tag=TAG,model_revision=records[0]['config']['model_revision'],
        cpu_tests=23,workload_configuration_points=13,formal_cohorts_per_point=5,rank_profiles=profiles,
        real_weight_layer_comparisons=layer_records,native_split_max_relative_l2=max_l2,serial_overlap_bit_exact=True,
        full_model_tbo=False,prefill_sm_reservation=False,default_policy='native',
        communication_modes=dict(native='recv-hook send/receive phases',serial='two async-event combined chunks',overlap='two async-event combined chunks'),
        qualification_scope='Functional experiment qualification on measured inputs and graphs; not a performance improvement claim or a whole-model quality evaluation.',
        evidence=evidence)
    (ROOT/'gpu_qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
