"""Archive only public CPU-sized artifacts; never copy runtime credentials or banks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['independent_validation_passed']
    assert manifest['id'] == 'single_layer_ep4_ep8_ce_rdma_20261024_r1'
    target = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / manifest['id']
    target.mkdir(exist_ok=True)
    sources = {name: run / name for name in manifest['artifacts']}
    for name in ('report_transport.py', 'plot_transport.py', 'verify_transport.py', 'archive_transport.py'):
        sources['analysis/' + name] = run.parent / name
    sources['build_result.json'] = run / 'candidate/build_result.json'
    for path in (run / 'frozen/harness').glob('*'):
        if path.is_file() and path.suffix in ('.py', '.cu'):
            sources['harness/' + path.name] = path
    for name, source in sources.items():
        destination = target / name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            assert sha(destination) == sha(source), f'Existing artifact differs: {destination}'
        else:
            shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
    archive = dict(manifest)
    archive['archive_root'] = str(target)
    archive['archive_artifacts'] = list(sources)
    archive['source_artifacts'] = {name: {'path': str(source), 'sha256': sha(source)} for name, source in sources.items()}
    archive['raw_rank_results'] = [
        {'path': str(source), 'sha256': sha(source)}
        for source in sorted(run.glob('*/result_rank*.json'))
    ]
    assert len(archive['raw_rank_results']) == 64
    (target / 'manifest.json').write_text(json.dumps(archive, indent=2) + '\n')
    checksums = {str(path.relative_to(target)): sha(path) for path in sorted(target.rglob('*'))
                 if path.is_file() and path.name != 'artifact_checksums.json'}
    (target / 'artifact_checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    assert all(sha(target / name) == digest for name, digest in checksums.items())
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker = f'<!-- begin {manifest["id"]} -->'
    entry = f'''{marker}
### {manifest['id']}

Snapshot: {target}

- Complete communication-only CE/RDMA comparison on booked A100 nodes 065/167, both verified ConnectX-6 Dx; no model weights, Attention, Expert GEMM or KV bank loaded. GLM geometry BF16 hidden5120/160experts/top8 and Hy3 geometry BF16 hidden4096/192experts/top8.
- Unique whole-batch tokens8 through4096 by doubling. Two boundaries: EP8 whole versus EP4 half per call (equal assignments/GPU), and equal unique tokens/call (EP4 has twice the assignments/GPU). EP4 measured independently on each node. Eager, capacity1024/rank, warmup3/formal3; aligned rank max before median. Both backends retain NVLink inside pairs; CE uses PCIe cross-pair locally and RDMA remotely.
- Node-local EP4 large messages benefit from CE: whole4096/half2048 on065 GLM17.7161 to2.4376ms, Hy3 14.3401 to2.0006ms (about86% lower). Small messages favor RDMA. Cross-node EP8 differs: GLM has no material CE win; Hy3 CE is4.4-18.8% lower at256-2048 tokens but23.9% slower at4096. No universal Prefill/Decode switch rule or full-PP speedup claim.
- Same native within each model's CE/RDMA arms. Hy3 changed exactly five CE hidden-size literals only; host/other CUDA objects/SM/notification retained. Full payload/count/identity/weighted-expert/skew validation and actual CE counters pass. Independently recomputed280 points/840 aligned samples from64 rank files; frozen65 input hashes,16 cleanup proofs and lease release pass. Three samples do not establish service tails or precise crossovers.

Artifacts: complete Markdown tables,80-row comparisons,280-point phase metrics,840 aligned samples,PNG/SVG/PDF curves,configuration,frozen harness,analysis scripts,independent validation and checksums. Raw rank results are referenced with hashes; private Guard authorization, banks, natives and large logs are not copied.

<!-- end {manifest['id']} -->

'''
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        if marker not in text:
            heading = '## Saved Experiments\n\n'
            assert heading in text
            text = text.replace(heading, heading + entry, 1)
            handle.seek(0)
            handle.write(text)
            handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
    assert marker in registry.read_text()
    receipt = dict(passed=True, archive=str(target), files=len(checksums),
                   raw_rank_references=64, registry_updated=True)
    (run / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
