"""Exercise three reports with a real complete sample and incomplete decoys."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'runs/eight_gpu_col_ep4_durable_r3'
SCRIPTS = ['analyze_cluster.py', 'analyze_exposure.py', 'analyze_tradeoffs.py']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = SOURCE/'results/col_ep4/agentx_c8/sticky_hash/formal.r1.json'
    initial_sha = sha(original)
    evidence = []
    with tempfile.TemporaryDirectory(prefix='agentx-completion-gate-') as temporary:
        run = Path(temporary)/'run'
        sample = run/'results/col_ep4/agentx_c8/sticky_hash'
        sample.mkdir(parents=True)
        (run/'config.json').write_text((SOURCE/'config.json').read_text())
        (sample/'formal.r1.json').symlink_to(original)
        (sample/'completed.r1.json').write_text(
            original.with_name('completed.r1.json').read_text())
        (sample/'formal.r99.json').write_text('intentionally incomplete JSON')
        (sample/'partial.r2.json').write_text(json.dumps(
            {'complete': False, 'collected_requests': 9999, 'partial_requests': []}))

        def execute(case, expected_success):
            for script in SCRIPTS:
                result = subprocess.run([sys.executable, str(ROOT/script), '--run', str(run)],
                    capture_output=True, text=True)
                assert (result.returncode == 0) == expected_success, result.stderr
                if not expected_success:
                    assert 'AssertionError' in result.stderr
                evidence.append(dict(case=case, script=script, exit_code=result.returncode))

        execute('one complete plus unmarked and partial decoys', True)
        out = run/'analysis'
        assert read(out/'audit.json')['finished_episodes'] == 1
        assert len(read(out/'exposure.json')['trials']) == 1
        assert read(out/'tradeoff_audit.json')['episodes'] == 1
        for name in ['audit.json', 'exposure.json', 'tradeoff_audit.json']:
            assert read(out/name)['excluded_unmarked_formals'] == [str(sample/'formal.r99.json')]
        with (out/'means.csv').open() as f:
            mean = next(csv.DictReader(f))
        for metric in ['ttft_ms', 'tpot_ms', 'e2e_ms']:
            assert abs(float(mean[metric])-read(original)['means'][metric]) < 1e-6

        (sample/'completed.r99.json').write_text(json.dumps({'passed': False}))
        execute('explicit failed completion marker', False)
        (sample/'completed.r99.json').unlink()
        (sample/'completed.r1.json').unlink()
        execute('zero complete episodes clears stale derived CSVs', True)
        assert read(out/'audit.json')['finished_episodes'] == 0
        assert not read(out/'exposure.json')['trials']
        assert read(out/'tradeoff_audit.json')['episodes'] == 0
        assert not list(out.glob('*.csv'))
    assert sha(original) == initial_sha
    result = dict(created_at=time.time(), passed=True, script_executions=evidence,
        source=str(original), source_sha256=initial_sha,
        original_artifact_unchanged=True, retained_metric_values_unchanged=True,
        checks=['unmarked malformed formal excluded', 'partial diagnostic excluded',
                'failed completion marker rejected', 'empty report clears old CSVs'],
        code=[dict(path=str(ROOT/name), sha256=sha(ROOT/name))
              for name in SCRIPTS+['completed_episode_files.py', Path(__file__).name]])
    (ROOT/'completion_gate_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, report_executions=len(evidence), original_unchanged=True)))


if __name__ == '__main__':
    main()
