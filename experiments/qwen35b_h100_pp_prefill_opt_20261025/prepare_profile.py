"""Freeze a separate profile-only run, without changing any completed run."""
import argparse,hashlib,json,shutil
from pathlib import Path
import prepare_ready


def build(run):
    prepare_ready.build(run);run=run.resolve()
    cfg=json.loads((run/'config.json').read_text());spec=json.loads((run/'dashboard_spec.json').read_text())
    cfg.update(port=21331,dist_port=21341,expected_points=2,protocol='Independent profile-only PP2 producer-ready A/B. Both GPU trace exports must pass gzip CRC and JSON before cleanup. No throughput result. Same qualified full CPU weight bank and real GPU cached prefixes.')
    shutil.copyfile(Path(__file__).with_name('profile_ready.py'),run/'code/optimization.py')
    runner=run/'code/runner.py';source=runner.read_text()
    needle="valid_points=len(list(run.glob('*/metrics.*.json')))"
    assert source.count(needle)==1
    runner.write_text(source.replace(needle,"valid_points=len(list(run.glob('pp2/traces/*/manifest.json')))"))
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    def desc(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs=[x for x in spec['frozen_inputs'] if not Path(x['path']).is_relative_to(run/'code') and x['path']!=str(run/'config.json')]
    spec.update(port=21330,title='H100 PP2 Prefill: joined two-stage diagnostic',frozen_inputs=inputs+[desc(p) for p in (run/'code').glob('*.py')]+[desc(run/'config.json')])
    spec['tasks'][0].update(expected=2,label='Independent PP2 A/B traces; both stage exports joined')
    (run/'dashboard_spec.json').write_text(json.dumps(spec,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
