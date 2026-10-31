"""Check whether EP probability differences also occur in the native control."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def verify(native,optimized):
    path=ROOT.parent/'h100_pp_expert_fragmentation_20261026/validate_probabilities.py'
    spec=importlib.util.spec_from_file_location('ep_probability_control',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=ROOT/'runs'/native/'ep2/validation.probabilities.json'
    reference=m.samples(source)
    candidate=m.samples(ROOT/'runs'/optimized/'ep2/validation.probabilities.json')
    self_check,_=m.compare(reference[0],reference[-1],'native first versus native last')
    matched=[m.compare(reference[-1],sample,'optimized versus native last '+str(i))[0]
             for i,sample in enumerate(candidate)]
    result=dict(native_reference=str(source),native_self_variation=self_check,
        optimized_vs_last_native=matched,
        explanation='First-to-last native EP logprob variation is also present without the expert optimization; no cause assigned from these probes.')
    (ROOT/'analysis/probability_native_last_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    selection=json.loads((ROOT/'deliverables/selection.json').read_text())
    print(json.dumps(verify(selection['native'],selection['optimized']),indent=2))
