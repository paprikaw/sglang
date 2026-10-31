"""CPU-only contract test: ordinary one-node demand drains a two-node background group."""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/hy3_idle_runtime_20261019/guard_candidate/scripts'))
from gpu_registry import SharedGpuRegistry,RegistryConflict

class GroupPreemption(unittest.TestCase):
    def test_default_claim_preempts_group_and_group_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=SharedGpuRegistry(root/'registry')
            bg=dict(id='background',port=18001,resource_priority='background',preemptible=True,preemption_capable=True,tasks=[dict(id='model',nodes=['n1','n2'])])
            receipt,lease=registry.claim(bg,'bgsha',['n1','n2'],root/'bg.json')
            normal=dict(id='foreground',port=18002,tasks=[dict(id='experiment',nodes=['n1'])]);result=[];errors=[]
            def claim():
                try:result.append(registry.claim(normal,'fgsha',['n1','n2'],root/'fg.json'))
                except Exception as exc:errors.append(exc)
            thread=threading.Thread(target=claim);thread.start()
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    owner=registry.assert_owned(receipt,'background','bgsha')
                    if owner.get('preemption_requested'):break
                    time.sleep(.01)
                else:self.fail('ordinary foreground claim did not request preemption')
                self.assertEqual(owner['nodes'],['n1','n2'])
                registry.release(receipt,True,'simulated owned-group cleanup')
                thread.join(5);self.assertFalse(thread.is_alive());self.assertFalse(errors)
                fg,_=result[0]
                with self.assertRaises(RegistryConflict):registry.claim(bg,'bgsha',['n1','n2'],root/'bg2.json')
                registry.release(fg,True,'simulated foreground cleanup')
                resumed,newlease=registry.claim(bg,'bgsha',['n1','n2'],root/'bg3.json')
                self.assertEqual(newlease['nodes'],['n1','n2']);registry.release(resumed,True,'test complete')
            finally:thread.join(5)
if __name__=='__main__':unittest.main()
