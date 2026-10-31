import gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from profile_ready import await_exports

class ExportTests(unittest.TestCase):
 def test_both_complete(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)
   for rank in (0,1):(root/f'PP-{rank}.trace.json.gz').write_bytes(gzip.compress(json.dumps({'traceEvents':[{'name':'test'}]}).encode()))
   files=await_exports(root,lambda:None,timeout=0)
   self.assertEqual(len(files),2)
   self.assertTrue(all(len(x['sha256'])==64 for x in files))
 def test_partial_export_is_retried(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);raw=gzip.compress(json.dumps({'traceEvents':[{'name':'test'}]}).encode())
   (root/'PP-0.trace.json.gz').write_bytes(raw);second=root/'PP-1.trace.json.gz';second.write_bytes(raw[:-8])
   with patch('profile_ready.time.sleep',lambda _:second.write_bytes(raw)):
    files=await_exports(root,lambda:None,timeout=2)
   self.assertEqual(len(files),2)
 def test_missing_rank_fails(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaises(TimeoutError):await_exports(Path(temp),lambda:None,timeout=0)
