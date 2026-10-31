import unittest
from geometry import counts,grid,model_features


class GeometryTests(unittest.TestCase):
    def test_grid_and_constant_work(self):
        self.assertEqual(len(grid()),32)
        for t,e in grid():
            ns=counts(t,e);self.assertEqual(sum(ns),t)
            active=[n for n in ns if n]
            self.assertEqual(len(active),e)
            self.assertLessEqual(max(active)-min(active),1)

    def test_tile_rounding_and_ep_mapping(self):
        self.assertEqual(model_features(128,8,16)['padded_rows'],128)
        self.assertEqual(model_features(128,24,16)['padded_rows'],384)
        for t in (8,32,128,512):
            for resident in (24,48):self.assertEqual(sum(counts(t,min(t,resident),resident)),t)


if __name__=='__main__':unittest.main()
