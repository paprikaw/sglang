import unittest
from pp_transfer_audit import compare_transfers


def receipt(rank, direction, kind='proxy', digest='abc', epoch='1'):
    return dict(pp_rank=rank, epoch=epoch, transfers=[dict(direction=direction,
        kind=kind, sequence=0, tensors={'x': dict(dtype='bfloat16', shape=[4, 8], sha256=digest)})])


class TransferComparisonTest(unittest.TestCase):
    def test_match(self):
        result=compare_transfers([receipt(0,'send'), receipt(1,'recv')])
        self.assertTrue(result['passed']);self.assertEqual(result['matched_messages'],1)

    def test_corruption(self):
        self.assertFalse(compare_transfers([receipt(0,'send'),receipt(1,'recv',digest='bad')])['passed'])

    def test_missing_message(self):
        right=receipt(1,'recv');right['transfers']=[]
        self.assertFalse(compare_transfers([receipt(0,'send'),right])['passed'])

    def test_wrong_kind(self):
        self.assertFalse(compare_transfers([receipt(0,'send'),receipt(1,'recv',kind='output')])['passed'])

    def test_epoch_mismatch(self):
        with self.assertRaises(ValueError):
            compare_transfers([receipt(0,'send'),receipt(1,'recv',epoch='2')])

    def test_order_mismatch(self):
        left,right=receipt(0,'send'),receipt(1,'recv')
        right['transfers'][0]['sequence']=1
        self.assertFalse(compare_transfers([left,right])['passed'])


if __name__ == '__main__': unittest.main()
