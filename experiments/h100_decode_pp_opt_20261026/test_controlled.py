import unittest
from controlled import admission_receipt


class AdmissionReceiptTest(unittest.TestCase):
    def info(self):
        return dict(internal_states=[dict(experiment_fixed_group_receipt=dict(epoch=3,
            ordered_markers=[5000,5002,5004,5006,5001,5003,5005,5007],waiting_ms=.2))])

    def test_native_head_only_response(self):
        self.assertTrue(admission_receipt(self.info(),3)['head_acknowledged'])

    def test_stale(self):
        with self.assertRaises(RuntimeError):admission_receipt(self.info(),4)

    def test_missing(self):
        with self.assertRaises(RuntimeError):admission_receipt(dict(internal_states=[{}]),3)

    def test_wrong_order(self):
        info=self.info();info['internal_states'][0]['experiment_fixed_group_receipt']['ordered_markers'].reverse()
        with self.assertRaises(RuntimeError):admission_receipt(info,3)


if __name__=='__main__':unittest.main()
