import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import humming_receipt as receipt


def kernel(kind):
    return SimpleNamespace(
        gemm_type=SimpleNamespace(value=kind), sm_version=80,
        use_batch_invariant=True, use_stream_k=False,
        block_shape=(64, 128, 32), warp_shape=(32, 32, 32),
        kernel_filename='/cache/example.cubin', kernel_name='humming', code='code',
    )


class ReceiptTests(unittest.TestCase):
    def collect(self):
        module = SimpleNamespace(HummingKernel=SimpleNamespace(
            _id2kernel={1: kernel('dense'), 2: kernel('grouped_masked')}))
        with patch.dict(sys.modules, {'humming.kernel.humming': module}):
            return receipt.collect_humming_kernel_receipt()

    def test_actual_registry_proof_is_serializable_and_explicitly_not_launch_proof(self):
        import json
        value = self.collect()
        json.dumps(value)
        self.assertTrue(receipt.validate_humming_kernel_receipt(value)['passed'])
        self.assertFalse(value['proves_actual_launch'])

    def test_missing_type_and_each_noninvariant_kernel_fail_closed(self):
        value = self.collect()
        value['kernels'].pop()
        with self.assertRaisesRegex(ValueError, 'absent'):
            receipt.validate_humming_kernel_receipt(value)
        for field, bad in [('use_batch_invariant', False), ('use_stream_k', True),
                           ('sm_version', 90), ('source_sha256', None),
                           ('block_shape', []), ('kernel_filename', None)]:
            value = self.collect()
            value['kernels'][1][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                receipt.validate_humming_kernel_receipt(value)

    def test_one_safe_kernel_does_not_hide_another_unsafe_candidate(self):
        value = self.collect()
        other = copy.deepcopy(value['kernels'][0])
        other['kernel_id'] = 3
        other['use_stream_k'] = True
        value['kernels'].append(other)
        with self.assertRaisesRegex(ValueError, 'StreamK'):
            receipt.validate_humming_kernel_receipt(value)

    def test_uninitialized_registry_and_unknown_boolean_fail_closed(self):
        with patch.dict(sys.modules, {'humming.kernel.humming': None}):
            with self.assertRaisesRegex(RuntimeError, 'initialized'):
                receipt.collect_humming_kernel_receipt()
        value = self.collect()
        value['kernels'][0]['use_stream_k'] = 0
        with self.assertRaisesRegex(ValueError, 'StreamK'):
            receipt.validate_humming_kernel_receipt(value)


if __name__ == '__main__':
    unittest.main()
