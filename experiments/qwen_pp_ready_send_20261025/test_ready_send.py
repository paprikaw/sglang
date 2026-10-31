import contextlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('ready', Path(__file__).with_name('pp_ready_send.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReadySendTests(unittest.TestCase):
    def test_coherent_policy_switch_and_invalid_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'policy'
            r.create_policy(p)
            with r.open_policy(p) as view:
                self.assertEqual(view[0], ord('0'))
                r.set_policy(p, True)
                self.assertEqual(view[0], ord('1'))
                r.set_policy(p, False)
                self.assertEqual(view[0], ord('0'))
            p.chmod(0o644)
            with self.assertRaises(RuntimeError):
                r.open_policy(p)

    def exercise(self, mode, kind='proxy', cuda=True):
        order = []
        class Event:
            def __init__(self, **kwargs): order.append('create')
            def record(self, stream): order.append('record')
            def synchronize(self): order.append('ready')
        scheduler = SimpleNamespace(ps=SimpleNamespace(pp_size=2), device_module=
            SimpleNamespace(Event=Event, current_stream=lambda: 'producer'))
        def original(self, tensors, **kwargs):
            order.append('publish_metadata_and_native_send')
            self.last = (tensors, kwargs)
            return 'native-work'
        fn = r.wrap_send(original, lambda: ord(mode), lambda _: contextlib.nullcontext())
        tensors = {'hidden': SimpleNamespace(is_cuda=cuda)}
        result = fn(scheduler, tensors, async_send=True, msg_type=kind, dst=1)
        self.assertEqual(result, 'native-work')
        self.assertIs(scheduler.last[0], tensors)
        self.assertEqual(scheduler.last[1], dict(async_send=True, msg_type=kind, dst=1))
        return order

    def test_readiness_precedes_metadata_for_both_message_kinds(self):
        for kind in ('proxy', 'output'):
            self.assertEqual(self.exercise('1', kind),
                             ['create', 'record', 'ready', 'publish_metadata_and_native_send'])

    def test_original_path_has_no_synchronization(self):
        self.assertEqual(self.exercise('0'), ['publish_metadata_and_native_send'])

    def test_cpu_messages_are_unchanged(self):
        self.assertEqual(self.exercise('1', cuda=False), ['publish_metadata_and_native_send'])

    def test_unknown_policy_fails(self):
        with self.assertRaises(RuntimeError):
            self.exercise('x')


if __name__ == '__main__':
    unittest.main()
