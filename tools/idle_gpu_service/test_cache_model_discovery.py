"""CPU-only regressions for changed-checkpoint recovery and live cache identity."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cache_model_discovery as d
import cluster_backfill as c
from manager import Booking

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'experiments/cluster_backfill_20261020/node_runner_cache_model_ephemeral.py'


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(architectures=['Glm4MoeForCausalLM'], model_type='glm4_moe',
                           hidden_size=5120, quantization_config={'quant_method': 'awq'},
                           name_or_path='old/checkpoint')
        self.old = self.model('historical-name', self.config)
        new_config = dict(self.config, name_or_path='entirely-new/checkpoint')
        self.new = self.model('newly-added-checkpoint', new_config)
        self.bank = dict(daemon_id='current-daemon', expected_job_id='1', source_commit='source',
                         state='active', cache_scheme='canonical_full_weight')
        self.recipe = dict(id='qualified-loader', engine_commit='source', fallback=True,
            node_template=dict(model_path=str(self.old), support_profile='glm4moe-canonical-v1'),
            node_cache_keys={'node': ['historical-key']},
            checkpoint_qualification=dict(layout_sha256=d.layout_sha(self.config), max_bank_bytes=100,
                node_runner=str(RUNNER), node_runner_sha256=c.digest(RUNNER)))

    def model(self, name, config):
        path = self.root / name
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'model.safetensors.index.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(b'different weights by checkpoint: ' + name.encode())
        return path

    def describe(self, path):
        # Full identity includes path and stat, as the real canonical provider
        # does; equal byte counts or the same architecture alone cannot match.
        path = Path(path)
        shard = path / 'model.safetensors'
        key = d.json_sha([str(path), shard.stat().st_ino, shard.stat().st_mtime_ns])
        return dict(model_root=str(path), group_key=key, items=[dict(key=key)],
            shards=[dict(size=shard.stat().st_size)], shard_count=1,
            total_bytes=shard.stat().st_size,
            fingerprint=dict(metadata_sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in ('config.json', 'model.safetensors.index.json')}))

    def observation(self):
        desc = self.describe(self.new)
        status = dict(entries=[dict(key=desc['group_key'], size=desc['total_bytes'], state='ready')])
        models, _ = d.match_models(d.checkpoint_paths([self.root]), status, self.describe)
        return dict(status=status, models=models)

    def test_new_checkpoint_name_and_path_are_discovered_without_new_recipe(self):
        observation = self.observation()
        self.assertEqual([m['model_root'] for m in observation['models']], [str(self.new)])
        recipes, reason = d.current_recipes('node', self.bank, observation, [self.recipe])
        self.assertEqual(reason, 'reuse_current_model')
        self.assertEqual(recipes[0]['node_template']['model_path'], str(self.new))
        self.assertEqual(recipes[0]['node_cache_keys']['node'], [self.describe(self.new)['group_key']])
        self.assertFalse(recipes[0]['fallback'])
        self.assertEqual(self.recipe['node_template']['model_path'], str(self.old))

    def test_incomplete_bank_and_same_size_wrong_keys_never_match(self):
        for entries in ([], [dict(key='other-model', size=self.describe(self.new)['total_bytes'])]):
            self.assertEqual(d.match_models([self.new], dict(entries=entries), self.describe)[0], [])

    def test_replaced_daemon_gets_new_plan_even_for_the_same_checkpoint(self):
        observation = self.observation()
        old = d.current_recipes('node', self.bank, observation, [self.recipe])[0][0]
        new = d.current_recipes('node', dict(self.bank, daemon_id='replacement'), observation, [self.recipe])[0][0]
        self.assertNotEqual(old['id'], new['id'])
        self.assertEqual(new['discovered_model']['daemon_id'], 'replacement')

    def test_changed_layout_quantization_or_capacity_waits_without_old_model_fallback(self):
        observation = self.observation()
        for change in ('quantization', 'shape', 'capacity', 'source'):
            recipe = copy.deepcopy(self.recipe)
            if change == 'source':
                recipe['engine_commit'] = 'different-engine'
            elif change == 'capacity':
                recipe['checkpoint_qualification']['max_bank_bytes'] = 1
            else:
                cfg = copy.deepcopy(self.config)
                if change == 'shape': cfg['hidden_size'] *= 2
                else: cfg['quantization_config']['quant_method'] = 'gptq'
                recipe['checkpoint_qualification']['layout_sha256'] = d.layout_sha(cfg)
            candidates, reason = d.current_recipes('node', self.bank, observation, [recipe])
            self.assertEqual(candidates, [])
            self.assertEqual(reason, 'waiting_for_model_loader')

    def test_unsupported_current_hint_never_uses_an_older_complete_bank(self):
        observation = self.observation()
        old = dict(observation['models'][0], model_root=str(self.old))
        observation['models'][0]['layout_sha256'] = 'new-unsupported-layout'
        observation['models'].append(old)
        observation['preferred_model_root'] = str(self.new)
        recipes, reason = d.current_recipes('node', self.bank, observation, [self.recipe])
        self.assertEqual(recipes, [])
        self.assertEqual(reason, 'waiting_for_model_loader')

    def test_broken_download_does_not_hide_complete_current_checkpoint(self):
        broken = self.root / 'broken-download'
        broken.mkdir()
        (broken / 'config.json').write_text('{')
        desc = self.describe(self.new)
        models, failures = d.match_models([broken, self.new],
            dict(entries=[dict(key=desc['group_key'], size=desc['total_bytes'])]), self.describe)
        self.assertEqual(len(models), 1)
        self.assertEqual(failures, 1)

    def test_snapshot_layout_symlink_and_directory_cycle_are_bounded(self):
        hub = self.root / 'hub'
        snapshot = hub / 'models--new--name/snapshots/revision'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        (hub / 'cycle').symlink_to(hub, target_is_directory=True)
        self.assertEqual(list(d.checkpoint_paths([hub])), [snapshot])
        with self.assertRaisesRegex(ValueError, 'bound'):
            list(d.checkpoint_paths([self.root], max_directories=1))

    def test_current_plan_updates_serving_path_revision_and_node_spec_together(self):
        recipe = d.current_recipes('node', self.bank, self.observation(), [self.recipe])[0][0]
        workload = self.root / 'benchmark.json'
        workload.write_text('{}')
        profile = self.root / 'template.json'
        profile.write_text(json.dumps(dict(model_path=str(self.old), model_revision='old',
                                           benchmark=dict(config_path=str(workload)))))
        recipe.update(profile=str(profile), profile_sha256=c.digest(profile))
        bank = dict(self.bank, max_bytes=1000, headroom_bytes=32 * 2**30, socket_path='/tmp/cache.sock')
        result = c.make_profile(dict(node_runner='/old-runner'), recipe,
            Booking('node', '1', time.time() + 3600, 'gpu:A100:4'), bank, self.root / 'service')
        self.assertEqual(result['model_path'], str(self.new))
        self.assertEqual(result['known_node_specs']['node']['cache_spec_template']['model_path'], str(self.new))
        self.assertNotEqual(result['model_revision'], 'old')
        self.assertEqual(result['node_runner'], str(RUNNER))
        self.assertTrue(result['cached_weights_only'])
        self.assertEqual(result['model_cache_on_miss'], 'fail')
        self.assertEqual(json.loads(profile.read_text())['model_path'], str(self.old))

    def test_unknown_current_weights_never_fall_through_to_historical_fallback(self):
        book = Booking('node', '1', time.time() + 3600, 'gpu:A100:4')
        bank = dict(self.bank, node='node')
        observation = self.observation()
        observation['models'][0]['layout_sha256'] = 'unqualified-layout'
        with patch.object(c, 'discover_bookings', return_value={'node': book}), \
             patch.object(c, 'registry_blocks', return_value={}), \
             patch.object(c, 'read_json', side_effect=[dict(leases=[]), dict(entries=[bank])]), \
             patch.object(d, 'discover', return_value=observation), \
             patch.object(c, 'discover_recipe_keys') as legacy, \
             patch.object(c.subprocess, 'Popen') as spawn:
            state = c.reconcile(dict(gpu_registry_dir='/tmp/registry', model_discovery={'enabled': True}),
                                self.root, {}, [self.recipe])
        self.assertEqual(state['nodes']['node']['reason'], 'waiting_for_model_loader')
        legacy.assert_not_called()
        spawn.assert_not_called()

    def test_discovery_source_pin_checked_before_ssh(self):
        with patch.object(d.subprocess, 'run') as ssh:
            with self.assertRaisesRegex(ValueError, 'source changed'):
                d.discover('node', self.bank, [self.recipe], dict(script_sha256='wrong'))
            ssh.assert_not_called()

    def test_generic_runner_rejects_changed_checkpoint_metadata(self):
        spec = importlib.util.spec_from_file_location('cache_model_runner_test', RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        model = self.observation()['models'][0]
        cfg = dict(model_path=str(self.new), discovered_model=model,
                   cache_support_profile='glm4moe-canonical-v1')
        runner.validate_discovered_model(cfg)
        (self.new / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            runner.validate_discovered_model(cfg)

    def hint_spec(self):
        node_spec = self.root / 'frozen-node-spec.json'
        node_spec.write_text(json.dumps(dict(node='node', expected_job_id='1', source_commit='source',
            model_path=str(self.new), cache_scheme='canonical_full_weight')))
        node_spec.chmod(0o600)
        return dict(model_cache=dict(policy='required', provider='sglang_canonical_full_weight_v1',
            node_specs={'node': dict(path=str(node_spec), sha256=c.digest(node_spec))}))

    def test_guard_hints_cover_paths_outside_catalog_without_per_call_files(self):
        spec = self.hint_spec()
        for _ in range(3):
            self.assertEqual(d.record_hints(spec, self.root / 'runtime'), 1)
        self.assertEqual(len(list((self.root / 'runtime/model_hints').iterdir())), 1)
        self.assertEqual(d.model_hint('node', self.bank, self.root / 'runtime'), str(self.new))
        source_sha = c.digest(Path(d.__file__))
        response = dict(daemon_id='current-daemon', expected_job_id='1', source_commit='source', models=[])
        recipe = copy.deepcopy(self.recipe)
        recipe['node_template']['engine_root'] = '/qualified/engine'
        with patch.object(d.subprocess, 'run', return_value=type('Result', (), {'stdout': json.dumps(response)})()) as ssh:
            d.discover('node', self.bank, [recipe], dict(roots=['/unrelated/catalog'], script_sha256=source_sha),
                       self.root / 'runtime')
        self.assertEqual(json.loads(ssh.call_args.kwargs['input'])['extra_paths'], [str(self.new)])

    def test_guard_after_run_hook_records_current_paths_without_starting_a_competitor(self):
        source = Path('/home/bxb1/experiment-guard/skills/llm-experiment-guard/scripts/backfill_trigger.py')
        spec = importlib.util.spec_from_file_location('guard_cache_hint_hook_test', source)
        trigger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trigger)
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'state.json').write_text(json.dumps(dict(manager_host='other-host', timestamp=time.time())))
        caller = self.root / 'foreground.json'
        caller.write_text(json.dumps(self.hint_spec()))
        cfg = dict(enabled=True, runtime_root=str(runtime), gpu_registry_dir='/tmp/registry',
            watchdog_script=str(RUNNER), watchdog_sha256=c.digest(RUNNER),
            model_discovery=dict(script_path=str(Path(d.__file__).resolve()),
                                 script_sha256=c.digest(Path(d.__file__))))
        config_path = self.root / 'hook-config.json'
        config_path.write_text(json.dumps(cfg))
        config_path.chmod(0o600)
        with patch.object(trigger, 'CONFIG', config_path), \
             patch.dict(os.environ, {'EXPERIMENT_GUARD_BACKFILL_CHILD': '0'}), \
             patch.object(trigger.subprocess, 'Popen') as spawn, \
             patch.object(trigger.subprocess, 'run') as service:
            result = trigger.notify(caller, 'after_run')
        self.assertEqual(result['supervisor'], 'live')
        self.assertEqual(d.model_hint('node', self.bank, runtime), str(self.new))
        spawn.assert_not_called()
        service.assert_not_called()

    def test_changed_frozen_spec_and_other_job_hints_are_ignored(self):
        spec = self.hint_spec()
        spec['model_cache']['node_specs']['node']['sha256'] = 'wrong'
        self.assertEqual(d.record_hints(spec, self.root / 'runtime'), 0)
        self.assertFalse((self.root / 'runtime/model_hints').exists())
        self.assertEqual(d.record_hints(self.hint_spec(), self.root / 'runtime'), 1)
        self.assertIsNone(d.model_hint('node', dict(self.bank, expected_job_id='different'), self.root / 'runtime'))

    def test_stale_daemon_identity_rejects_before_any_cache_rpc(self):
        bank = dict(self.bank, uid=os.getuid(), manifest=dict(node='node', cache_id='current-daemon',
            pid=123, process_start_ticks=100), socket_path='/tmp/cache.sock')
        with patch.object(d.socket, 'gethostname', return_value='node'), \
             patch.object(Path, 'read_text', return_value='123 (daemon) S ' + ' '.join(['0'] * 30)), \
             patch.object(d, 'verified_status', wraps=d.verified_status):
            client = unittest.mock.Mock()
            with self.assertRaisesRegex(ValueError, 'process identity changed'):
                d.verified_status(bank, 'node', client)
            client.assert_not_called()

    def test_fp8_entry_and_canonical_profile_follow_the_qualified_template(self):
        spec = importlib.util.spec_from_file_location('cache_model_runner_entry_test', RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(runner.server_entry_args({}), ['-m', 'sglang.launch_server'])
        self.assertEqual(runner.server_entry_args(dict(server_entry='/qualified/fp8_entry.py')),
                         ['/qualified/fp8_entry.py'])


if __name__ == '__main__':
    unittest.main()
