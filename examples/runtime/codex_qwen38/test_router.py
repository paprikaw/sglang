import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zstandard

os.environ.setdefault("QWEN_LOCAL_API_KEY", "test-local-key")

import router
import rollback_desktop


class RouterTests(unittest.TestCase):
    def test_rollback_preserves_unrelated_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'config.toml'
            content = ''.join(f'{key} = {json.dumps(value)}\n' for key, value in rollback_desktop.EXPECTED.items())
            config.write_text(content + 'model = "keep-this"\n[features]\nexample = true\n')
            with patch.object(rollback_desktop, 'CONFIG', config):
                rollback_desktop.main()
            self.assertEqual(config.read_text(), 'model = "keep-this"\n[features]\nexample = true\n')

    def test_auth_rejects_foreign_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            auth = Path(temporary) / "auth.json"
            auth.write_text(json.dumps({"tokens": {"access_token": "test-owner"}}))
            with patch.object(router, "AUTH_PATH", auth):
                self.assertTrue(router.owned_token({"authorization": "Bearer test-owner"}))
                self.assertFalse(router.owned_token({"authorization": "Bearer another-user"}))
                self.assertFalse(router.owned_token({}))

    def test_zstd_body_and_fixed_official_routes(self):
        body = {"model": "test", "input": "hello"}
        compressed = zstandard.ZstdCompressor().compress(json.dumps(body).encode())
        self.assertEqual(router.parse_body(compressed, {"content-encoding": "zstd"}), body)
        self.assertEqual(router.upstream({"authorization": "Bearer sk-test"}), "https://api.openai.com/v1")
        self.assertEqual(router.upstream({"authorization": "Bearer test-owner"}), "https://chatgpt.com/backend-api/codex")

    def test_namespace_and_custom_tool_roundtrip(self):
        adapter = router.ToolAdapter({"tools": [{"type": "namespace", "name": "functions", "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
            {"type": "function", "name": "shell_command", "parameters": {"type": "object"}},
        ]}], "input": [
            {"type": "custom_tool_call", "name": "apply_patch", "namespace": "functions", "call_id": "c1", "input": "patch"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
        ]})
        self.assertEqual(adapter.body["input"][0]["name"], "codex_tool_0")
        self.assertEqual(json.loads(adapter.body["input"][0]["arguments"]), {"input": "patch"})
        self.assertEqual(adapter.body["input"][1]["type"], "function_call_output")
        wire = {"type": "function_call", "id": "i1", "name": "codex_tool_0", "call_id": "c1", "arguments": ""}
        events = adapter.events({"type": "response.output_item.added", "output_index": 0, "item": wire})
        self.assertEqual(events[0]["item"]["type"], "custom_tool_call")
        self.assertEqual(events[0]["item"]["namespace"], "functions")
        self.assertEqual(adapter.events({"type": "response.function_call_arguments.delta", "item_id": "i1", "output_index": 0, "delta": '{"input":'}), [])
        arguments = json.dumps({"input": "*** Begin Patch\n*** End Patch"})
        events = adapter.events({"type": "response.function_call_arguments.done", "item_id": "i1", "output_index": 0, "arguments": arguments})
        self.assertEqual(events[0]["delta"], "*** Begin Patch\n*** End Patch")
        self.assertEqual(events[1]["type"], "response.custom_tool_call_input.done")
        final = adapter.response({"output": [{**wire, "arguments": arguments}]})
        self.assertEqual(final["output"][0]["name"], "apply_patch")
        self.assertEqual(final["output"][0]["input"], events[0]["delta"])

    def test_local_reasoning_is_capped_and_shape_has_no_prompt(self):
        adapter = router.ToolAdapter({
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "input": [{"type": "message", "role": "user", "content": "private prompt"}],
            "tools": [{"type": "function", "name": "tool", "parameters": {"type": "object"}}],
        })
        self.assertEqual(adapter.body["reasoning"]["effort"], "low")
        self.assertEqual(adapter.body["reasoning"]["summary"], "auto")
        shape = router.request_shape(adapter.body)
        self.assertEqual(shape["input"]["roles"], ["user"])
        self.assertNotIn("private prompt", json.dumps(shape))

    def test_backend_error_message(self):
        raw = json.dumps({"error": {"message": "No user query found in messages."}})
        self.assertEqual(router.backend_error_message(raw, 400), "No user query found in messages.")
        self.assertEqual(router.backend_error_message("not-json", 503),
                         "Local model request failed with HTTP 503")

    def test_local_compaction_roundtrip(self):
        adapter = router.ToolAdapter({
            "input": [
                {"type": "message", "role": "user", "content": "original"},
                {"type": "compaction_trigger"},
            ],
            "tools": [{"type": "function", "name": "tool"}],
            "max_output_tokens": 8192,
        })
        self.assertTrue(adapter.compaction_requested)
        self.assertEqual(adapter.body["tools"], [])
        self.assertEqual(adapter.body["tool_choice"], "none")
        self.assertEqual(adapter.body["max_output_tokens"], 4096)
        self.assertEqual(adapter.body["input"][-1]["role"], "user")
        completed = {
            "type": "response.completed",
            "sequence_number": 9,
            "response": {"id": "resp_test", "status": "completed", "output": [{
                "type": "message", "content": [{"type": "output_text", "text": "summary"}],
            }]},
        }
        events = router.local_compaction_events(adapter, completed)
        self.assertEqual(events[0]["type"], "response.output_item.done")
        compacted = events[0]["item"]
        self.assertEqual(compacted["type"], "compaction")
        self.assertEqual(router.decode_local_compaction(compacted["encrypted_content"]), "summary")
        followup = router.ToolAdapter({"input": [compacted]})
        self.assertIn("summary", followup.body["input"][0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
