import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1] / 'pp_elastic/runtime_profiles/agentx_a100'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('diagnostic_replay', ROOT / 'agentx_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Response:
    status = 200
    headers = {'Content-Type': 'text/event-stream; charset=utf-8'}

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        return iter(self.lines)


PAYLOAD = dict(sampling_params=dict(max_new_tokens=2))
ENDPOINT = dict(host='127.0.0.1', port=1, ep=1, bootstrap_port=2)


def test_empty_stream_records_framing_without_response_payload(monkeypatch):
    monkeypatch.setattr(replay.urllib.request, 'urlopen', lambda *a, **k: Response([b'\n']))
    with pytest.raises(RuntimeError, match='data_events=0, ignored_lines=1, done=False') as error:
        replay.stream(ENDPOINT, 'private-key', PAYLOAD)
    assert 'HTTP=200' in str(error.value)
    assert 'private-key' not in str(error.value)


def test_done_without_tokens_is_still_incomplete(monkeypatch):
    monkeypatch.setattr(replay.urllib.request, 'urlopen', lambda *a, **k: Response([b'data: [DONE]\n']))
    with pytest.raises(RuntimeError, match='done=True'):
        replay.stream(ENDPOINT, '', PAYLOAD)


def test_valid_stream_retains_output_and_timing(monkeypatch):
    lines = [b'data: ' + json.dumps(dict(meta_info=dict(completion_tokens=n), output_ids=list(range(n)))).encode() + b'\n'
             for n in [1, 2]] + [b'data: [DONE]\n']
    monkeypatch.setattr(replay.urllib.request, 'urlopen', lambda *a, **k: Response(lines))
    result = replay.stream(ENDPOINT, '', PAYLOAD)
    assert result['output_ids'] == [0, 1]
    assert result['output_tokens'] == 2
    assert result['tpot_ms'] >= 0


def test_prefill_failure_is_not_hidden_by_decode_stream_failure(monkeypatch):
    def failed_prefill(*_args, **_kwargs):
        raise ConnectionError('private-key: prefill unavailable')

    def failed_decode(*_args, **_kwargs):
        raise RuntimeError('Incomplete response stream')

    monkeypatch.setattr(replay, 'http', failed_prefill)
    monkeypatch.setattr(replay, 'stream', failed_decode)
    with pytest.raises(RuntimeError, match='Prefill=ConnectionError:') as error:
        replay.request_one(ENDPOINT, ENDPOINT, 'private-key', [1, 2], 2, 0)
    assert 'private-key' not in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_completed_prefill_is_distinguished_from_a_prefill_exception(monkeypatch):
    monkeypatch.setattr(replay, 'http', lambda *_args, **_kwargs: dict(meta_info=dict(completion_tokens=1)))

    def failed_decode(*_args, **_kwargs):
        raise RuntimeError('Incomplete response stream')

    monkeypatch.setattr(replay, 'stream', failed_decode)
    with pytest.raises(RuntimeError, match='Prefill=completed, completion_tokens=1'):
        replay.request_one(ENDPOINT, ENDPOINT, '', [1, 2], 2, 0)
