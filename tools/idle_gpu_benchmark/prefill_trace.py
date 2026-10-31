"""Durable per-node BurstGPT prefill replay without arrival/RPS scheduling.

Original output length remains metadata. Every served request generates one
output token; only prompt length + 1 must fit the current model context.
"""
from __future__ import annotations
import collections
import copy
import json
import os
from pathlib import Path
import tempfile
from trace_stream import TraceStream


def _save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class PrefillTrace:
    """Independent fixed-node partition using public TraceStream peek/commit.

    The underlying batch size is batch_size * len(nodes) valid source rows.
    Source-index partition and context exclusions can produce fewer own rows,
    including zero. Commit empty batches immediately, nonempty batches only
    after all their own requests succeed. Uncommitted work retries on restart.

    TraceStream's cursor is ephemeral. One atomic durable wrapper cursor stores
    its committed snapshot with partition/context metadata and counters, avoiding
    a two-file partial commit. No payloads, text, or token IDs are persisted.
    """
    def __init__(self, dataset_path, cursor_path, node, nodes, batch_size=64,
                 max_context_tokens=32768, expected_source_sha256=None):
        if (not isinstance(nodes, (list, tuple)) or not nodes
                or any(not isinstance(item, str) or not item for item in nodes)
                or len(set(nodes)) != len(nodes)):
            raise ValueError('nodes must contain distinct nonempty node names')
        if node not in nodes:
            raise ValueError('node must belong to the fixed partition nodes')
        for name, value in (('batch_size', batch_size), ('max_context_tokens', max_context_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if max_context_tokens < 2:
            raise ValueError('max_context_tokens must fit a prompt and one token')
        self.node, self.nodes = node, sorted(nodes)
        self.slot = self.nodes.index(node)
        self.batch_size, self.max_context_tokens = batch_size, max_context_tokens
        self.cursor_path = Path(cursor_path).expanduser().resolve()
        self._temporary = tempfile.TemporaryDirectory(prefix='idle-prefill-trace-')
        self._pending = self._pending_counts = None
        self._commit_failed, self._stream, self.state = False, None, None
        partition = dict(method='source_index_modulo_fixed_nodes', node=node,
                         nodes=self.nodes, slot=self.slot,
                         max_context_tokens=max_context_tokens, output_tokens=1)
        try:
            source_cursor = Path(self._temporary.name) / 'source_cursor.json'
            if self.cursor_path.exists():
                with self.cursor_path.open(encoding='utf-8') as handle:
                    self.state = json.load(handle)
                if not isinstance(self.state, dict) or self.state.get('schema_version') != 1:
                    raise ValueError('Unsupported prefill trace cursor')
                if self.state.get('partition') != partition:
                    raise ValueError('Prefill cursor partition/context differs; use a new cursor')
                for key in ('committed_requests', 'committed_batches', 'excluded_for_context', 'other_partition_rows'):
                    count = self.state.get(key)
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError(f'Malformed prefill cursor {key}')
                if not isinstance(self.state.get('source_cursor'), dict):
                    raise ValueError('Prefill cursor is missing its source cursor')
                _save(source_cursor, self.state['source_cursor'])
            self._stream = TraceStream(dataset_path, source_cursor,
                batch_size=batch_size * len(self.nodes), max_total_tokens=None,
                min_output_tokens=1, expected_source_sha256=expected_source_sha256)
            if self.state is None:
                self.state = dict(schema_version=1, kind='burstgpt_prefill_partition_cursor',
                    partition=partition, source_cursor=copy.deepcopy(self._stream.state),
                    committed_requests=0, committed_batches=0,
                    excluded_for_context=0, other_partition_rows=0)
            self.sha256, self.source = self._stream.sha256, self._stream.source
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._stream is not None:
            self._stream.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def peek_batch(self):
        if self._commit_failed:
            raise RuntimeError('Cursor commit failed; reopen from durable cursor before continuing')
        if self._pending is not None:
            return copy.deepcopy(self._pending)
        batch = self._stream.peek_batch()
        requests, context_exclusions, other_partition = [], 0, 0
        for original in batch['requests']:
            if original['source_index'] % len(self.nodes) != self.slot:
                other_partition += 1
                continue
            if original['prompt_tokens'] + 1 > self.max_context_tokens:
                context_exclusions += 1
                continue
            row = {key: value for key, value in original.items() if key != 'offset_s'}
            row['original_output_tokens'] = original['output_tokens']
            row['output_tokens'] = 1
            requests.append(row)
        self._pending_counts = dict(excluded_for_context=context_exclusions,
                                    other_partition_rows=other_partition)
        self._pending = dict(schema_version=1, kind='burstgpt_sequential_prefill_partition',
            contains_original_prompt_text=False, dataset=batch['dataset'],
            partition=copy.deepcopy(self.state['partition']),
            selection=dict(model_filter='ALL', min_original_output_tokens=1,
                max_original_output_tokens=None, max_original_total_tokens=None,
                max_prompt_tokens=self.max_context_tokens - 1, output_tokens=1,
                length_policy='preserve_prompt_length_generate_one_token',
                arrival_policy='completion_driven_no_timestamp_or_rps_schedule'),
            stream=dict(batch['stream'], own_requests=len(requests),
                excluded_for_context=context_exclusions, other_partition_rows=other_partition),
            summary=dict(requests=len(requests),
                models=dict(collections.Counter(row['source_model'] for row in requests)),
                total_prompt_tokens=sum(row['prompt_tokens'] for row in requests),
                total_output_tokens=len(requests),
                total_original_output_tokens=sum(row['original_output_tokens'] for row in requests)),
            requests=requests)
        return copy.deepcopy(self._pending)

    def commit(self):
        if self._pending is None:
            raise RuntimeError('There is no pending prefill trace batch to commit')
        if self._commit_failed:
            raise RuntimeError('Cursor commit failed; reopen from durable cursor before continuing')
        next_state = copy.deepcopy(self.state)
        next_state['committed_requests'] += len(self._pending['requests'])
        next_state['committed_batches'] += 1
        for key, increment in self._pending_counts.items():
            next_state[key] += increment
        try:
            next_state['source_cursor'] = self._stream.commit()
            _save(self.cursor_path, next_state)
        except BaseException:
            # A source-only advance is ephemeral. Reopening uses durable truth
            # and safely retries the entire uncommitted own batch.
            self._commit_failed = True
            raise
        self.state = next_state
        self._pending = self._pending_counts = None
        return copy.deepcopy(self.state)


PrefillTraceStream = PrefillTrace
