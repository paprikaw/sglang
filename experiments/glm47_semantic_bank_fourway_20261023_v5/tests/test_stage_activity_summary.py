"""Saved-result aggregation uses CPU-only fixtures, not benchmark results."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fourway_summary', ROOT / 'analysis' / 'summarize.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture(arm='ep8_mtp_on', cohort='timed'):
    requests = [dict(request_index=i, completion_tokens=64, first_stream_chunk_tokens=1,
                     ttft_s=1+i/100, tpot_s=.1, end_to_end_s=7.3+i/100,
                     token_events=[[1,1],[16,2.5],[48,5.7],[64,7.3]],
                     output_ids=list(range(64)), output_text='fixture only',
                     meta=dict(spec_accept_length=2., spec_accept_rate=.5, spec_verify_ct=32))
                for i in range(8)]
    return dict(arm=arm, layout='ep8' if arm.startswith('ep8') else 'pp2ep4', mtp=arm.endswith('_on'),
                cohort=cohort, engine_commit='fixed', bank_identity_sha256='semantic-bank',
                requests=requests, cuda_profiler_off_during_production=True,
                prefix_tokens_per_request=4096, actual_cached_tokens_per_request=4032, wall_s=8.)


def write(run, arm, filename, value):
    path = run / arm / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_partial_summary_never_invents_missing_four_arms(tmp_path):
    write(tmp_path, 'ep8_mtp_on', 'p4096_decode_results.json', fixture())
    summary = s.summarize(tmp_path)
    assert summary['arms']['ep8_mtp_on']['status'] == 'partial'
    assert summary['arms']['pp2ep4_mtp_on']['status'] == 'not_measured'
    assert not summary['cross_arm_comparisons']


def test_fresh_saved_metrics_and_repeat_parity(tmp_path):
    write(tmp_path, 'ep8_mtp_on', 'p4096_decode_results.json', fixture())
    write(tmp_path, 'ep8_mtp_on', 'p4096_decode_repeat_results.json', fixture(cohort='repeat'))
    summary = s.summarize(tmp_path)
    data = summary['arms']['ep8_mtp_on']
    assert data['status'] == 'measured_repeatable'
    row = data['cohorts']['timed']
    assert row['output_throughput_tok_s'] == 64
    assert row['ttft_ms']['p99'] == 1070
    assert row['ttft_ms']['mean'] == pytest.approx(1035)
    assert row['tpot_ms']['median'] == 100
    assert row['spec_verify_count_per_request']['median'] == 32
    assert row['exact_count16_to48_cadence_ms']['mean'] == 100
    assert 'output_rows' not in row
    assert 'P99' in s.markdown(summary)


def test_mtp_chunk_skips_do_not_get_interpolated(tmp_path):
    value = fixture()
    for row in value['requests']:
        row['token_events'] = [[1,1],[15,2.4],[17,2.6],[48,5.7],[64,7.3]]
    path = write(tmp_path, 'ep8_mtp_on', 'p4096_decode_results.json', value)
    assert s.cohort(path)['exact_count16_to48_cadence_ms'] is None


def test_profile_and_buffered_first_token_not_production(tmp_path):
    value = fixture()
    value['profile_only'] = True
    path = write(tmp_path, 'ep8_mtp_on', 'results.json', value)
    with pytest.raises(ValueError, match='Profile data'):
        s.cohort(path)
    value.pop('profile_only')
    value['requests'][0]['first_stream_chunk_tokens'] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='buffered'):
        s.cohort(path)


def test_cross_on_off_parity_is_actual_not_assumed(tmp_path):
    left, right = fixture(), fixture('ep8_mtp_off')
    right['requests'][3]['output_ids'][20] = -1
    write(tmp_path, 'ep8_mtp_on', 'p4096_decode_results.json', left)
    write(tmp_path, 'ep8_mtp_off', 'p4096_decode_results.json', right)
    pair = s.summarize(tmp_path)['cross_arm_comparisons'][0]
    assert pair['same_semantic_bank_identity']
    assert not pair['output_parity']['passed']
    assert pair['output_parity']['requests'][3]['first_different_output_index'] == 20


def test_bad_eight_request_identity_fails_closed():
    rows = fixture()['requests']
    rows[7]['request_index'] = 0
    with pytest.raises(ValueError, match='unique'):
        s.index_requests(rows)
