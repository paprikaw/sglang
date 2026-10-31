"""Bounded, passive progress projection. Estimates never become request counts."""
import json
import math
import re
import time


def point_progress(started_path, read_json, now=None):
    now = time.time() if now is None else now
    started = read_json(started_path, {})
    trial = started_path.parent
    result = read_json(trial/'result.json', {})
    common = dict(case=started.get('case'), trial=started.get('trial'),
                  elapsed_s=max(0, now-started.get('timestamp', now)),
                  completed_requests=None, dispatched_requests=None,
                  counters_available=False, basis='saved_files')
    if result:
        return dict(common, phase='validated' if result.get('passed') else 'invalid',
                    checks=result.get('checks', {}), estimated=False, progress_percent=100)
    live = read_json(trial/'live_progress.json', {})
    if live and live.get('case') == started.get('case'):
        reserved=set(common)|{'counters_available','sample_age_s','stale','estimated'}
        return dict(common, **{k:v for k,v in live.items() if k not in reserved},
                    completed_requests=live.get('completed_requests'),
                    dispatched_requests=live.get('dispatched_requests'),
                    counters_available=True, sample_age_s=max(0, now-live['timestamp']),
                    stale=now-live['timestamp'] > 10, estimated=False, basis='client_telemetry')
    match = re.fullmatch(r'P-C(\d+)-A(\d+)-R([\d.]+)-W([\d.]+)', started.get('case',''))
    if not match:
        return dict(common, phase='preparation_or_decode', estimated=False,
                    note='此客户端未写入逐请求进度；不能从 GPU 利用率推算完成数')
    C, A, rate, window = map(float, match.groups())
    common.update(context_tokens=int(C), append_tokens=int(A), request_rate=rate,
                  score_window_s=window, arrival_warmup_s=30, drain_limit_s=120,
                  planned_requests=math.ceil((30+window)*rate),
                  planned_scored_requests=math.ceil((30+window)*rate)-math.ceil(30*rate))
    warmup = trial/'append_warmup/requests.jsonl'
    try:
        # At most eight short warmup records; do not read bank or full result logs.
        rows = [json.loads(line) for line in warmup.read_text().splitlines() if line]
        anchor = max(row['end_wall'] for row in rows)
    except (OSError, ValueError, KeyError):
        return dict(common, phase='context_and_append_warmup', estimated=False,
                    note='正在准备固定 context / append；正式窗口尚无开始证据')
    # The before-metrics scrape occurs after this anchor. It is only an estimate.
    elapsed = max(0, now-anchor)
    phase = ('arrival_warmup' if elapsed < 30 else 'score_window' if elapsed < 30+window
             else 'drain_or_validation' if elapsed < 30+window+120 else 'validation_or_delayed')
    return dict(common, phase=phase, estimated=True, basis='warmup_end_wall_estimate',
                phase_elapsed_s=elapsed if elapsed<30 else elapsed-30 if elapsed<30+window else elapsed-30-window,
                progress_percent=min(100, max(0, elapsed-30)*100/window),
                window_remaining_estimate_s=max(0, 30+window-elapsed),
                scheduled_arrivals_estimate=min(common['planned_requests'], math.ceil(elapsed*rate)),
                note='窗口时间 / 计划到达数为估计；实际发送、完成、排队数尚不可用')


def task_details(cards, specs, read_json, now=None):
    now = time.time() if now is None else now
    tasks = []
    for card, (_,_,base,attempt,_) in zip(cards, specs):
        paths = list((base/'measurements'/card['arm']/attempt).glob('*/trial*/started.json'))
        current = [p for p in paths if not p.with_name('result.json').exists()]
        path = max(current, key=lambda p:read_json(p,{}).get('timestamp',0), default=None)
        detail = point_progress(path, read_json, now) if path and card['live'] else None
        accepted = card.get('accepted_kinds', {})
        tasks.append(dict(id=card['arm'], label=card['label'], state=card['state'],
                          pid=card['controller_pid'], stage=card['stage'], live=card['live'],
                          valid=card['valid'], expected=card['expected'], attempt=attempt,
                          phases=accepted, point=detail, last_case=card['last_case']))
    return tasks
