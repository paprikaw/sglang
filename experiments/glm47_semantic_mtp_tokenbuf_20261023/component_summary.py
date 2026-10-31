"""Summarize measured MTP phases without converting Verify time into TPOT."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def average_partition(rows):
    if not rows:
        raise ValueError('No measured phase rows')
    for row in rows:
        if abs(sum(row['partition_ms'].values()) - row['envelope_ms']) > 0.001:
            raise ValueError('Nonconserved phase envelope')
    keys = {key for row in rows for key in row['partition_ms']}
    return dict(
        envelope_ms=statistics.mean(row['envelope_ms'] for row in rows),
        partition_ms={key: statistics.mean(row['partition_ms'].get(key, 0) for row in rows)
                      for key in sorted(keys)},
        measured_rows=len(rows),
    )


def grouped(partition):
    groups = {
        'Attention': ('attention',),
        'Routed Expert': ('expert_routed',),
        'Shared Expert / dense FFN': ('expert_shared_or_dense',),
        'Dispatch + Combine': ('dispatch', 'combine'),
        'TP communication': ('tp_communication',),
        'PP SendRecv including wait': ('pp_communication',),
    }
    used = {key for keys in groups.values() for key in keys}
    result = {name: sum(partition.get(key, 0) for key in keys)
              for name, keys in groups.items()}
    result['Other / unknown / idle / overlap'] = sum(
        value for key, value in partition.items() if key not in used)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    run = p.parse_args().run
    profile_path = run / 'mtp_component_profile.json'
    profile = json.loads(profile_path.read_text())
    performance_path = run / 'comparison.json'
    performance = json.loads(performance_path.read_text())
    config = json.loads((run / 'config.json').read_text())
    phases = {}
    ep = {row['rank']: row for row in profile['ep8']['ranks']}
    pp = {row['rank']: row for row in profile['pp2ep4']['ranks']}
    for rank in (0, 4):
        if ep[rank]['phases']['target_verify']['batch_sizes'] != [4]:
            raise ValueError('Expected actual B4 EP Verify')
    phases['EP8 target Verify (mean of two DP groups)'] = average_partition([
        dict(envelope_ms=ep[r]['phases']['target_verify']['mean_envelope_ms'],
             partition_ms=ep[r]['phases']['target_verify']['mean_partition_ms'])
        for r in (0, 4)])
    pairs = profile['pp2ep4']['matched_serial_target_stage_sums']['pairs']
    if not pairs or any(row['local_batch_size'] != 4 for row in pairs):
        raise ValueError('Missing matched real B4 PP stage pairs')
    phases['PP2 x EP4 target Verify (matched stage sum)'] = average_partition([
        dict(envelope_ms=row['serial_target_stage_envelope_sum_ms'],
             partition_ms=row['serial_target_stage_partition_sum_ms']) for row in pairs])
    for name, ranks, rank_ids in [('EP8', ep, (0, 4)), ('PP2 x EP4', pp, (4,))]:
        for phase in ('draft', 'draft_extend'):
            phases[name + ' ' + phase] = average_partition([
                dict(envelope_ms=ranks[r]['phases'][phase]['mean_envelope_ms'],
                     partition_ms=ranks[r]['phases'][phase]['mean_partition_ms'])
                for r in rank_ids])
    windows = {}
    for name, ranks in [('EP8', ep), ('PP2 x EP4', pp)]:
        for rank in (0, 4):
            parts = ranks[rank]['full_profile_window_partition_ms']
            total = sum(parts.values())
            if total <= 0:
                raise ValueError('Empty measured profile window')
            windows[name + ' rank' + str(rank)] = dict(
                window_ms=total, grouped_ms=grouped(parts),
                grouped_percent={key: value * 100 / total for key, value in grouped(parts).items()})
    output = dict(
        engine_commit=config['engine_commit'],
        workload=performance['workload'],
        phase_units='ms per local B4 API; Verify has 3 candidate rows/request. PP target is the sum of matched stages for the same request IDs and Verify ordinal. Not TPOT.',
        profile_protocol='Separate profile cohort after unprofiled production measurement; no conversion to accepted-token latency.',
        communication_semantics='GPU communication kernels include waiting/polling. PP SendRecv is not pure wire time. Window percentages are local GPU timelines, not summed across ranks.',
        phases=phases, local_profile_windows=windows,
        source_files={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (profile_path, performance_path)},
    )
    (run / 'component_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    columns = list(grouped({}))
    text = ['# Semantic MTP component profile', '', output['phase_units'], '',
            output['communication_semantics'], '',
            '| Phase | Total ms | ' + ' | '.join(columns) + ' |',
            '|---|' + '---:|' * (len(columns) + 1)]
    for name, row in phases.items():
        values = grouped(row['partition_ms'])
        text.append('| ' + name + f" | {row['envelope_ms']:.3f} | " +
                    ' | '.join(f'{values[key]:.3f}' for key in columns) + ' |')
    text.extend(['', '## Local GPU profile-window shares', '',
                 '| Rank / layout | Window ms | ' + ' | '.join(columns) + ' |',
                 '|---|' + '---:|' * (len(columns) + 1)])
    for name, row in windows.items():
        text.append('| ' + name + f" | {row['window_ms']:.3f} | " +
                    ' | '.join(f"{row['grouped_percent'][key]:.2f}%" for key in columns) + ' |')
    text.extend(['', 'All other/unknown/idle/overlap time is retained. A/E scopes include the calibrated projections and shared/dense paths as labeled. Full raw partitions and source hashes are in component_summary.json.', ''])
    (run / 'COMPONENTS.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
