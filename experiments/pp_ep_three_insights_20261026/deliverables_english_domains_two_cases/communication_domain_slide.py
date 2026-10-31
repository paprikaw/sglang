"""Append one English slide using the two supplied communication-domain datasets."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from deck_engine import BLUE, TEAL, ORANGE, INK, MUTED
from build import finish, style, write_csv

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'h100_ep_pp_comm_domain_comparison_20261026'
PURPLE = '#9464CA'
LOCAL2 = '#75BDB2'
ATTACHMENTS = [
    Path('/home/bxb1/.codex/attachments/1c138112-4448-4e11-9d58-a7fae4d778a3/codex-clipboard-5894a80f-b953-477d-a4bb-a2dfe811e657.png'),
    Path('/home/bxb1/.codex/attachments/e5d337d6-a38b-4cdf-a941-2b13b1b76c03/codex-clipboard-de98d2a8-2386-4ccf-810d-bfd21838f75f.png')]


def rows(path):
    return list(csv.DictReader(path.open()))


def throughput_figure(path, comparisons):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.65), sharey=True)
    labels_by_case = {'2gpu_b8': '2 GPUs · NVLink\nQwen122B FP8 · B8',
                      '8gpu_b4': '8 GPUs · 2 nodes\nGLM4.7 INT4 · B4'}
    cases = [r['case'] for r in comparisons if r['phase'] == 'prefill']
    labels = [labels_by_case[case] for case in cases]
    y = np.arange(len(cases))*1.30
    for ax, phase in zip(axes, ['prefill', 'decode']):
        selected = [r for r in comparisons if r['phase'] == phase]
        ratios = [float(r['ep_over_pp']) for r in selected]
        ax.barh(y, ratios, height=.48, color=[BLUE if v >= 1 else TEAL for v in ratios])
        ax.axvline(1, color=MUTED, linestyle='--', linewidth=1.1)
        for yy, row, value in zip(y, selected, ratios):
            ax.text(value+.025, yy-.015, f'{value:.3f}×', va='center', fontsize=12, fontweight='bold')
            er, pr = float(row['ep_throughput']), float(row['pp_throughput'])
            text = f'EP {er/1000:.1f}k / PP {pr/1000:.1f}k' if phase == 'prefill' else f'EP {er:.1f} / PP {pr:.1f}'
            ax.text(.025, yy+.41, text, fontsize=10.3, color=MUTED, va='center')
        ax.set_xlim(0, 1.62)
        ax.set_xticks([0, .5, 1, 1.5])
        ax.set_yticks(y, labels, fontsize=10.5)
        ax.tick_params(axis='both', length=0, labelsize=10)
        ax.set_title('Prefill · input tok/s' if phase == 'prefill' else 'Decode · output tok/s',
                     fontsize=13, fontweight='bold', pad=13)
        ax.grid(axis='x', color='#E8EDF5', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_ylim(y[-1]+.72, -.53)
    fig.supxlabel('EP throughput / PP throughput  ·  PP = 1', y=.025, fontsize=11)
    fig.subplots_adjust(left=.27, right=.985, bottom=.13, top=.88, wspace=.25)
    finish(fig, path)


def communication_figure(path, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.65), gridspec_kw={'width_ratios': [1.6, 1]})
    cases = [(['decode4', 'decode16'], ['4 tokens', '16 tokens'], 'Decode-like · Graph', .40),
             (['prefill1024'], ['1024 tokens'], 'Prefill-like · eager', 2.45)]
    for ax, (keys, labels, title, ymax) in zip(axes, cases):
        for i, (phase, color) in enumerate([('local174', TEAL), ('local175', LOCAL2), ('cross8', PURPLE)]):
            selected = [next(r for r in metrics if r['case'] == case and r['phase'] == phase)
                        for case in keys]
            values = np.array([float(r['median_ms']) for r in selected])
            low = np.array([float(r['min_ms']) for r in selected])
            high = np.array([float(r['max_ms']) for r in selected])
            x = np.arange(len(keys))+(i-1)*.23
            ax.bar(x, values, width=.23, color=color, yerr=np.vstack([values-low, high-values]),
                   capsize=2.5, error_kw={'linewidth': .8})
            for xx, value, hi in zip(x, values, high):
                lift = .080 if len(keys) == 2 and phase == 'local175' else .027
                ax.text(xx, hi+ymax*lift, f'{value:.4f}', ha='center', va='bottom', fontsize=10.2,
                        rotation=0)
        ax.set_xticks(np.arange(len(keys)), labels, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(axis='y', labelsize=10)
        style(ax)
    fig.legend(handles=[Patch(facecolor=TEAL, label='EP4 · NVLink 174'),
                        Patch(facecolor=LOCAL2, label='EP4 · NVLink 175'),
                        Patch(facecolor=PURPLE, label='EP8 · cross-node')],
               loc='upper center', bbox_to_anchor=(.5, 1.035), ncol=3,
               frameon=False, fontsize=9.2, columnspacing=.9, handlelength=1.1)
    fig.subplots_adjust(left=.08, right=.985, bottom=.13, top=.78, wspace=.45)
    finish(fig, path)


def add_communication_domain_slide(deck, out):
    comparisons = [r for r in rows(SOURCE/'e2e_comparisons.csv')
                   if r['case'] in ['2gpu_b8', '8gpu_b4']]
    metrics = [r for r in rows(SOURCE/'sources/communication_metrics.csv')
               if r['operation'] == 'dc' and r['case'] in ['decode4', 'decode16', 'prefill1024']]
    assert len(comparisons) == 4 and len(metrics) == 9
    assert json.loads((SOURCE/'validation.json').read_text())['passed']
    for r in comparisons:
        ratio = float(r['ep_throughput'])/float(r['pp_throughput'])
        assert abs(ratio-float(r['ep_over_pp'])) < 1e-12
    assert all(int(r['n']) == 3 for r in metrics)
    left = out/'figures/domain_throughput.png'
    right = out/'figures/domain_communication.png'
    throughput_figure(left, comparisons)
    communication_figure(right, metrics)
    local_comparisons = rows(SOURCE/'communication_comparisons.csv')
    small = [float(r['cross_over_local']) for r in local_comparisons if r['case'].startswith('decode')]
    large = [float(r['cross_over_local']) for r in local_comparisons if r['case'] == 'prefill1024']
    title = 'Communication domains change the EP / PP trade-off'
    subtitle = 'Wider EP crosses the next communication boundary at every MoE layer; PP can keep expert exchange local'
    takeaway = 'Re-evaluate EP at each domain boundary: local Expert gains must offset repeated cross-domain exchange.'
    notes = (
        'The three cards are a conceptual communication-domain progression: one GPU has no inter-GPU '
        'expert exchange; two local GPUs can exchange over NVLink; a wider EP group across two nodes '
        'uses RDMA/RoCE. The single-GPU card is explanatory only: no single-GPU timing is fabricated. '
        'Left panel compares two deployment cases: two GPUs/NVLink and eight GPUs/two nodes, '
        'each shown for Prefill and Decode. The extra eight-GPU B16 row is omitted. '
        'Ratios are normalized within each saved cohort, with raw throughputs shown under each bar. '
        'Two-GPU Qwen3.5-122B FP8 uses B8, cold 4096->1 Prefill and GPU-cached4096/new1->128 Decode. '
        'The comparison is EP2 Attention TP1/DP2 versus two-slot PP2 Attention TP1. Eight-GPU '
        'GLM-4.7-AWQ INT4 uses B4, cold8192->1 Prefill and cold128->512 Decode on H100174/175. '
        'EP8 uses node-local Attention TP4 and DP2 with expert exchange across nodes; PP2xEP4 uses '
        'one node per stage, Attention TP4 and node-local EP4. PP retains cross-node stage communication. '
        'The models, precisions, Attention layouts, workloads, cache starts and local chunks differ '
        'between two- and eight-GPU cohorts, so the full-model preference shift is not a network-only '
        'causal speedup. Historical eight-GPU EP and PP also use different native Expert/communication '
        'paths. No absolute throughput ranking across models is intended. '
        'Right panel reproduces the nine Dispatch+Combine medians and observed min/max whiskers from '
        'the second supplied figure. Same DeepEP native SHA256 '
        'd9b26ac86748517800ce9f9734d865be20d089d473d9d4f6d66dde98c673d76b, BF16 hidden5120, '
        '160 experts, top8, deterministic routing and equal global logical token counts. No model, '
        'Attention or Expert computation. Local EP4 on node174 and node175 remain separate; cross-node '
        'EP8 spans both nodes. Three warmups and three aligned rank-max samples, then median. '
        'Decode-like 4/16-token calls use CUDA Graph; Prefill-like1024-token calls use eager. '
        'Cross/local dispatch+combine is5.7–6.1x for small messages and9.8–10.1x for1024 tokens. '
        'EP width, rank count, per-rank rows and QP count differ. Event spans include waits and are '
        'independent of full-model timings; they are not stacked into E2E or interpreted as NIC-only '
        'transfer time. Full original values and protocols are copied into domain_data/.')
    sources = [SOURCE/'e2e_comparisons.csv', SOURCE/'communication_comparisons.csv',
               SOURCE/'full_metrics.csv', SOURCE/'sources/communication_metrics.csv',
               SOURCE/'source_manifest.json', SOURCE/'validation.json',
               SOURCE/'sources/communication_manifest.json', SOURCE/'sources/eight_gpu_manifest.json']
    deck.slide(title, subtitle, sources=sources, notes=notes)
    for x, heading, caption, color in [
            (.78, '1 GPU · GPU-local', 'No inter-GPU expert exchange', MUTED),
            (4.93, '2 GPUs · NVLink', 'Expert exchange across local GPUs', BLUE),
            (9.08, '2 nodes · RDMA', 'Expert exchange across nodes', PURPLE)]:
        deck.rect(x, 1.76, 3.48, .62, '#FFFFFF', line='#D2DCEB')
        deck.text(heading, x+.12, 1.86, 3.24, .25, 13.5, color, True, align='center')
        deck.text(caption, x+.08, 2.16, 3.32, .17, 8.7, MUTED, align='center')
    for x in [4.39, 8.54]:
        deck.text('→', x, 1.96, .35, .32, 17, ORANGE, True, align='center')
    deck.rect(.78, 2.60, 6.27, 3.91)
    deck.rect(7.27, 2.60, 5.29, 3.91)
    deck.text('Deployment throughput · EP / PP', 1.00, 2.76, 5.83, .32, 16.5, INK, True)
    deck.text('Dispatch + Combine (ms)', 7.49, 2.76, 4.85, .32, 16.5, INK, True)
    deck.image(left, .88, 3.15, 6.07, 2.98)
    deck.image(right, 7.35, 3.13, 5.13, 2.96)
    deck.text('2 GPUs: EP2 / PP2; 8 GPUs: EP8 / PP2×EP4', 1.01, 6.25, 5.80, .18, 8.8, MUTED)
    deck.text(f'Cross / local: {min(small):.1f}–{max(small):.1f}× small; '
              f'{min(large):.1f}–{max(large):.1f}× large', 7.49, 6.25, 4.85, .18, 9.0, PURPLE, True)
    deck.rect(.78, 6.70, 11.78, .35, '#E4EDF9')
    deck.text(takeaway, 1.00, 6.76, 11.30, .24, 11.7, INK, True)
    deck.finish_slide()
    data = out/'domain_data'
    data.mkdir(exist_ok=True)
    for source in sources:
        name = source.name if source.parent == SOURCE else 'source_'+source.name
        shutil.copy2(source, data/name)
    write_csv(data/'plotted_communication.csv', metrics)
    write_csv(data/'plotted_throughput.csv', comparisons)
    (data/'attachments.json').write_text(json.dumps([
        dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in ATTACHMENTS], indent=2)+'\n')
    return sources, dict(title=title, subtitle=subtitle, takeaway=takeaway)
