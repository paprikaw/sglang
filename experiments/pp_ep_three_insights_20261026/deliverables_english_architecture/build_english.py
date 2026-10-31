"""Four English slides with saved Decode TTFT and Prefill input throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import build as original
from deck_engine import Deck, BLUE, TEAL, ORANGE, INK, MUTED
from attention_vertical import operator, down
from deployment_diagram import contact_sheet
from matplotlib import pyplot as plt
import numpy as np

ROOT = original.ROOT
EXP = original.EXP
STEM = 'PP_EP_four_slides_EN_20261026'
DEFAULT_OUT = ROOT / 'deliverables_english'
PROTOCOL = original.OLD / 'deliverables/data/protocol.json'


class EnglishDeck(Deck):
    def text(self, value, *args, **kwargs):
        if value == 'PP / EP  |  固定 GPU 预算下的部署决策':
            value = 'PP / EP  |  Deployment decisions under a fixed GPU budget'
        assert not re.search(r'[\u3400-\u9fff]', str(value)), value
        super().text(value, *args, **kwargs)
        for paragraph in self.current.shapes[-1].text_frame.paragraphs:
            for run in paragraph.runs:
                run._r.get_or_add_rPr().set('lang', 'en-US')

    def slide(self, title, subtitle='', sources=(), notes=''):
        super().slide(title, subtitle, sources, notes)
        self.current.notes_slide.notes_text_frame.text = (
            notes + '\n\nSources (absolute paths):\n' + '\n'.join(map(str, sources)))


def collect():
    metrics, components, model_components = original.collect()
    for row in original.rows(original.HIST / 'full_metrics.csv'):
        if row['case'] != 'moderate':
            continue
        model = '35B' if '35B' in row['model'] else '122B'
        key = ('H100', model, row['phase'], row['arm'])
        metrics[key]['ttft_s'] = float(row['ttft_s_mean'])
        metrics[key]['rounds'] = int(row['rounds'])
    for row in original.rows(EXP / 'experiment_summary/qwen_ep2_pp2_metrics.csv'):
        if row['workload'] != 'decode' or not (
                row['configuration'] == 'EP2' or row['pipeline_slots'] == '2'):
            continue
        arm = 'ep2' if row['configuration'] == 'EP2' else 'pp2'
        metrics[('A100', '122B', 'decode', arm)]['ttft_s'] = float(row['ttft_s_mean'])
        metrics[('A100', '122B', 'decode', arm)]['rounds'] = int(row['rounds'])
    for key, row in metrics.items():
        assert 0 < row['ttft_s'] <= row['e2e_s'] and row['rounds'] == 3
        row['throughput_unit'] = 'input tokens/s' if key[2] == 'prefill' else 'output tokens/s'
        row['ttft_unit'] = 's'
    return metrics, components, model_components


def add_deployment_slide(deck):
    notes = (
        'Conceptual two-layer deployment, read top to bottom. EP2 partitions the E routed experts '
        'of each layer equally across both GPUs and dispatches/combines tokens at every layer. '
        'PP2 places layer L1 on stage 1/GPU 0 and layer L2 on stage 2/GPU 1, keeping all E '
        'experts of each local layer. Stage activations cross the PP boundary. The PP Expert '
        'block is twice the width of an EP E/2 block: width encodes expert count, not latency. '
        'The recorded protocol uses Attention TP=1, global batch 8, local Attention batch 4, '
        'EP Attention DP=2, and two PP microbatch slots. Matching local Attention shapes '
        'supports similar expected per-call efficiency at matched contexts; total batch alone '
        'does not guarantee this for arbitrary microbatching. More local experts and fewer '
        'tokens per expert can reduce PP Expert reuse. Compare Expert execution against '
        'dispatch/combine, while retaining PP stage communication. This is a placement diagram; '
        'the next three slides contain measurements.')
    deck.slide('Similar Attention efficiency; different Expert efficiency',
        'Read top to bottom · 2-layer example · Global B=8 / local B=4 · Attention TP=1 · E experts per layer',
        sources=[PROTOCOL], notes=notes)
    for x in [.78, 6.82]:
        deck.rect(x, 1.76, 5.74, 4.88)
    deck.text('EP2', 1.03, 1.98, 1.05, .42, 23, BLUE, True)
    deck.text('Both GPUs execute each layer', 2.16, 2.06, 3.95, .30, 13, BLUE, True)
    deck.text('PP2', 7.07, 1.98, 1.05, .42, 23, TEAL, True)
    deck.text('Stage 1 above stage 2', 8.20, 2.06, 3.95, .30, 14, TEAL, True)
    gpu_x = [1.55, 4.08]
    centers = [x + 1.025 for x in gpu_x]
    for gpu, x in enumerate(gpu_x):
        deck.text(f'GPU {gpu}', x, 2.57, 2.05, .23, 11, INK, True, align='center')
    for layer, attn_y in [(1, 2.99), (2, 5.10)]:
        deck.text(f'L{layer}', 1.01, attn_y + .11, .43, .24, 12, BLUE, True)
        dispatch_y, expert_y, combine_y = attn_y + .48, attn_y + .77, attn_y + 1.25
        for x in gpu_x:
            operator(deck, x, attn_y, 'Attention · B4')
            operator(deck, x, expert_y, 'Expert · E/2', True, True)
        deck.rect(1.55, dispatch_y, 4.58, .21, '#FFF0DF')
        deck.text('Dispatch: exchange tokens across GPUs', 1.62, dispatch_y + .023,
                  4.44, .19, 10.2, ORANGE, True, align='center')
        deck.rect(1.55, combine_y, 4.58, .21, '#FFF0DF')
        deck.text('Combine: return and merge outputs', 1.62, combine_y + .023,
                  4.44, .19, 10.2, ORANGE, True, align='center')
        for center in centers:
            down(deck, center, attn_y + .415, .055)
            down(deck, center, dispatch_y + .223, .055)
            down(deck, center, expert_y + .415, .055)
    for center in centers:
        down(deck, center, 4.50, .50)
    deck.text('Next layer', 3.25, 4.69, 1.18, .23, 10.5, MUTED, align='center')
    for gpu, stage, layer, top, attn_y, expert_y in [
            (0, 1, 1, 2.55, 2.99, 3.76), (1, 2, 2, 4.63, 5.10, 5.87)]:
        deck.rect(7.07, top, 5.24, 1.74, '#F6FBF9', line='#CAE6DC')
        deck.text(f'GPU {gpu} · Stage {stage} (L{layer})', 7.29, top + .12,
                  4.80, .25, 12.5, TEAL, True)
        operator(deck, 8.665, attn_y, 'Attention · B4')
        down(deck, 9.69, attn_y + .45, .25)
        operator(deck, 7.64, expert_y, 'Expert · E (all experts of this layer)',
                 True, False, width=4.10)
    down(deck, 9.69, 4.34, .24, ORANGE)
    deck.text('Transfer activations', 9.99, 4.37, 2.00, .22, 10.5, ORANGE, True)
    deck.text('More local experts; fewer tokens per expert', 7.07, 6.43, 5.24, .20, 10.5, MUTED)
    deck.rect(.78, 6.76, 11.78, .33, '#E4EDF9')
    deck.text('Key trade-off: Expert execution efficiency  ↔  Dispatch / Combine communication',
              1.00, 6.815, 11.30, .25, 13.1, INK, True)
    deck.finish_slide()


def metric_chart(path, pairs, metrics, plot_rows):
    """Two columns retain the comparison groups; rows have explicit metric units."""
    fig, axes = plt.subplots(len(plot_rows), 2, figsize=(5.3, 4.6))
    data = []
    for r, selectors in enumerate(plot_rows):
        for c, ((label, prefix), (key, scale, heading, precision)) in enumerate(zip(pairs, selectors)):
            ax = axes[r, c]
            values = [metrics[(*prefix, arm)][key] * scale for arm in ['ep2', 'pp2']]
            bars = ax.bar([0, 1], values, color=[BLUE, TEAL], width=.58)
            for bar, value, arm in zip(bars, values, ['ep2', 'pp2']):
                ax.annotate(f'{value:.{precision}f}',
                            (bar.get_x() + bar.get_width()/2, value), xytext=(0, 4),
                            textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
                data.append(dict(group=label, hardware=prefix[0], model=prefix[1], phase=prefix[2],
                                 arm=arm, metric=key, raw_value=metrics[(*prefix, arm)][key],
                                 display_scale=scale, displayed_value=value, display_label=heading))
            ax.set_xticks([0, 1], ['EP2', 'PP2'], fontsize=11)
            ax.set_ylim(0, max(values)*1.35)
            ax.set_title((label + '\n' if r == 0 else '') + heading,
                         fontsize=11.5, fontweight='bold', pad=9)
            ax.tick_params(axis='y', labelsize=9)
            original.style(ax)
    fig.subplots_adjust(left=.10, right=.985, bottom=.08, top=.87, wspace=.43, hspace=.93)
    original.finish(fig, path)
    return data


def component_chart(path, pairs, components, scope):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))
    labels = ['Expert\n' + scope, 'Dispatch +\nCombine', 'PP comm.\nNCCL residency']
    for index, (ax, (label, prefix)) in enumerate(zip(axes, pairs)):
        top = 0
        for offset, arm, color in [(-.18, 'ep2', BLUE), (.18, 'pp2', TEAL)]:
            values = [components[(*prefix, arm)][key] for key in ['expert', 'exchange', 'pipeline']]
            bars = ax.bar(np.arange(3) + offset, values, .32, color=color, label=arm.upper())
            top = max(top, max(values))
            for bar, value in zip(bars, values):
                ax.annotate(f'{value:.1f}' if scope == 'module' else f'{value:.3f}',
                            (bar.get_x() + bar.get_width()/2, value), xytext=(0, 5),
                            textcoords='offset points', ha='center', fontsize=9.5)
        ax.set_xticks(np.arange(3), labels, fontsize=9.6)
        ax.set_ylim(0, top*1.30)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=14)
        original.style(ax)
        if index == 1:
            ax.legend(frameon=False, ncol=2, fontsize=10, loc='upper right')
    fig.subplots_adjust(left=.07, right=.995, bottom=.18, top=.85, wspace=.34)
    original.finish(fig, path)


def main(out=None, gpu_throughput=False, with_domains=False, with_architecture=False):
    with_domains = with_domains or with_architecture
    gpu_throughput = gpu_throughput or with_domains
    if out is None:
        out = (ROOT / 'deliverables_english_architecture' if with_architecture else
               ROOT / 'deliverables_english_domains_two_cases' if with_domains else
               ROOT / 'deliverables_english_throughput' if gpu_throughput else DEFAULT_OUT)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    original.setup()
    metrics, components, model_components = collect()
    hardware = [('A100 · Marlin', ('A100', '122B', 'decode')),
                ('H100 · DeepGEMM', ('H100', '122B', 'decode'))]
    models = [('35B · BF16', ('H100', '35B', 'prefill')),
              ('122B · FP8', ('H100', '122B', 'prefill'))]
    phases = [('Prefill', ('H100', '122B', 'prefill')),
              ('Decode', ('H100', '122B', 'decode'))]
    e2e = ('e2e_s', 1, 'E2E latency (s) ↓', 3)
    ttft = ('ttft_s', 1000, 'TTFT (ms) ↓', 1)
    throughput = ('throughput', .001, 'Input throughput\n(k tokens/s) ↑', 1)
    shared_notes = (
        'E2E and TTFT are formal mean per-request latencies across three rounds. TTFT means time '
        'to first streamed token within this workload. Decode starts from 4096 cached tokens, '
        'extends by 1 fresh token and produces 128 output tokens; its TTFT does not include a '
        'cold 4096-token prefill. Prefill is cold, 4096 fresh tokens and 1 output token. '
        'Prefill input throughput is the mean of per-round fresh input tokens / cohort makespan, '
        'reported in k tokens/s, and is not derived by dividing tokens by mean request latency. '
        'Profiled GPU activities are independent of formal latency and are not additive E2E slices. '
        'PP communication is NCCL GPU residency including wait, not wire-only latency. ')
    def reduction(prefix, winner):
        loser = 'pp2' if winner == 'ep2' else 'ep2'
        return 100*(1-metrics[(*prefix, winner)]['e2e_s']/metrics[(*prefix, loser)]['e2e_s'])
    small_extra = model_components[('H100', '35B', 'prefill', 'pp2')]['expert'] - model_components[('H100', '35B', 'prefill', 'ep2')]['expert']
    big_extra = model_components[('H100', '122B', 'prefill', 'pp2')]['expert'] - model_components[('H100', '122B', 'prefill', 'ep2')]['expert']
    slides = [
        dict(name='gpu', title='GPU platform can reverse the EP / PP winner',
             subtitle='122B FP8 · 2 GPUs · 8 requests · Per request: 4096 cached + 1 fresh token → 128 output tokens',
             pairs=hardware, components=components, scope='GEMM',
             unit='GPU ms / card / actual layer invocation',
             plot_rows=[(e2e, e2e), (ttft, ttft)],
             takeaway=f'E2E latency: PP2 wins on A100 (−{reduction(hardware[0][1], "pp2"):.1f}%); EP2 wins on H100 (−{reduction(hardware[1][1], "ep2"):.1f}%).',
             notes=shared_notes + 'Native platform baselines: A100 Marlin/eager and H100 DeepGEMM/Graph. '
             'Expert scope is routed GEMM only. D+C covers DeepEP GPU kernels. Normalize by actual '
             'layer calls, because EP and PP place different numbers of layers on each card. '
             'The added TTFT data show lower TTFT for PP2 on both platforms, even when EP2 has '
             'lower complete-request E2E on H100.'),
        dict(name='model', title='PP pays a different Expert cost for each model',
             subtitle='H100 · 2 GPUs · 8 requests · Cold Prefill: 4096 fresh tokens → 1 output token per request',
             pairs=models, components=model_components, scope='module',
             unit='GPU ms / card / complete Prefill cohort',
             plot_rows=[(e2e, e2e), (throughput, throughput)],
             takeaway=f'Extra PP2 Expert time: {small_extra:.1f} ms/card for 35B; {big_extra:.1f} ms/card for 122B.',
             notes=shared_notes + 'These are the named 35B BF16 and 122B FP8 checkpoints, not a size-only '
             'ablation. Right chart shows routed Expert module GPU kernel activity for the complete '
             'B8 Prefill cohort, averaged over ranks; shared Expert is excluded. Historical 122B PP '
             'Prefill uses ready-send; the 35B baseline does not. The small-model EP Prefill varies '
             'across the three saved rounds.'),
        dict(name='phase', title='Prefill and Decode can favor different layouts',
             subtitle='H100 · 122B FP8 · 2 GPUs · 8 requests · Prefill: cold 4096 → 1; Decode: cached 4096 + 1 → 128',
             pairs=phases, components=components, scope='GEMM',
             unit='GPU ms / card / actual layer invocation',
             plot_rows=[(e2e, e2e), (throughput, ttft)],
             takeaway=f'E2E latency: PP2 wins Prefill (−{reduction(phases[0][1], "pp2"):.1f}%); EP2 wins Decode (−{reduction(phases[1][1], "ep2"):.1f}%).',
             notes=shared_notes + 'Same 122B H100 checkpoint, phase-specific native paths. '
             'Both phase component charts select routed Expert GEMM and DeepEP dispatch/combine. '
             'The bottom-left supplemental plot is Prefill input throughput; the bottom-right '
             'is Decode TTFT. PP2 has lower TTFT while EP2 has lower full-request Decode E2E.')]
    if gpu_throughput:
        output_throughput = ('throughput', 1, 'Output throughput\n(tokens/s) ↑', 1)
        slides[0]['plot_rows'] = [(e2e, e2e), (output_throughput, output_throughput)]
        a100_gain = 100*(metrics[('A100', '122B', 'decode', 'pp2')]['throughput'] /
                         metrics[('A100', '122B', 'decode', 'ep2')]['throughput']-1)
        h100_gain = 100*(metrics[('H100', '122B', 'decode', 'ep2')]['throughput'] /
                         metrics[('H100', '122B', 'decode', 'pp2')]['throughput']-1)
        slides[0]['takeaway'] = (f'Output throughput: PP2 +{a100_gain:.1f}% on A100; '
                                f'EP2 +{h100_gain:.1f}% on H100.')
        slides[0]['notes'] = (
            'E2E is formal mean complete-request latency across three saved rounds. Output '
            'throughput is the mean of per-round output tokens / cohort makespan, in tokens/s. '
            'Each round has 8 requests producing 128 tokens each (1024 output tokens in total). '
            'The cached 4096-token prefix and single fresh input token are unchanged. The lower '
            'row replaces the previous TTFT plots with output throughput for A100 and H100. '
            'Native platform baselines: A100 Marlin/eager and H100 DeepGEMM/Graph. Expert scope '
            'is routed GEMM only. D+C covers DeepEP GPU kernels. Normalize by actual layer calls '
            'because EP and PP place different numbers of layers on each card. PP communication '
            'is NCCL GPU residency including wait. Profiled activities are independent of formal '
            'latency and are not additive E2E slices. Both E2E and throughput favor PP2 on A100 '
            'and EP2 on H100.')
    deck = EnglishDeck(out)
    add_deployment_slide(deck)
    latency_data, component_data, displayed_data = [], [], []
    sources = original.SOURCES + [PROTOCOL]
    for slide in slides:
        left, right = figures/(slide['name']+'_latency.png'), figures/(slide['name']+'_components.png')
        displayed_data.extend(dict(slide=slide['name'], **row) for row in
                              metric_chart(left, slide['pairs'], metrics, slide['plot_rows']))
        component_chart(right, slide['pairs'], slide['components'], slide['scope'])
        deck.slide(slide['title'], slide['subtitle'], sources=sources, notes=slide['notes'])
        deck.rect(.78, 1.82, 4.25, 4.62)
        deck.rect(5.26, 1.82, 7.30, 4.62)
        deck.text('End-to-end and serving metrics', 1.00, 2.03, 3.83, .40, 15.5, INK, True)
        deck.text('Expert and communication', 5.51, 2.03, 6.80, .40, 19, INK, True)
        deck.image(left, .91, 2.47, 4.0, 3.77)
        deck.image(right, 5.41, 2.53, 7.00, 3.51)
        deck.text(slide['unit'], 5.57, 6.10, 6.65, .25, 10.5, MUTED)
        deck.rect(.78, 6.62, 11.78, .40, '#E4EDF9')
        deck.text(slide['takeaway'], 1.00, 6.69, 11.3, .26, 13, INK, True)
        deck.finish_slide()
        for label, prefix in slide['pairs']:
            for arm in ['ep2', 'pp2']:
                identity = dict(slide=slide['name'], group=label, hardware=prefix[0],
                                model=prefix[1], phase=prefix[2], arm=arm)
                latency_data.append(dict(**identity, **metrics[(*prefix, arm)]))
                component_data.append(dict(**identity, scope='routed Expert '+slide['scope'],
                                           unit=slide['unit'], **slide['components'][(*prefix, arm)]))
    domain_insight = None
    if with_domains:
        from communication_domain_slide import add_communication_domain_slide
        extra_sources, domain_insight = add_communication_domain_slide(deck, out)
        sources += extra_sources
    architecture_insight = None
    if with_architecture:
        from proposed_architecture import add_proposed_architecture
        architecture_insight = add_proposed_architecture(deck, out)
    stem = 'PP_EP_six_slides_EN_architecture_20261026' if with_architecture else (
        'PP_EP_five_slides_EN_domains_two_cases_20261026' if with_domains else (
            'PP_EP_four_slides_EN_throughput_20261026' if gpu_throughput else STEM))
    result = deck.save(stem)
    contact_sheet(deck, out/'contact_sheet.png')
    original.write_csv(out/'latency.csv', latency_data)
    original.write_csv(out/'expert_communication.csv', component_data)
    original.write_csv(out/'displayed_metrics.csv', displayed_data)
    original.dump(out/'sources.json', [dict(path=str(p), bytes=p.stat().st_size,
                                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    original.dump(out/'build_result.json', result)
    insights = [dict(title=s['title'], subtitle=s['subtitle'], takeaway=s['takeaway']) for s in slides]
    if domain_insight:
        insights.append(domain_insight)
    if architecture_insight:
        insights.append(architecture_insight)
    original.dump(out/'insights.json', insights)
    (out/'speaker_notes.md').write_text('\n\n'.join(
        f"## {s['slide']} {s['title']}\n\n{s['notes']}" for s in deck.notes)+'\n')
    for name in ['build_english.py', 'build.py', 'attention_vertical.py', 'deployment_diagram.py']:
        shutil.copy2(ROOT/name, out/name)
    if with_domains:
        shutil.copy2(ROOT/'communication_domain_slide.py', out/'communication_domain_slide.py')
    if with_architecture:
        shutil.copy2(ROOT/'proposed_architecture.py', out/'proposed_architecture.py')
    shutil.copy2(original.OLD/'deck_engine.py', out/'deck_engine.py')
    shutil.copy2(PROTOCOL, out/'attention_placement_protocol.json')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--gpu-throughput', action='store_true',
                        help='Replace slide 2 TTFT with Decode output throughput')
    parser.add_argument('--with-domains', action='store_true',
                        help='Append the communication-domain slide to the throughput revision')
    parser.add_argument('--with-architecture', action='store_true',
                        help='Append the proposed profiler/model/autoscaler architecture as slide 6')
    args = parser.parse_args()
    main(args.out, gpu_throughput=args.gpu_throughput, with_domains=args.with_domains,
         with_architecture=args.with_architecture)
