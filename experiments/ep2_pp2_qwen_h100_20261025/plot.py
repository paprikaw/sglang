"""Small publication-friendly grouped bars, sourced only from validated CSV."""
import argparse
import csv
from pathlib import Path


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = run / 'analysis'
    rows = list(csv.DictReader((out / 'full_metrics.csv').open()))
    names = ['moderate', 'near_capacity_b8', 'near_capacity_b16']
    labels = ['Moderate\n8 requests', 'Near EP2 KV limit\n8 requests', 'Near EP2 KV limit\n16 requests']
    for phase in ('prefill', 'decode'):
        metrics = [('tokens_per_s', 'Fresh input token/s' if phase == 'prefill' else 'Output token/s'),
                   ('ttft_s_mean', 'Mean TTFT (s)'), ('e2e_s_mean', 'Mean end-to-end (s)')]
        if phase == 'decode':
            metrics.insert(2, ('tpot_ms_mean', 'Mean TPOT (ms)'))
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.9), constrained_layout=True)
        for ax, (metric, title) in zip(axes, metrics):
            for arm, shift, color in [('ep2', -.19, '#158f80'), ('pp2', .19, '#8d5ccd')]:
                selected = {r['case']: r for r in rows if r['phase'] == phase and r['arm'] == arm}
                values = [float(selected[name][metric]) for name in names]
                bars = ax.bar([i + shift for i in range(len(names))], values, width=.36,
                              color=color, label=arm.upper())
                precision = 3 if '_s_' in metric else (2 if '_ms_' in metric else 1)
                ax.bar_label(bars, labels=[f'{v:.{precision}f}' for v in values], fontsize=8, padding=3)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(3), labels, fontsize=9)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.13)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.16)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False, loc='upper right' if phase == 'prefill' else 'upper left')
        fig.suptitle('Qwen3.5-122B-A10B-FP8 | 2 H100 GPUs per layout | ' + phase.capitalize(), fontsize=12)
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'{phase}_comparison.{suffix}', dpi=180)
        plt.close(fig)
    report = out / 'REPORT.md'
    text = report.read_text()
    text = text.split('\n<!-- BEGIN GENERATED FIGURES -->')[0].rstrip()
    text += '\n\n<!-- BEGIN GENERATED FIGURES -->\n## 可视化\n\n![Prefill 对比](prefill_comparison.png)\n\n![Decode 对比](decode_comparison.png)\n<!-- END GENERATED FIGURES -->\n'
    report.write_text(text)
    print(str(report))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    plot(ap.parse_args().run)
