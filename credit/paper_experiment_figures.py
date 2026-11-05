"""Generate standalone figures from audited experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate(attribution, toolsandbox, output):
    for directory in (attribution, toolsandbox):
        audit = json.loads((directory / 'execution_audit.json').read_text())
        if not audit['verified']:
            raise ValueError('only audited artifacts can be plotted')
    output.mkdir(parents=True, exist_ok=True)
    a = json.loads((attribution / 'summary.json').read_text())
    names = ['Full finite', 'No decision history', 'No memory interactions']
    with (attribution / 'MC_sensitivity.csv').open() as handle:
        sensitivity = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    rows = a['table']
    axes[0].bar(range(3), [r['Shapley_MAE'] for r in rows], yerr=[r['Shapley_MAE_seed_SD'] for r in rows],
        color=['#346888', '#a7b8c5', '#c3d0d8'], capsize=4)
    axes[0].set_xticks(range(3), names, rotation=20, ha='right')
    axes[0].set_ylabel('Shapley MAE (mean ± seed SD)')
    axes[0].set_title('5 frozen training seeds; 16 contexts')
    counts = [16, 64, 256]
    values = [np.mean([float(r['MC_error_vs_exact_model_Shapley']) for r in sensitivity
        if int(r['permutations']) == n]) for n in counts]
    axes[1].plot(counts, values, 'o-', color='#346888')
    axes[1].set_xscale('log', base=2); axes[1].set_xticks(counts, [str(n) for n in counts])
    axes[1].set_xlabel('MC permutations (model queries)')
    axes[1].set_ylabel('MAE vs exact model Shapley')
    axes[1].set_title('Additional service calls: 0')
    axes[2].bar(['Full replay', 'Incremental'], [a['full_calls'], a['incremental_calls']], color=['#a7b8c5', '#346888'])
    axes[2].set_ylabel('Local calls across 512 replay pairs')
    axes[2].set_title(f"Identical state/outcome; saved {100*a['call_savings_fraction']:.2f}%")
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('finite_attribution_replay.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    t = json.loads((toolsandbox / 'summary.json').read_text())['table']
    stats = json.loads((toolsandbox / 'paired_statistics.json').read_text())['table']
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.4))
    labels = [r['method'].replace('Version-aware heuristic', 'Version heuristic').replace('CREDIT-support-gated', 'CREDIT support gate') for r in t]
    for axis, metric, column, title in [(axes[0], 'strict_VSR', 'strict_VSR', 'Strict original-grader success'),
            (axes[1], 'upstream_score', 'upstream_mean_score', 'Original continuous task score')]:
        values = np.array([r[column] for r in t])
        intervals = np.array([next(s['cluster_CI95'] for s in stats if s['method'] == r['method'] and s['metric'] == metric) for r in t])
        axis.barh(range(len(t)), values, color=['#a7b8c5'] * 8 + ['#346888'])
        axis.errorbar(values, range(len(t)), xerr=np.maximum(0, np.array([values - intervals[:, 0], intervals[:, 1] - values])),
            fmt='none', ecolor='#333333', capsize=2)
        axis.set_yticks(range(len(t)), labels); axis.invert_yaxis()
        axis.set_xlim(0, 1); axis.set_title(title); axis.set_xlabel('Task-family bootstrap 95% interval')
    fig.suptitle('ToolSandbox local adaptation: 40 tasks, one greedy Qwen actor seed; bounded CREDIT reference', fontsize=10)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('toolsandbox_comparison.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    manifest = {'attribution': str(attribution), 'toolsandbox': str(toolsandbox),
        'interpretation': 'Measured local diagnostics; neither panel establishes full manuscript CREDIT or provider drift.'}
    (output / 'figure_sources.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribution', type=Path, default=Path('results/paper_attribution_v1'))
    parser.add_argument('--toolsandbox', type=Path, default=Path('results/toolsandbox_local_v3'))
    parser.add_argument('--output', type=Path, default=Path('results/paper_figures_v1'))
    args = parser.parse_args()
    generate(args.attribution, args.toolsandbox, args.output)
