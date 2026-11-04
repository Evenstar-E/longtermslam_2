"""Exploratory paired-world intervals; all raw observations remain visible."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_profiled_identity_actions_v1 import read, save, digest


METHODS = ['legacy_compressed', 'point_one_to_one', 'point_many', 'point_set_ray', 'union_fixed_pose', 'union_joint']
LABELS = ['Compressed', 'Point 1:1', 'Point many', 'Point set + ray', 'Union fixed G', 'Union joint']


def main():
    out = Path('results/fragment_union_joint_v1'); rows = read(out/'rows.json'); cfg = read(out/'frozen_contract.json')['config']
    table = []
    for r in rows:
        table.append(dict(split=r['split'], world=r['world'], action=r['action'], method=r['method'],
            qualified_tp=r['metric']['qualified_tp'], fp=r['metric']['fp'], bad_motion=r['metric']['bad_motion_reports'],
            task_loss=r['metric']['task_loss'], known=r['metric']['known'], elapsed_s=r['method_elapsed_s'],
            evaluations=r['evaluations'], render_calls=r['render_calls']))
    with (out/'metrics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    comparisons = []; descriptions = []
    rng = np.random.default_rng(2026102307)
    for split in ['development', 'fresh']:
        worlds = sorted({r['world'] for r in rows if r['split'] == split})
        # Both actions stay together when a world is resampled. Preserve 4/8
        # static/changed design strata; no asset/frame pseudo-replication.
        static = [w for w in worlds if w < cfg['static_worlds']]
        changed = [w for w in worlds if w >= cfg['static_worlds']]
        samples = np.c_[rng.choice(static, (10000, len(static)), replace=True),
                        rng.choice(changed, (10000, len(changed)), replace=True)]
        values = {}
        for method in METHODS:
            x = np.array([np.mean([r['metric']['task_loss'] for r in rows
                if r['split'] == split and r['world'] == world and r['method'] == method]) for world in worlds])
            assert np.isfinite(x).all() and len(x) == 12
            values[method] = x
            descriptions.append(dict(split=split, method=method, world_mean_losses=x.tolist(),
                n_worlds=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), median=float(np.median(x)),
                zero_worlds=int(np.sum(x == 0)), missing=0))
        candidate = values['union_joint']
        for method in METHODS[:-1]:
            difference = candidate-values[method]
            boot = difference[samples].mean(axis=1)
            lower, upper = np.quantile(boot, [.025, .975])
            comparisons.append(dict(split=split, candidate='union_joint', comparator=method,
                mean_paired_loss_difference=float(difference.mean()), interval95=[float(lower), float(upper)],
                wins=int(np.sum(difference < -1e-9)), ties=int(np.sum(abs(difference) <= 1e-9)),
                losses=int(np.sum(difference > 1e-9)), differences=difference.tolist(),
                degenerate_interval=bool(lower == upper), primary=method == 'point_set_ray'))
    save(out/'paired_world_analysis.json', dict(status='exploratory, not preregistered confirmatory inference',
        replication_unit='world; average the two fixed actions within world',
        bootstrap='10000 paired world draws, stratified 4 static / 8 changed; percentile intervals',
        distribution='bounded, zero-heavy discrete task losses; no Gaussian assumption or p-values',
        multiple_comparisons='one main point_set_ray contrast; other contrasts descriptive; no familywise claim',
        generalization='same procedural scene family, not external layouts or real data',
        source_sha256=digest(__file__), comparisons=comparisons, descriptions=descriptions))
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout='constrained', sharey=True)
        maximum = max(r['metric']['task_loss'] for r in rows)
        colors = ['#777777', '#777777', '#777777', '#0072B2', '#777777', '#D55E00']
        for rowindex, split in enumerate(['development', 'fresh']):
            for colindex, action in enumerate([1, 6]):
                ax = axes[rowindex, colindex]
                for m, method in enumerate(METHODS):
                    selected = sorted([r for r in rows if r['split'] == split and r['action'] == action and r['method'] == method], key=lambda r:r['world'])
                    y = np.array([r['metric']['task_loss'] for r in selected]); x = m+np.linspace(-.11, .11, len(y))
                    ax.scatter(x[:4], y[:4], marker='s', s=26, facecolors='none', edgecolors=colors[m], linewidths=1.1)
                    ax.scatter(x[4:], y[4:], marker='o', s=25, color=colors[m], alpha=.8)
                    ax.scatter([m], [y.mean()], marker='_', s=260, color='black', linewidths=2.4, zorder=5)
                ax.set_xticks(range(6), LABELS, rotation=22, ha='right')
                ax.set_title(f"{split.capitalize()}: {'short backoff' if action == 1 else 'right yaw'} (12 worlds)")
                ax.set_ylim(-.15, maximum+.4); ax.grid(axis='y', color='.9', linewidth=.7); ax.set_axisbelow(True)
                ax.set_ylabel('Observed task loss (lower is better)')
        fig.suptitle('Same acquired images; six inference methods\nSquares: static worlds. Circles: changed worlds. Black ticks: means.', fontsize=13)
        fig.savefig(out/'comparison.png', dpi=180)
        fig.savefig(out/'comparison.svg'); plt.close(fig)
    save(out/'figure_metadata.json', dict(source_data='metrics.csv', source_data_sha256=digest(out/'metrics.csv'),
        analysis_source_sha256=digest(__file__), transformations='no exclusion or smoothing; horizontal deterministic jitter only',
        uncertainty='raw world outcomes shown; no error bars; separate paired intervals in paired_world_analysis.json',
        alt_text='Four panels show all 12 world losses for six methods, separated by development/fresh and short-backoff/right-yaw acquisitions. Static and changed worlds use different marker shapes; black marks show means. Shared vertical scale starts at zero.',
        intended_medium='provisional research report; no journal-specific compliance claim'))
    print('PAIRED MAIN', [r for r in comparisons if r['primary']], flush=True)


if __name__ == '__main__': main()
