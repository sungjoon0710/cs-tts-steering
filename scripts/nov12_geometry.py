"""Reproduce November 12 layer summaries from saved prompt-anchor states."""

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import read_jsonl
from src.representation_geometry import calibration_mean, contrasts, cosine, mean_direction_consistency


CONDITIONS = ['default', 'anger', 'happy', 'sad', 'control_news']


def summarize(states, rows, reference, output):
    splits = [row['split'] for row in rows]
    validation = [i for i, row in enumerate(rows) if row['cohort'] == 'independent_94']
    if len(rows) != 100 or splits.count('calibration') != 4 or len(validation) != 94:
        raise ValueError('Expected the frozen 100/4/94 November 12 cohort')
    if len({rows[i]['source_document'] for i in validation}) != len(validation):
        raise ValueError('Validation must use distinct source documents')
    if states.shape != (len(rows), len(CONDITIONS), 28, 3072):
        raise ValueError('Expected text × condition × layer × hidden states in historical order')

    genres = np.array([rows[i]['genre'] for i in validation])
    rng = np.random.default_rng(1234)
    weights = np.zeros((1000, len(validation)))
    for genre in sorted(set(genres)):
        ix = np.flatnonzero(genres == genre)
        for b, sample in enumerate(rng.choice(ix, (len(weights), len(ix)), replace=True)):
            weights[b] += np.bincount(sample, minlength=len(validation))
    weights /= len(validation)

    per_text = []
    summaries = []
    means = {}
    ref = states[:, CONDITIONS.index(reference)]
    for j, condition in enumerate(CONDITIONS):
        if condition == reference or (reference == 'control_news' and condition == 'default'):
            continue
        delta, magnitude = contrasts(ref, states[:, j])
        fixed = calibration_mean(delta, splits)
        alignment = cosine(delta, np.broadcast_to(fixed, delta.shape))
        consistency, counts = mean_direction_consistency(delta[validation])
        means[condition] = delta[validation].mean(axis=0)
        boot_magnitude = weights @ magnitude[validation]
        boot_alignment = weights @ alignment[validation]
        peaks = np.argmax(boot_magnitude, axis=1)
        for i, row in enumerate(rows):
            for layer in range(28):
                per_text.append(dict(id=row['id'], cohort=row['cohort'], source_document=row['source_document'],
                    genre=row['genre'], condition=condition, layer=layer, normalized_magnitude=magnitude[i, layer],
                    cosine_to_fixed_calibration_direction=alignment[i, layer]))
        for layer in range(28):
            lo, hi = np.percentile(boot_magnitude[:, layer], [2.5, 97.5])
            cos_lo, cos_hi = np.percentile(boot_alignment[:, layer], [2.5, 97.5])
            summaries.append(dict(condition=condition, layer=layer, independent_documents=len(validation),
                mean_normalized_magnitude=magnitude[validation, layer].mean(), magnitude_CI_low=lo, magnitude_CI_high=hi,
                mean_cosine_to_fixed_calibration=alignment[validation, layer].mean(), cosine_CI_low=cos_lo,
                cosine_CI_high=cos_hi, mean_pairwise_direction_cosine=consistency[layer],
                valid_direction_documents=counts[layer], bootstrap_peak_probability=(peaks == layer).mean()))
    cross = [dict(condition_a=a, condition_b=b, layer=layer, cosine=value)
             for a, b in combinations(means, 2) for layer, value in enumerate(cosine(means[a], means[b]))]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(per_text).to_csv(output / 'per_text_geometry.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'layer_validation_summary.csv', index=False)
    pd.DataFrame(cross).to_csv(output / 'cross_condition_direction_cosines.csv', index=False)


def plot(tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = pd.read_csv(tables / 'layer_validation_summary.csv')
    panels = [
        ('mean_normalized_magnitude', 'Normalized L2 magnitude', 'magnitude_CI_low', 'magnitude_CI_high'),
        ('mean_cosine_to_fixed_calibration', 'Cosine to the four-text direction', 'cosine_CI_low', 'cosine_CI_high'),
        ('mean_pairwise_direction_cosine', 'Mean pairwise cosine', None, None),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for condition, group in data.groupby('condition', sort=False):
        group = group.sort_values('layer')
        for ax, (field, title, low, high) in zip(axes, panels):
            line, = ax.plot(group.layer, group[field], label=condition)
            if low:
                ax.fill_between(group.layer, group[low], group[high], color=line.get_color(), alpha=.12)
            ax.set(xlabel='Zero-based layer', ylabel=title)
    axes[0].legend()
    figure.suptitle('November 12 prompt anchors on 94 validation documents')
    output.mkdir(parents=True, exist_ok=False)
    figure.savefig(output / 'layer_geometry.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    analysis = commands.add_parser('summarize')
    analysis.add_argument('--states', type=Path, required=True)
    analysis.add_argument('--cohort', type=Path, required=True)
    analysis.add_argument('--reference', choices=['default', 'control_news'], default='default')
    plotting = commands.add_parser('plot')
    plotting.add_argument('--tables', type=Path, required=True)
    for command in (analysis, plotting):
        command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'summarize':
        summarize(np.load(args.states, mmap_mode='r', allow_pickle=False), read_jsonl(args.cohort), args.reference, args.output)
    else:
        plot(args.tables, args.output)


if __name__ == '__main__':
    main()
