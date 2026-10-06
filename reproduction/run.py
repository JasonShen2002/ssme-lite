"""Recompute the manuscript's image benchmarks with the installed package.

Run from a source checkout: python reproduction/run.py --jobs 2
No training, network access, or accelerators are required.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from ssme_lite.benchmark import benchmark

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, keys, columns, atol):
    a = pd.read_csv(actual).set_index(keys).sort_index()
    b = pd.read_csv(expected).set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a[columns], b[columns], atol=atol, rtol=0, equal_nan=True)
    return float(np.nanmax(np.abs(a[columns].to_numpy() - b[columns].to_numpy())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--pipeline', choices=['all', 'checkpoints', 'transfer'], default='all')
    parser.add_argument('--verify-only', action='store_true', help='Check existing outputs without refitting')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduction')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(python=platform.python_version(), platform=platform.platform(),
                   versions={name: importlib.metadata.version(name) for name in
                             ['ssme-lite', 'numpy', 'scipy', 'pandas', 'scikit-learn']},
                   scope='SSME refits from frozen model predictions; no classifier retraining',
                   verification_only=args.verify_only, absolute_tolerance=1e-10, runs=[])
    manifest = json.loads((ROOT / 'reproduction/inputs/manifest.json').read_text())
    hashes = {item['file']: item['sha256'] for item in manifest['files']}
    jobs = [('checkpoints', 20, 'pneumoniamnist/results/nl20_nu400_m7'),
            ('transfer', 20, 'transfer/results/nl20_nu400_m5'),
            ('transfer', 50, 'transfer/results/nl50_nu400_m5'),
            ('transfer', 100, 'transfer/results/nl100_nu400_m5')]
    for name, labels, reference in jobs:
        if args.pipeline != 'all' and name != args.pipeline:
            continue
        start = time.monotonic()
        source = ROOT / 'reproduction/inputs' / (name + '.npz')
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[source.name]:
            raise ValueError(f'Input hash mismatch: {source}')
        destination = args.output / f'{name}_nl{labels}'
        expected = ROOT / 'experiments' / reference
        with np.load(source, allow_pickle=False) as data:
            scores, y, names = data['scores'], data['y'], data['model_names'].tolist()
        if not args.verify_only:
            benchmark(scores, y, names, destination, seeds=list(range(10)),
                      n_labeled=labels, n_unlabeled=400, n_draws=100, max_iter=100,
                      contribution=(labels == 20), n_jobs=args.jobs,
                      dataset='PneumoniaMNIST_infer' if name == 'checkpoints' else 'PneumoniaMNIST_transfer')
        error = compare(destination / 'details.csv', expected / 'details.csv',
                        ['seed', 'target', 'method', 'metric', 'model'],
                        ['value_estimate', 'value_truth', 'absolute_error'], 1e-10)
        selection_error = compare(destination / 'per_seed.csv', expected / 'per_seed.csv',
                                  ['seed', 'target', 'method', 'metric'],
                                  ['mae', 'spearman', 'top1_correct', 'selection_regret'], 1e-10)
        assert json.loads((destination / 'splits.json').read_text()) == json.loads((expected / 'splits.json').read_text())
        influence_error = None
        if labels == 20:
            influence_error = max(compare(destination / f'contribution_seed{s}.csv',
                                          expected / f'contribution_seed{s}.csv', ['removed_model'],
                                          ['posterior_total_variation'], 1e-10) for s in range(10))
        receipt['runs'].append(dict(name=name, n_labeled=labels, seeds=10, status='passed',
                                   input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                   max_metric_difference=error, max_selection_difference=selection_error,
                                   max_influence_difference=influence_error,
                                   seconds=round(time.monotonic() - start, 2)))
        (args.output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'PASS {name}, labels={labels}: max metric difference={error:.3g}', flush=True)
    print(args.output / 'verification.json')


if __name__ == '__main__':
    main()
