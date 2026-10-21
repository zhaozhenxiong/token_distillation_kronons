#!/usr/bin/env python3
"""Independent probability arithmetic from completed raw evaluations; no fitting."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    experiment = ROOT / 'artifacts/runs/study240_20261021/ablation240'
    manifest = json.loads((experiment / 'experiment_manifest.json').read_text())
    arms, seeds = ('none', 'czsc', 'ict', 'dual'), (42, 123, 2026)
    expected = {(a, s) for a in arms for s in seeds}
    if (manifest['status'] != 'completed' or len(manifest['runs']) != 12
            or {(r['arm'], r['seed']) for r in manifest['runs']} != expected
            or any(r['status'] != 'completed' for r in manifest['runs'])):
        raise ValueError('A complete 4 x 3 grid is required')
    dataset = Path(manifest['dataset'])
    dm = json.loads(dataset.read_text())
    if sha(dataset) != manifest['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    train_file = dataset.parent / dm['splits']['train']['path']
    if sha(train_file) != dm['splits']['train']['sha256']:
        raise ValueError('Raw training shard changed')
    counts = np.zeros(3, dtype=int)
    with train_file.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row['split'] != 'train':
                    raise ValueError('Train prior includes a different stage')
                counts[row['direction_class']] += 1
    if counts.sum() != dm['splits']['train']['row_count']:
        raise ValueError('Train count differs from manifest')
    prior = (counts + 1) / (counts.sum() + 3)
    population = json.loads((ROOT / 'artifacts/reports/population240_20261021/population.json').read_text())
    profile = {r['symbol']: r for r in population['main']}
    references, raw, hashes = None, {}, {}
    for arm in arms:
        raw[arm] = []
        for seed in seeds:
            directory = experiment / f'{arm}_seed{seed}_evaluation'
            prediction_file = directory / 'predictions.json'
            evaluation_file = directory / 'evaluation_summary.json'
            rows = sorted((r for r in json.loads(prediction_file.read_text())['rows'] if r['split'] == 'holdout'),
                          key=lambda r: (r['symbol'], r['as_of']))
            keys = [(r['symbol'], r['as_of'], r['direction_class']) for r in rows]
            if references is None:
                references = keys
            if keys != references or len(set((s, t) for s, t, _ in keys)) != len(keys):
                raise ValueError('Raw predictions are not uniquely paired')
            p = np.array([r['direction_probabilities'] for r in rows], dtype=np.float64)
            y = np.array([r['direction_class'] for r in rows], dtype=int)
            if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                    and np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)):
                raise ValueError('Invalid probabilities')
            loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
            brier = ((p - np.eye(3)[y]) ** 2).sum(1)
            accuracy = (p.argmax(1) == y).mean()
            measured = json.loads(evaluation_file.read_text())['stages']['holdout']
            values = {'log_loss': float(loss.mean()), 'brier': float(brier.mean()), 'accuracy': float(accuracy)}
            if any(not math.isclose(v, measured['direction'][k], abs_tol=1e-6) for k, v in values.items()):
                raise ValueError('Raw scores differ from evaluation summary')
            prior_loss = -np.log(prior[y])
            prior_brier = ((prior - np.eye(3)[y]) ** 2).sum(1).mean()
            if not math.isclose(prior_loss.mean(), measured['direction_train_prior_baseline']['log_loss'], abs_tol=1e-6):
                raise ValueError('Prior differs from train-only Laplace counts')
            weights = np.array([1 / population['strata'][profile[r['symbol']]['stratum']]['inclusion_probability'] for r in rows])
            raw[arm].append({'values': values, 'loss': loss, 'mean_probability': p.mean(0),
                             'weighted_log_loss': float(np.average(loss, weights=weights)),
                             'class_contributions_vs_prior': [float((loss[y == c] - prior_loss[y == c]).sum() / len(y)) for c in range(3)]})
            hashes[f'{arm}_seed{seed}'] = {'predictions': sha(prediction_file), 'evaluation_summary': sha(evaluation_file)}
    result = {'schema': 'independent_raw_probability_check_240_v1',
              'status': 'raw_scores_verified', 'script_sha256': sha(Path(__file__)),
              'dataset_manifest_sha256': sha(dataset), 'experiment_manifest_sha256': sha(experiment / 'experiment_manifest.json'),
              'rows': len(references), 'symbols': len({r[0] for r in references}),
              'dates': len({r[1] for r in references}), 'train_class_counts': counts.tolist(),
              'actual_holdout_class_counts': np.bincount(y, minlength=3).tolist(),
              'train_laplace_prior': prior.tolist(), 'prior_log_loss': float(prior_loss.mean()),
              'prior_brier': float(prior_brier), 'arms': {}, 'raw_file_sha256': hashes,
              'training_performed': False, 'source_database_accessed': False,
              'inference_scope': 'retrospective fixed selected checkpoints; no calibration fitted; old research used these market dates'}
    for arm in arms:
        metrics = {k: {'mean': statistics.mean(r['values'][k] for r in raw[arm]),
                       'seed_SD': statistics.stdev(r['values'][k] for r in raw[arm])} for k in ('log_loss', 'brier', 'accuracy')}
        metrics.update(weighted_log_loss=statistics.mean(r['weighted_log_loss'] for r in raw[arm]),
                       mean_probability=np.mean([r['mean_probability'] for r in raw[arm]], axis=0).tolist(),
                       class_contributions_vs_prior=np.mean([r['class_contributions_vs_prior'] for r in raw[arm]], axis=0).tolist(),
                       delta_log_loss_vs_none=[float((r['loss'] - n['loss']).mean()) for r, n in zip(raw[arm], raw['none'])])
        if not math.isclose(sum(metrics['class_contributions_vs_prior']), metrics['log_loss']['mean'] - result['prior_log_loss'], abs_tol=1e-10):
            raise ValueError('Class error decomposition does not sum to the prior gap')
        result['arms'][arm] = metrics
    output = ROOT / 'artifacts/reports/ablation240_independent_20261021.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_file_sha256',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
