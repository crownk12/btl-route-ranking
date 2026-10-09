"""Benchmark v14, round 3: does more training data of the same kind help?

Replica panels (benchmark.replay; independent draws of the same generator, months up to the training cut-off only) are
appended to the training rows of every inner fold and of the validation fit. Scoring rows are always the real panel's.
Same learner as round 2 (XGBoost classifier, defaults, seed 42), lean feature set (v1 + core + core peer deviations).
Run: .venv-bench/Scripts/python -m benchmark.exp_round3   (in ml/)
"""
import numpy as np

from benchmark.lab import load_dataset, CACHE  # sets the import path for `common`
from common import mi
from benchmark.experiments import Model, FOLD_CUTS, score_split, log, PRIMARY
from benchmark.exp_round2 import families

COUNTS = (0, 1, 2, 4, 8)


def main():
    d = load_dataset(CACHE / 'grid.npz')
    fam = families(d['features'])
    feats = fam['v1'] + fam['core'] + [n for n in fam['peer'] if not any(k in n for k in ('_ep_', 'exp_ep', '_zz_', '_pl_'))]
    cols = [d['features'].index(n) for n in feats]
    X = d['X'][:, cols]; y = d['Y'][:, 0].astype(int); A = d['A']
    replicas = []
    for k in range(1, max(COUNTS) + 1):
        z = np.load(CACHE / f'replica-{k}.npz', allow_pickle=True)
        assert list(z['features']) == d['features']
        replicas.append((z['X'][:, cols], z['Y'][:, 0].astype(int), z['A']))
    splits = [(f'fold_{cut[:4]}', mi(cut), d['masks']['train'] & d['grid'] & (d['B'] >= mi(cut) + 1) & (A <= mi(cut) + 36)) for cut in FOLD_CUTS]
    splits.append(('validation', mi('2018-12'), d['masks']['validation']))
    for count in COUNTS:
        scores = {}
        for name, cut, score in splits:
            fit = d['masks']['train'] & (A <= cut)
            Xs = [X[fit]] + [r[0][r[2] <= cut] for r in replicas[:count]]; ys = [y[fit]] + [r[1][r[2] <= cut] for r in replicas[:count]]
            model = Model('xgb_clf')
            import xgboost as xgb
            model.model = xgb.XGBClassifier(objective='binary:logistic', subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8, random_state=42,
                                            max_depth=6, learning_rate=0.05, n_estimators=400).fit(np.concatenate(Xs), np.concatenate(ys))
            scores[name] = score_split(d, score, model.score(X[score]))[PRIMARY]
            scores[name + '_fit_rows'] = int(sum(len(v) for v in ys))
        cv = float(np.mean([scores[f'fold_{c[:4]}'] for c in FOLD_CUTS]))
        log({'round': 3, 'name': f'R{count}_replicas', 'replicas': count, 'n_features': len(feats), 'cv_' + PRIMARY: cv, 'scores': scores})
        print(f"replicas {count}: train rows (validation fit) {scores['validation_fit_rows']:>9,}  cv {cv:.4f}  folds {[round(scores[f'fold_{c[:4]}'], 4) for c in FOLD_CUTS]}  val {scores['validation']:.4f}", flush=True)


if __name__ == '__main__':
    main()
