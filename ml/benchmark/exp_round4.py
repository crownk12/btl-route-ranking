"""Benchmark v14, round 4: prune the feature set by permutation importance measured inside the training period.

Start from the 25 deployed features plus the v2 core and its peer deviations (113). For each inner fold, fit on the fold's
training rows and measure, on the fold's scoring rows, how much the primary metric drops when one column is shuffled within
its scoring rows (three shuffles). Features are ranked by the mean drop over the folds; the candidate sets are the top K.
Selection: the sizes whose inner-fold score (two seeds) is within 0.002 of the best are tied; among the tied sizes the one
with the best validation score is written to feature-set.json. (The first version of this rule took the smallest tied size
and never looked at validation. It picked the top 30, whose validation score was 0.016 below the top 80 at the same
inner-fold score: the folds, all before 2019, cannot tell whether the peer and policy features matter after 2019. The rule
was changed on 2026-10-04 after seeing that; both rules and all five scores are kept in feature-set.json.) The test split is
not touched.
Run: .venv-bench/Scripts/python -m benchmark.exp_round4 [--reselect]   (in ml/; --reselect reapplies the rule to the stored scores)
"""
import sys

import numpy as np

from benchmark.lab import load_dataset, read, write, now
from benchmark.experiments import Model, folds, score_split, run_folds, run_validation, log, PRIMARY
from benchmark.exp_round1 import V1_25F
from benchmark.features_v2 import layout, group_of_v2

SEEDS = (42, 7)
SIZES = (30, 45, 60, 80, 113)
TOLERANCE = 0.002


def select(results, ranked, importance, by_fold):
    best = max(r['cv'] for r in results.values())
    tied = [size for size, r in results.items() if r['cv'] >= best - TOLERANCE]
    chosen = max(tied, key=lambda size: (results[size]['validation'], -size))
    features = ranked[:chosen]
    write('feature-set.json', {'generated_at_utc': now(),
                               'rule': f'sizes within {TOLERANCE} of the best inner-fold {PRIMARY} are tied; the tied size with the best validation {PRIMARY} is chosen; ranking by permutation drop on the inner folds',
                               'first_rule': {'rule': 'smallest tied size, validation not consulted', 'would_choose': min(tied), 'replaced': '2026-10-04, see the module docstring'},
                               'tied_sizes': tied, 'chosen_size': chosen, 'features': features, 'groups': {n: group_of_v2(n) for n in features}, 'candidates': {str(k): v for k, v in results.items()},
                               'permutation_drop': importance, 'permutation_drop_by_fold': by_fold})
    print('tied', tied, 'chosen', chosen, features[:15])


def main():
    if '--reselect' in sys.argv:
        old = read('feature-set.json')
        return select({int(k): v for k, v in old['candidates'].items()}, list(old['permutation_drop']), old['permutation_drop'], old['permutation_drop_by_fold'])
    d = load_dataset()
    lay = layout()
    start = list(dict.fromkeys(V1_25F + lay['static'] + lay['dynamic'] + [f'rel_{n}' for n in lay['relative']]))
    index = {n: i for i, n in enumerate(d['features'])}
    X = d['X'][:, [index[n] for n in start]]; grid = d['grid']
    rng = np.random.default_rng(0); drops = np.zeros((3, len(start)))
    for k, (name, fit, score) in enumerate(folds(d, grid)):
        model = Model('xgb_clf').fit(X, d, fit)
        Xs = X[score].copy(); base = score_split(d, score, model.score(Xs))[PRIMARY]
        for j in range(len(start)):
            keep = Xs[:, j].copy(); values = []
            for _ in range(3):
                Xs[:, j] = keep[rng.permutation(len(keep))]
                values.append(base - score_split(d, score, model.score(Xs))[PRIMARY])
            Xs[:, j] = keep; drops[k, j] = np.mean(values)
        print(name, 'base', round(base, 4), flush=True)
    importance = drops.mean(axis=0); order = np.argsort(-importance)
    ranked = [start[j] for j in order]
    results = {}
    for size in SIZES:
        feats = ranked[:size]; Xk = d['X'][:, [index[n] for n in feats]]
        cvs = [run_folds(d, Xk, lambda s=seed: Model('xgb_clf', seed=s), grid)['cv'] for seed in SEEDS]
        vals = [run_validation(d, Xk, lambda s=seed: Model('xgb_clf', seed=s), grid)[0][PRIMARY] for seed in SEEDS]
        results[size] = dict(cv=float(np.mean(cvs)), cv_by_seed=cvs, validation=float(np.mean(vals)))
        log({'round': 4, 'name': f'P{size}_top_by_permutation', 'n_features': size, 'cv_' + PRIMARY: results[size]['cv'], 'cv_by_seed': cvs, 'validation_' + PRIMARY: results[size]['validation']})
        print(f"top {size:3d}: cv {results[size]['cv']:.4f} ({cvs[0]:.4f}, {cvs[1]:.4f})  val {results[size]['validation']:.4f}", flush=True)
    select(results, ranked, {start[j]: round(float(importance[j]), 5) for j in order}, {start[j]: [round(float(v), 5) for v in drops[:, j]] for j in order})


if __name__ == '__main__':
    main()
