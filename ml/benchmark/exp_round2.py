"""Benchmark v14, round 2: which feature groups earn their place.

Round 1 showed the learner family hardly matters once the v2 features are in, so this round fixes one learner (XGBoost
binary classifier, default settings, two seeds) and varies only the feature set. Scores are the mean of the three inner
folds; validation is recorded for the record but the choice is made on the folds.
Run: .venv-bench/Scripts/python -m benchmark.exp_round2 [dataset file in the cache folder]   (in ml/)
"""
import sys

import numpy as np

from benchmark.lab import load_dataset, CACHE
from benchmark.experiments import Model, run_folds, run_validation, log, PRIMARY
from benchmark.features import ALL_FEATURES as V1_ALL
from benchmark.exp_round1 import V1_25F

SEEDS = (42, 7)


def families(names):
    """Feature-name families of the v2 block."""
    def has(n, *keys): return any(k in n for k in keys)
    f = {'v1': [n for n in names if n in V1_ALL],
         'lags': [n for n in names if has(n, '_lag')],
         'episode': [n for n in names if has(n, '_ep_', 'exp_ep')],
         'swing': [n for n in names if has(n, '_zz_', '_pl_', '_hinge_')],
         'peer': [n for n in names if n.startswith('rel_')]}
    taken = set(sum(f.values(), []))
    f['core'] = [n for n in names if n not in taken]   # seasonal, cycle state, current gap, leg lead, policy response, reporting
    return f


def main(dataset):
    d = load_dataset(CACHE / dataset)
    names = d['features']; fam = families(names); grid = d['grid']
    core_peer = [n for n in fam['peer'] if not any(k in n for k in ('_ep_', 'exp_ep', '_zz_', '_pl_'))]
    variants = {
        'F0_all': names,
        'F1_v1+core+peer(core)': fam['v1'] + fam['core'] + core_peer,
        'F2_F1+episode': fam['v1'] + fam['core'] + fam['episode'] + [n for n in fam['peer'] if not any(k in n for k in ('_zz_', '_pl_'))],
        'F3_F1+swing': fam['v1'] + fam['core'] + fam['swing'] + [n for n in fam['peer'] if not any(k in n for k in ('_ep_', 'exp_ep'))],
        'F4_F1+lags': fam['v1'] + fam['core'] + core_peer + fam['lags'],
        'F5_F1-peer': fam['v1'] + fam['core'],
        'F6_core+peer(core) (no v1)': fam['core'] + core_peer + ['horizon'],
        'F7_25f+core+peer(core)': V1_25F + fam['core'] + core_peer,
        'F8_all-lags': [n for n in names if n not in fam['lags']],
    }
    index = {n: i for i, n in enumerate(names)}
    for name, feats in variants.items():
        feats = list(dict.fromkeys(feats)); X = d['X'][:, [index[n] for n in feats]]
        cvs = []; vals = []
        for seed in SEEDS:
            make = lambda: Model('xgb_clf', seed=seed)
            cvs.append(run_folds(d, X, make, grid)['cv']); vals.append(run_validation(d, X, make, grid)[0][PRIMARY])
        log({'round': 2, 'name': name, 'n_features': len(feats), 'seeds': list(SEEDS), 'cv_' + PRIMARY: float(np.mean(cvs)), 'cv_by_seed': cvs, 'validation_' + PRIMARY: float(np.mean(vals)), 'validation_by_seed': vals})
        print(f'{name:30s} n={len(feats):3d}  cv {np.mean(cvs):.4f} ({cvs[0]:.4f}, {cvs[1]:.4f})  val {np.mean(vals):.4f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'grid.npz')
