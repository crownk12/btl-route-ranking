"""Benchmark v14, round 5: hyperparameter search (Optuna TPE) for three learner families on the pruned feature set.

Objective: mean primary metric over the three inner folds (grid training rows). Validation and test are not read here.
Every trial is kept in tuning-<family>.json (one file per family, so the families can run side by side); the best
configuration of each family goes to the experiment log.
Run: .venv-bench/Scripts/python -m benchmark.exp_round5 [family ...] [--trials N]   (in ml/)
"""
import argparse
import json

import numpy as np
import optuna

from benchmark.lab import load_dataset, read, write, now
from benchmark.experiments import Model, run_folds, log, PRIMARY

SPACES = {
    'xgb_clf': lambda t: dict(max_depth=t.suggest_int('max_depth', 3, 9), learning_rate=t.suggest_float('learning_rate', 0.02, 0.15, log=True),
                              n_estimators=t.suggest_int('n_estimators', 200, 1000, step=100), min_child_weight=t.suggest_float('min_child_weight', 1, 100, log=True),
                              subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
                              reg_lambda=t.suggest_float('reg_lambda', 0.1, 30, log=True), gamma=t.suggest_float('gamma', 0.0, 5.0)),
    'xgb_rank': lambda t: dict(objective=t.suggest_categorical('objective', ['rank:ndcg', 'rank:pairwise']),
                               lambdarank_pair_method=t.suggest_categorical('lambdarank_pair_method', ['topk', 'mean']),
                               lambdarank_num_pair_per_sample=t.suggest_int('lambdarank_num_pair_per_sample', 2, 8),
                               max_depth=t.suggest_int('max_depth', 3, 9), learning_rate=t.suggest_float('learning_rate', 0.02, 0.15, log=True),
                               n_estimators=t.suggest_int('n_estimators', 200, 1000, step=100), min_child_weight=t.suggest_float('min_child_weight', 1, 100, log=True),
                               subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
                               reg_lambda=t.suggest_float('reg_lambda', 0.1, 30, log=True)),
    'lgbm_clf': lambda t: dict(num_leaves=t.suggest_int('num_leaves', 15, 255, log=True), learning_rate=t.suggest_float('learning_rate', 0.02, 0.15, log=True),
                               n_estimators=t.suggest_int('n_estimators', 200, 1200, step=100), min_child_samples=t.suggest_int('min_child_samples', 10, 300, log=True),
                               subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
                               reg_lambda=t.suggest_float('reg_lambda', 0.01, 30, log=True)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('families', nargs='*', default=list(SPACES))
    parser.add_argument('--trials', type=int, default=30)
    args = parser.parse_args()
    d = load_dataset()
    feats = read('feature-set.json')['features']
    X = d['X'][:, [d['features'].index(n) for n in feats]]; grid = d['grid']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for family in args.families:
        trials = []

        def objective(trial):
            params = SPACES[family](trial)
            cv = run_folds(d, X, lambda: Model(family, params), grid)
            trials.append({'number': trial.number, 'params': params, 'cv': cv['cv'], 'cv_min': cv['cv_min'], 'folds': {k: v[PRIMARY] for k, v in cv['folds'].items()}})
            print(f"{family} trial {trial.number:2d} cv {cv['cv']:.4f} {json.dumps(params, default=float)}", flush=True)
            return cv['cv']
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=args.trials)
        best = max(trials, key=lambda t: t['cv'])
        write(f'tuning-{family}.json', {'generated_at_utc': now(), 'family': family, 'trials': trials, 'best': best, 'n_features': len(feats),
                                        'sampler': 'optuna TPE, seed 0', 'objective': f'mean {PRIMARY} over the three inner folds, grid training rows'})
        log({'round': 5, 'name': f'T_{family}_best_of_{len(trials)}', 'n_features': len(feats), 'cv_' + PRIMARY: best['cv'], 'cv_min': best['cv_min'], 'params': best['params']})
        print(f"{family}: best cv {best['cv']:.4f} {best['params']}", flush=True)


if __name__ == '__main__':
    main()
