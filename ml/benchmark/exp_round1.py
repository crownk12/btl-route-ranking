"""Benchmark v14, round 1: what moves the primary metric — features, objective, training density, horizons.

Each line holds everything else fixed. Scores are the mean of the three inner folds (search) and the validation split
(confirmation); the test split is not touched. Run: .venv-bench/Scripts/python -m benchmark.exp_round1 [names...]   (in ml/)
"""
import sys

import numpy as np

from benchmark.lab import load_dataset
from benchmark.experiments import Model, columns, run_folds, run_validation, log, PRIMARY
from benchmark.features import ALL_FEATURES as V1_ALL
from benchmark.features_v2 import ALL_FEATURES_V2, RELATIVE_NAMES

V1_25F = ['import_growth', 'export_growth', 'import_yoy', 'export_yoy', 'coupled_growth', 'import_volatility', 'export_volatility', 'coverage', 'mirror_gap', 'event_count',
          'months_since_event', 'horizon', 'mirror_gap_level', 'mirror_gap_volatility', 'change_point_detected', 'change_magnitude', 'months_since_change', 'geographic_adjacency',
          'neighbor_deviation', 'similar_item_deviation', 'hs_chapter', 'absorption_gap', 'reexport_share_trend', 'inflow_level_z', 'inflow_outflow_coupling']
DEPLOYED = dict(max_depth=8, learning_rate=0.05, n_estimators=300)


def main(only):
    d = load_dataset()
    grid = d['grid']; short = d['H'] <= 24
    sets = {'25f': V1_25F, 'v1_36': list(V1_ALL), 'v2_all': list(ALL_FEATURES_V2), 'v2_no_rel': [n for n in ALL_FEATURES_V2 if n not in RELATIVE_NAMES]}
    X = {k: d['X'][:, columns(d, v)] for k, v in sets.items()}
    runs = [
        # name, feature set, model factory, training-row filter
        ('E00_baseline_25f_xgb_rank', '25f', lambda: Model('xgb_rank', DEPLOYED), grid),
        ('E01_v1_36_xgb_rank', 'v1_36', lambda: Model('xgb_rank', DEPLOYED), grid),
        ('E02_v2_all_xgb_rank', 'v2_all', lambda: Model('xgb_rank', DEPLOYED), grid),
        ('E03_v2_no_rel_xgb_rank', 'v2_no_rel', lambda: Model('xgb_rank', DEPLOYED), grid),
        ('E04_v2_xgb_clf', 'v2_all', lambda: Model('xgb_clf'), grid),
        ('E05_v2_xgb_reg_joint', 'v2_all', lambda: Model('xgb_reg'), grid),
        ('E06_v2_lgbm_rank', 'v2_all', lambda: Model('lgbm_rank'), grid),
        ('E07_v2_lgbm_clf', 'v2_all', lambda: Model('lgbm_clf'), grid),
        ('E08_v2_xgb_rank_graded', 'v2_all', lambda: Model('xgb_rank', DEPLOYED, target='graded'), grid),
        ('E09_v2_xgb_rank_pairwise', 'v2_all', lambda: Model('xgb_rank', {**DEPLOYED, 'objective': 'rank:pairwise'}), grid),
        ('E10_v2_xgb_clf_short_horizons', 'v2_all', lambda: Model('xgb_clf'), grid & short),
        ('E11_v2_xgb_clf_step2', 'v2_all', lambda: Model('xgb_clf'), d['B'] % 2 == 0),
        ('E12_v2_xgb_clf_monthly', 'v2_all', lambda: Model('xgb_clf'), None),
        ('E13_v2_cat_rank', 'v2_all', lambda: Model('cat_rank'), grid),
        ('E14_v2_cat_clf', 'v2_all', lambda: Model('cat_clf'), grid),
        ('E15_v2_extra_trees', 'v2_all', lambda: Model('et'), grid),
        ('E16_v2_random_forest_100x18', 'v2_all', lambda: Model('rf', dict(n_estimators=100, max_depth=18)), grid),  # the 300-tree unlimited-depth forest did not finish in the time limit
        ('E17_v2_mlp', 'v2_all', lambda: Model('mlp'), grid),
        ('E18_v2_logit', 'v2_all', lambda: Model('logit'), grid),
    ]
    for name, fs, make, rows in runs:
        if only and not any(name.startswith(o) for o in only):
            continue
        cv = run_folds(d, X[fs], make, rows)
        val, _, _ = run_validation(d, X[fs], make, rows, breakdown=True)
        rec = log({'round': 1, 'name': name, 'feature_set': fs, 'n_features': len(sets[fs]), 'cv_' + PRIMARY: cv['cv'], 'cv_min': cv['cv_min'], 'folds': cv['folds'], 'validation': val})
        print(f"{name:34s} cv {cv['cv']:.4f} (min {cv['cv_min']:.4f})  val {val[PRIMARY]:.4f}  val_all {val['ndcg_at_5']:.4f}  P@5 {val['precision_at_5']:.4f}  auc {val['roc_auc']:.4f}  fit {val['fit_seconds']}s", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
