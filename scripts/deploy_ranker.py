"""Build models/model.joblib for the exported ranker inside the demo environment (no xgboost needed).

The benchmark environment writes <benchmark folder>/cache/ranker-export.json with the XGBoost trees, the feature list and
the verification rows: benchmark.final export (v14, the default folder; one or more weighted models on feature set v2) or
benchmark.export / scripts/export_panel_candidate.py (v12/v13, one ranker on the 25 features; set BTL_BENCH_OUT).
This script, run with the demo .venv, loads those trees into scoring.NumpyTreeEnsemble, fits the auxiliary Extra Trees
ratio head with the demo's own scikit-learn (so the pickle matches the runtime version), archives the previous pack and
writes models/model.joblib and models/metrics.json.

Usage: .venv\\Scripts\\python.exe -B scripts/deploy_ranker.py
"""
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
from common import mt
from scoring import NumpyTreeEnsemble, RankerWithAux

OUT = ROOT / __import__('os').environ.get('BTL_BENCH_OUT', 'artifacts/model-benchmark-v14')  # an earlier benchmark or a candidate panel set has its own folder
EXPORT = OUT / 'cache/ranker-export.json'
DATASET = OUT / 'cache/dataset.npz'
TOLERANCE = 1e-3  # margins of several hundred float32 trees summed in float64


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_ensemble(export):
    if 'models' in export:  # export v2: weighted members, each with its measured raw-margin intercept
        return NumpyTreeEnsemble.combine([(NumpyTreeEnsemble.from_xgboost_json(m['xgboost_model'], m['base_margin']), m['weight']) for m in export['models']])
    return NumpyTreeEnsemble.from_xgboost_json(export['xgboost_model'])


def training_rows(export, z):
    """The rows the exported trees were fitted on (v14 keeps monthly bases in the dataset and may train on a thinner grid)."""
    rule = export.get('training_row_filter')
    if rule == 'grid':
        return z['grid']
    if rule == 'step2':
        return z['B'] % 2 == 0
    return np.ones(len(z['Y']), bool)


def metric_block(m):
    return {'rows': m['rows'], 'positive_rate': m['positive_rate'], 'roc_auc': m['roc_auc'], 'pr_auc': m['pr_auc'], 'brier': None, 'log_ratio_mae': m['log_ratio_mae'],
            'model_ndcg_at_5_pos': m['ndcg_at_5_pos'], 'model_ndcg_at_10_pos': m['ndcg_at_10_pos'], 'queries': m['queries'], 'queries_with_positive': m['queries_with_positive'],
            'model_precision_at_5': m['precision_at_5'], 'model_ndcg_at_5': m['ndcg_at_5'], 'model_recall_at_5': m['recall_at_5'], 'model_hit_rate_at_5': m['hit_rate_at_5'],
            'model_map_at_5': m['map_at_5'], 'model_map': m['map'], 'model_mrr': m['mrr'], 'baseline_precision_at_5': None, 'baseline_ndcg_at_5': None, 'basis_range': None}


def metrics_v14(export, old_metrics):
    """models/metrics.json from the final evaluation of benchmark v14 (primary metric: NDCG@5 over queries with a positive)."""
    final = export['final_evaluation']; after = final['after']; before = final['before']
    primary = final['primary_metric']
    return {'validation': metric_block(after['validation']), 'test': metric_block(after['test']),
            'split_counts': {'train': after['train_rows'], 'validation': after['validation']['rows'], 'test': after['test']['rows']},
            'features': export['features'], 'feature_importance': None,
            'task': old_metrics.get('task'), 'evaluation_scope': final['scope'],
            'leakage_controls': ['features use available_month <= basis (60-month as-of history; checked by removing later rows)', 'labels mature after horizon+2+5 months',
                                 'train label maturity <= 2018-12; validation basis >= 2019-01; test basis >= 2022-01',
                                 'country names and HS6 identifiers excluded; hs_chapter is a candidate feature',
                                 'features, learner and settings chosen on three rolling-origin folds inside the training period and on validation; the test split was read once, after the configuration was frozen',
                                 'peer deviations use the other candidates of the same basis month only; see artifacts/model-benchmark-v14/'],
            'model_selection': {'chosen': export['candidate'], 'feature_set': export['feature_set'], 'version': export['version'], 'note': export['deployment']['reason'],
                                'benchmark': 'artifacts/model-benchmark-v14/selection.json', 'members': export['deployment']['members'],
                                'previous_version': old_metrics.get('model_selection', {}).get('version')},
            'per_target': {t: {s: {'model_ndcg_at_5': after[s]['by_target'][t]['ndcg_at_5'], 'model_ndcg_at_5_pos': after[s]['by_target'][t]['ndcg_at_5_pos']} for s in ('validation', 'test')}
                           for t in after['test']['by_target']},
            'per_horizon': {h: {s: {'model_ndcg_at_5_pos': after[s]['by_horizon'][h]['ndcg_at_5_pos']} for s in ('validation', 'test') if h in after[s]['by_horizon']} for h in after['test']['by_horizon']},
            'production_training_rows': export['deployment']['training_rows'],
            'previous_model_same_protocol': {'version': old_metrics.get('model_selection', {}).get('version'), 'note': 'the previously deployed configuration refitted and scored by the same code on the same splits',
                                             'validation': metric_block(before['validation']), 'test': metric_block(before['test'])},
            'random_order_reference': {s: {'model_ndcg_at_5_pos': final['random_order'][s]['ndcg_at_5_pos'], 'model_ndcg_at_5': final['random_order'][s]['ndcg_at_5']} for s in ('validation', 'test')},
            'supplementary_test_protocols': {k: {'fit': v['fit'], 'model_ndcg_at_5_pos': v['ndcg_at_5_pos'], 'model_ndcg_at_5': v['ndcg_at_5'],
                                                 'previous_model_ndcg_at_5_pos': (v.get('previous_model') or {}).get('ndcg_at_5_pos')} for k, v in final['supplementary_test_protocols'].items()},
            'primary_metric': {'name': 'NDCG@5 over queries with at least one positive', 'key': 'model_' + primary, 'target': final['target'], 'target_status': final['target_status'],
                               'actual_validation': after['validation'][primary], 'actual_test': after['test'][primary],
                               'previous_validation': before['validation'][primary], 'previous_test': before['test'][primary],
                               'why_not_all_queries': 'a query without a positive scores 0 whatever the order, so the all-query mean (model_ndcg_at_5) cannot exceed the share of queries with a positive: '
                                                      f"{after['validation']['queries_with_positive'] / after['validation']['queries']:.4f} on validation, {after['test']['queries_with_positive'] / after['test']['queries']:.4f} on test",
                               'ceiling': 'artifacts/model-benchmark-v14/ceiling.json', 'scope': final['scope']}}


def main():
    export = json.loads(EXPORT.read_text(encoding='utf-8'))
    ensemble = load_ensemble(export)
    names = export['features']
    z = np.load(DATASET, allow_pickle=True)
    features = list(z['features']); idx = [features.index(n) for n in names]
    X = np.nan_to_num(z['X'][:, idx]); Y = z['Y']
    check = np.asarray(export['check_rows'], int)
    reproduced = ensemble.predict_margin(X[check])
    max_diff = float(np.max(np.abs(np.asarray(export['check_margins']) - reproduced)))
    if max_diff > TOLERANCE:
        raise SystemExit(f'numpy ensemble differs from the exported margins: {max_diff}')
    rows = training_rows(export, z)
    aux = ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=42, n_jobs=2).fit(X[rows], Y[rows][:, 1:])
    deployed = RankerWithAux(ensemble, aux, 'xgb_ranker')
    models = ROOT / 'models'; previous = models / 'model.joblib'
    previous_version = None
    old_metrics = json.loads((models / 'metrics.json').read_text(encoding='utf-8')) if (models / 'metrics.json').exists() else {}
    if previous.exists():
        old = joblib.load(previous); previous_version = old.get('version')
        if previous_version != export['version']:  # a redeploy of the same version keeps the archive of the version before it
            archive = models / 'archive' / (previous_version or 'unknown'); archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, archive / 'model.joblib')
            if (models / 'metrics.json').exists(): shutil.copy2(models / 'metrics.json', archive / 'metrics.json')
    version = export['version']
    pack = {'model': deployed, 'features': names, 'trained_through': mt(int(z['panel_end'])), 'training_rows': int(rows.sum()), 'version': version, 'candidate': export['candidate'],
            'panel': export['panel'], 'deployed_by': 'benchmark.export', 'deployed_at_utc': now(), 'config': export.get('config') or export['deployment'].get('members'), 'score_kind': 'margin',
            'numpy_export_check': {'rows': int(len(check)), 'max_abs_diff': max_diff, 'trees': len(ensemble.trees)}, 'aux_head': 'ExtraTreesRegressor fitted in the demo environment'}
    joblib.dump(pack, previous, compress=3)
    if 'final_evaluation' in export:
        metrics = metrics_v14(export, old_metrics)
    else:
        metrics = export['metrics']
        # The v12/v13 primary metric (all-query NDCG@5); its 0.93 target was above what the metric can reach (see docs/ML.md).
        metrics['primary_metric'] = {'name': 'NDCG@5', 'target': 0.93, 'target_status': 'achieved' if metrics['test']['model_ndcg_at_5'] >= 0.93 else 'not_achieved',
                                     'actual_validation': metrics['validation']['model_ndcg_at_5'], 'actual_test': metrics['test']['model_ndcg_at_5'], 'scope': metrics.get('evaluation_scope')}
    metrics['deployed_at_utc'] = now(); metrics['deployed_by'] = 'benchmark export + scripts/deploy_ranker.py'
    # Missing reference metrics must stay missing: the previous deployment can
    # have a different item set or split. Copy dates from this export's split,
    # never inherit an old baseline and present it as a comparable experiment.
    split_manifest = OUT / 'split-manifest.json'
    if split_manifest.exists():
        splits = json.loads(split_manifest.read_text(encoding='utf-8'))['splits']
        for split in ('validation', 'test'):
            metrics[split]['basis_range'] = splits[split]['basis_range']
    # LF on every platform: manifest.json hashes this file and the repository stores it with LF
    (models / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    record = dict(export['deployment'], deployed_at_utc=now(), numpy_export_check=pack['numpy_export_check'], previous_version=previous_version,
                  rollback='copy models/archive/<previous version>/model.joblib and metrics.json back to models/ and restart the server')
    (OUT / 'deployment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'version': version, 'trees': len(ensemble.trees), 'max_abs_diff': max_diff, 'training_rows': int(rows.sum()), 'previous_version': previous_version}, ensure_ascii=False))


if __name__ == '__main__':
    main()
