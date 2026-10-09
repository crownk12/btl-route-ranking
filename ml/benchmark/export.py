"""Export the benchmark champion for the demo runtime (benchmark v12, step 1 of 2).

Trains the chosen XGBoost ranker with its tuned configuration on every sampled row (the deployment
convention of train.py), exports the trees to JSON, loads them into scoring.NumpyTreeEnsemble and
verifies the numpy margins against xgboost on 5,000 rows. The hand-off file
artifacts/model-benchmark-v12/cache/ranker-export.json carries the trees, the feature list, the metrics
and the check rows; scripts/deploy_ranker.py (demo environment, step 2) builds models/model.joblib from it.

Run with .venv-bench/Scripts/python -m benchmark.export [--combination 25f:M11_xgb_ranker]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
from common import TARGETS
from scoring import NumpyTreeEnsemble
from benchmark.run import OUT, read, load_dataset, now, panel_metadata
from benchmark.models import LTRScorer
from benchmark.stages import feature_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combination', default='25f:M11_xgb_ranker')
    parser.add_argument('--reason', default='user direction 2026-09-18: apply the best gate-passing model and the expanded features')
    args = parser.parse_args()
    fs, name = args.combination.split(':')
    if name != 'M11_xgb_ranker':
        raise SystemExit('export currently supports the XGBoost ranker only')
    d = load_dataset()
    idx, names = feature_indices(d, fs)
    X = np.nan_to_num(d['X'][:, idx]); Y = d['Y']; G = d['G']
    cfg = read(f'benchmark-{fs}.json')['tuned_configs'][name]
    sel = read('model-selection.json')
    verdict = sel['verdicts'][args.combination]
    if not verdict['eligible']:
        raise SystemExit(f'{args.combination} did not pass the gates: {verdict["reasons"]}')
    model = LTRScorer('xgb', cfg, 42, 'xgb_ranker')
    model.fit(X, Y, None, groups=G)
    tmp = Path(tempfile.mkdtemp()) / 'ranker.json'
    model.ranker.save_model(str(tmp))
    ensemble = NumpyTreeEnsemble.from_xgboost_json(json.loads(tmp.read_text(encoding='utf-8')))
    rng = np.random.default_rng(0); check = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    reference = model.ranker.predict(X[check]); reproduced = ensemble.predict_margin(X[check])
    max_diff = float(np.max(np.abs(reference - reproduced)))
    if max_diff > 1e-4:
        raise SystemExit(f'numpy ensemble differs from xgboost: max abs diff {max_diff}')
    meta = panel_metadata()
    version = f"xgb-ranker-{fs}-{meta.get('model_tag', 'panel')}"
    bench = read(f'benchmark-{fs}.json')['models'][name]; test = sel['test_results'][args.combination]['test']
    old_metrics = json.loads((ROOT / 'models/metrics.json').read_text(encoding='utf-8')) if (ROOT / 'models/metrics.json').exists() else {}

    def block(m):
        return {'rows': m['rows'], 'positive_rate': m['positive_rate'], 'roc_auc': m['roc_auc'], 'brier': m['brier'], 'log_ratio_mae': m['log_ratio_mae'],
                'model_precision_at_5': m['precision_at_5'], 'model_ndcg_at_5': m['ndcg_at_5'], 'model_recall_at_5': m['recall_at_5'], 'model_map_at_5': m['map_at_5'], 'model_mrr': m['mrr'],
                'baseline_precision_at_5': None, 'baseline_ndcg_at_5': None,
                'basis_range': None}
    metrics = {'validation': block(bench['validation']), 'test': block(test),
               'split_counts': {k: v['rows'] for k, v in read('split-manifest.json')['splits'].items() if k in ('train', 'validation', 'test')},
               'features': names, 'feature_importance': None,
               'task': old_metrics.get('task'), 'evaluation_scope': 'generated trade panel (observation-calibrated); no validated real-world sanctions-evasion performance',
               'leakage_controls': old_metrics.get('leakage_controls', []) + ['expanded features dated by available_month/announce_date; see artifacts/model-benchmark-v12/leakage-audit.json'],
               'model_selection': {'chosen': name, 'feature_set': fs, 'version': version, 'note': args.reason, 'benchmark': 'artifacts/model-benchmark-v12/model-selection.json',
                                   'gates': verdict['gates'], 'previous_version': old_metrics.get('model_selection', {}).get('version')},
               'per_target': {t: {'validation': {'model_ndcg_at_5': bench['validation_breakdown']['by_target'][t]['ndcg_at_5']},
                                  'test': {'model_ndcg_at_5': sel['test_results'][args.combination]['test_breakdown']['by_target'][t]['ndcg_at_5']}} for t in TARGETS},
               'production_training_rows': int(len(X))}
    record = {'combination': args.combination, 'version': version, 'config': cfg, 'features': names, 'training_rows': int(len(X)),
              'validation_ndcg_at_5': bench['validation']['ndcg_at_5'], 'test_ndcg_at_5': test['ndcg_at_5'], 'gates': verdict['gates'], 'reason': args.reason}
    # The demo environment builds the pack (scripts/deploy_ranker.py) so the auxiliary head is pickled with the runtime's scikit-learn.
    export = {'exported_at_utc': now(), 'version': version, 'candidate': name, 'feature_set': fs, 'features': names, 'config': cfg, 'panel': meta.get('dataset_version'),
              'xgboost_model': json.loads(tmp.read_text(encoding='utf-8')), 'check_rows': check.tolist(), 'check_margins': reference.tolist(),
              'numpy_export_check': {'rows': int(len(check)), 'max_abs_diff': max_diff, 'trees': len(ensemble.trees)}, 'metrics': metrics, 'deployment': record}
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    (OUT / 'cache/ranker-export.json').write_text(json.dumps(export, ensure_ascii=False), encoding='utf-8')
    record = dict(record, numpy_export_check=export['numpy_export_check'], previous_version=None)
    print(json.dumps({k: record[k] for k in ('version', 'training_rows', 'numpy_export_check', 'validation_ndcg_at_5', 'test_ndcg_at_5', 'previous_version')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
