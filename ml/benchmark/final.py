"""Benchmark v14 final stage: evaluate the frozen configuration once and hand it to the demo runtime.

Reads selection.json (round 6) and feature-set.json (round 4). Nothing is tuned here.

  evaluate  fit on the train split, score validation and test (the test split is read here for the first time), with the
            deployed 25-feature ranker fitted and scored the same way as the reference. Also: per target / horizon / item,
            hold-outs (a target, an item or a candidate country left out of training), two supplementary test protocols
            (refit on train+validation; yearly walk-forward refits), Top-5 examples, importances, timings
            -> final-evaluation.json (FINAL_REPORT.md in the same folder is the written account of it)
  export    fit the same configuration on every mature row and write cache/ranker-export.json for scripts/deploy_ranker.py

Run: .venv-bench/Scripts/python -m benchmark.final evaluate|export|all   (in ml/)
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmark.lab import load_dataset, read, write, now, panel_metadata, CACHE
from common import TARGETS, ITEMS, CANDIDATES, mi, mt
from benchmark.experiments import Model, PRIMARY, THREADS
from benchmark.metrics import evaluate, ranking_metrics, per_query_indices
from benchmark.exp_round1 import V1_25F, DEPLOYED
from benchmark.features_v2 import group_of_v2

VERSION = 'xgb-v2f-v3-calibrated-17items'
KEYS = ('ndcg_at_5_pos', 'ndcg_at_5', 'ndcg_at_10_pos', 'precision_at_5', 'recall_at_5', 'recall_at_10', 'hit_rate_at_5', 'map', 'map_at_5', 'mrr', 'roc_auc', 'pr_auc',
        'queries', 'queries_with_positive', 'rows', 'positive_rate')


class Frozen:
    """The selected configuration as a list of (family, params, seed, weight) members and the training-row filter."""

    def __init__(self, d):
        sel = read('selection.json'); rec = sel['chosen_record']; self.name = sel['chosen']; self.kind = rec['kind']
        self.features = sel['features']; self.columns = [d['features'].index(n) for n in self.features]
        rows = rec.get('rows', 'grid')
        self.rows_label = rows
        self.rows = d['grid'] if rows == 'grid' else (d['B'] % 2 == 0) if rows == 'step2' else np.ones(len(d['B']), bool)
        if self.kind == 'blend':
            w = rec['weight_classifier']
            self.members = [('xgb_clf', sel['tuned']['xgb_clf'], 42, w), ('xgb_rank', sel['tuned']['xgb_rank'], 42, 1 - w)]
            self.scale = True     # each margin is divided by its spread on the fit rows before mixing
        else:
            seeds = rec.get('seeds', [42])
            self.members = [(rec['family'], rec['params'], s, 1.0 / len(seeds)) for s in seeds]
            self.scale = False

    def fit(self, d, X, mask):
        """Returns (models, weights): score = sum(weight * margin)."""
        fit = mask & self.rows
        models = [Model(family, params, seed=seed).fit(X, d, fit) for family, params, seed, _ in self.members]
        weights = [w for *_, w in self.members]
        if self.scale:
            sample = np.flatnonzero(fit)[::7]
            weights = [w / float(np.std(m.score(X[sample]))) for w, m in zip(weights, models)]
        return models, weights

    @staticmethod
    def score(models, weights, X):
        return sum(w * m.score(X) for w, m in zip(weights, models))


def aux_head(X, Y):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=42, n_jobs=THREADS).fit(np.nan_to_num(X), Y[:, 1:])


def split_metrics(d, mask, score, ratios=None):
    pred = np.column_stack([score, ratios if ratios is not None else np.full((len(score), 2), np.nan)])
    m = evaluate(d['Y'][mask], pred, d['G'][mask], probabilistic=False)
    return {k: m[k] for k in KEYS} | {'log_ratio_mae': m['log_ratio_mae']}


def by(d, mask, score, field, values):
    out = {}
    for v in values:
        m = d[field][mask] == v
        if m.any():
            r = ranking_metrics(d['Y'][mask][m, 0], score[m], d['G'][mask][m])
            out[str(v)] = {k: r[k] for k in ('ndcg_at_5_pos', 'ndcg_at_5', 'precision_at_5', 'recall_at_5', 'queries', 'queries_with_positive')}
    return out


def top5_examples(d, mask, new, old, count=8):
    """Test queries with at least one positive, spread over targets and horizons: who each model put in its Top-5 and who rose."""
    g = d['G'][mask]; y = d['Y'][mask, 0]; c = d['C'][mask]
    queries = [idx for idx in per_query_indices(g) if y[idx].sum() > 0 and len(idx) >= 5]
    rng = np.random.default_rng(0); picks = []
    for t in TARGETS:
        pool = [idx for idx in queries if g[idx[0]].startswith(t + ':')]
        for j in rng.choice(len(pool), size=min(count // len(TARGETS), len(pool)), replace=False):
            picks.append(pool[j])
    out = []
    for idx in picks:
        target, hs, basis, h = g[idx[0]].split(':')
        rank = lambda s: [str(c[idx][j]) for j in np.argsort(-s[idx], kind='stable')[:5]]
        positives = sorted(str(v) for v in c[idx][y[idx] > 0]); a, b = rank(old), rank(new)
        out.append({'query': {'target': target, 'hs6': hs, 'basis': mt(int(basis)), 'horizon_months': int(h)}, 'candidates': len(idx), 'rose': positives,
                    'top5_before': a, 'hits_before': sum(v in positives for v in a), 'first3_hits_before': sum(v in positives for v in a[:3]),
                    'top5_after': b, 'hits_after': sum(v in positives for v in b), 'first3_hits_after': sum(v in positives for v in b[:3])})
    return out


def stage_evaluate():
    d = load_dataset(); frozen = Frozen(d)
    X = d['X'][:, frozen.columns]; X25 = np.nan_to_num(d['X'][:, [d['features'].index(n) for n in V1_25F]])
    train, val, test, grid = d['masks']['train'], d['masks']['validation'], d['masks']['test'], d['grid']
    out = {'generated_at_utc': now(), 'version': VERSION, 'primary_metric': PRIMARY, 'target': 0.85, 'selection': frozen.name, 'kind': frozen.kind,
           'members': [dict(family=f, params=p, seed=s, weight=w) for f, p, s, w in frozen.members], 'training_rows': frozen.rows_label, 'features': frozen.features,
           'protocol': 'fit on the train split (label maturity <= 2018-12); validation and test scored with that fit; the test split was not read before this stage',
           'scope': 'generated observation-calibrated panel; ranking of future coupled trade growth, not real-world evasion accuracy'}
    # --- reference: the deployed configuration, same splits, same metric code
    t = time.perf_counter(); old = Model('xgb_rank', DEPLOYED).fit(X25, d, train & grid); old_fit = time.perf_counter() - t
    old_aux = aux_head(X25[train & grid], d['Y'][train & grid])
    # --- frozen configuration
    t = time.perf_counter(); models, weights = frozen.fit(d, X, train); new_fit = time.perf_counter() - t
    new_aux = aux_head(X[train & frozen.rows], d['Y'][train & frozen.rows])
    scores = {}
    for label, model_score, aux, M in (('before', lambda Z: old.score(Z), old_aux, X25), ('after', lambda Z: Frozen.score(models, weights, Z), new_aux, X)):
        out[label] = {}
        for split, mask in (('validation', val), ('test', test)):
            t = time.perf_counter(); s = model_score(M[mask]); seconds = time.perf_counter() - t
            scores[(label, split)] = s
            out[label][split] = split_metrics(d, mask, s, aux.predict(np.nan_to_num(M[mask]))) | {'predict_ms_per_1000_rows': round(seconds / mask.sum() * 1e6, 2)}
            out[label][split]['by_target'] = by(d, mask, s, 'T', TARGETS); out[label][split]['by_horizon'] = by(d, mask, s, 'H', (3, 6, 12, 24))
            out[label][split]['by_item'] = by(d, mask, s, 'HS', ITEMS)
        out[label]['train_seconds'] = round(old_fit if label == 'before' else new_fit, 1)
        out[label]['train_rows'] = int((train & grid).sum() if label == 'before' else (train & frozen.rows).sum())
        out[label]['n_features'] = M.shape[1]
    rng = np.random.default_rng(0)
    out['random_order'] = {split: split_metrics(d, mask, rng.random(mask.sum())) for split, mask in (('validation', val), ('test', test))}
    out['improvement'] = {split: {k: out['after'][split][k] - out['before'][split][k] for k in ('ndcg_at_5_pos', 'ndcg_at_5', 'precision_at_5', 'recall_at_5', 'map', 'mrr', 'roc_auc', 'pr_auc', 'log_ratio_mae')}
                          for split in ('validation', 'test')}
    out['target_status'] = 'achieved' if out['after']['test'][PRIMARY] >= 0.85 else 'not_achieved'
    print(json.dumps({s: {'before': round(out['before'][s][PRIMARY], 4), 'after': round(out['after'][s][PRIMARY], 4)} for s in ('validation', 'test')}), flush=True)
    # --- supplementary test protocols (same configuration, more recent training rows)
    extra = {}
    m2, w2 = frozen.fit(d, X, train | val_period(d))
    extra['refit_train_plus_validation'] = {'fit': 'every row with label maturity <= 2021-12', **split_metrics(d, test, Frozen.score(m2, w2, X[test]))}
    walk = np.full(test.sum(), np.nan); tb = d['B'][test]; refits = []
    for year in sorted(set((tb // 12).tolist())):
        cut = int(year) * 12 - 1  # last month of the previous year: every training label has matured before the year's first basis
        mk, wk = frozen.fit(d, X, d['A'] <= cut)
        part = (tb // 12) == year
        walk[part] = Frozen.score(mk, wk, X[test][part]); refits.append({'bases_in': 1990 + int(year), 'trained_on_maturity_through': mt(cut), 'rows': int(((d['A'] <= cut) & frozen.rows).sum())})
    extra['walk_forward_yearly'] = {'fit': 'refit at the start of each test year on every label matured by then', 'refits': refits, **split_metrics(d, test, walk),
                                    'by_target': by(d, test, walk, 'T', TARGETS), 'by_horizon': by(d, test, walk, 'H', (3, 6, 12, 24))}
    for name, block in reference_protocols(d).items():
        extra[name]['previous_model'] = block
    out['supplementary_test_protocols'] = extra
    print(json.dumps({k: {'after': round(v[PRIMARY], 4), 'before': round(v['previous_model'][PRIMARY], 4)} for k, v in extra.items()}), flush=True)
    # --- hold-outs: leave one group out of training, score that group's validation rows (seen-in-training score alongside)
    holdouts = {}
    seen = scores[('after', 'validation')]
    for field, values in (('T', TARGETS), ('HS', ITEMS), ('C', CANDIDATES)):
        block = {}
        for v in values:
            group = d[field] == v
            mh, wh = frozen.fit(d, X, train & ~group)
            # A target or an item owns whole queries: score its queries. A candidate is one row of every query: score all queries
            # with the model that never saw that candidate in training.
            part = np.ones(val.sum(), bool) if field == 'C' else group[val]
            a = ranking_metrics(d['Y'][val][part, 0], Frozen.score(mh, wh, X[val][part]), d['G'][val][part])
            b = ranking_metrics(d['Y'][val][part, 0], seen[part], d['G'][val][part])
            block[v] = {'held_out': a[PRIMARY], 'seen': b[PRIMARY], 'delta': a[PRIMARY] - b[PRIMARY]}
        holdouts[{'T': 'target', 'HS': 'item', 'C': 'candidate'}[field]] = {'groups': block, 'mean_held_out': float(np.mean([x['held_out'] for x in block.values()])),
                                                                            'mean_seen': float(np.mean([x['seen'] for x in block.values()])),
                                                                            'mean_delta': float(np.mean([x['delta'] for x in block.values()]))}
        print(field, {k: round(v, 4) for k, v in holdouts[{'T': 'target', 'HS': 'item', 'C': 'candidate'}[field]].items() if k != 'groups'}, flush=True)
    out['holdouts_validation'] = holdouts
    out['top5_examples_test'] = top5_examples(d, test, scores[('after', 'test')], scores[('before', 'test')])
    # --- importance on validation: drop of the primary metric when one column is shuffled (3 shuffles)
    Xv = X[val].copy(); base = ranking_metrics(d['Y'][val, 0], seen, d['G'][val])[PRIMARY]; drops = {}
    for j, name in enumerate(frozen.features):
        keep = Xv[:, j].copy(); values = []
        for _ in range(3):
            Xv[:, j] = keep[rng.permutation(len(keep))]
            values.append(base - ranking_metrics(d['Y'][val, 0], Frozen.score(models, weights, Xv), d['G'][val])[PRIMARY])
        Xv[:, j] = keep; drops[name] = float(np.mean(values))
    groups = {}
    for name, v in drops.items():
        groups[group_of_v2(name)] = groups.get(group_of_v2(name), 0.0) + v
    X25v = X25[val].copy(); base_old = ranking_metrics(d['Y'][val, 0], scores[('before', 'validation')], d['G'][val])[PRIMARY]; drops_old = {}
    for j, name in enumerate(V1_25F):
        keep = X25v[:, j].copy(); values = []
        for _ in range(3):
            X25v[:, j] = keep[rng.permutation(len(keep))]
            values.append(base_old - ranking_metrics(d['Y'][val, 0], old.score(X25v), d['G'][val])[PRIMARY])
        X25v[:, j] = keep; drops_old[name] = float(np.mean(values))
    groups_old = {}
    for name, v in drops_old.items():
        groups_old[group_of_v2(name)] = groups_old.get(group_of_v2(name), 0.0) + v
    out['importance_validation'] = {'method': f'drop of {PRIMARY} on validation when one column is shuffled (mean of 3)',
                                    'after': {'top': sorted(drops.items(), key=lambda kv: -kv[1])[:20], 'by_group': dict(sorted(groups.items(), key=lambda kv: -kv[1])), 'all': drops},
                                    'before': {'top': sorted(drops_old.items(), key=lambda kv: -kv[1])[:10], 'by_group': dict(sorted(groups_old.items(), key=lambda kv: -kv[1])), 'all': drops_old}}
    write('final-evaluation.json', out)
    return out


def reference_protocols(d):
    """The previously deployed configuration (25 features, XGBoost ranker, grid bases) under the two supplementary test protocols."""
    X25 = np.nan_to_num(d['X'][:, [d['features'].index(n) for n in V1_25F]]); test = d['masks']['test']; grid = d['grid']
    out = {}
    model = Model('xgb_rank', DEPLOYED).fit(X25, d, (d['masks']['train'] | val_period(d)) & grid)
    out['refit_train_plus_validation'] = split_metrics(d, test, model.score(X25[test]))
    walk = np.full(test.sum(), np.nan); tb = d['B'][test]
    for year in sorted(set((tb // 12).tolist())):
        model = Model('xgb_rank', DEPLOYED).fit(X25, d, (d['A'] <= int(year) * 12 - 1) & grid)
        part = (tb // 12) == year
        walk[part] = model.score(X25[test][part])
    out['walk_forward_yearly'] = split_metrics(d, test, walk)
    return out


def stage_reference():
    """Add the previous model's scores under the supplementary protocols to an existing final-evaluation.json."""
    d = load_dataset(); final = read('final-evaluation.json')
    for name, block in reference_protocols(d).items():
        final['supplementary_test_protocols'][name]['previous_model'] = block
        print(name, 'before', round(block[PRIMARY], 4), 'after', round(final['supplementary_test_protocols'][name][PRIMARY], 4), flush=True)
    write('final-evaluation.json', final)


def val_period(d):
    """Rows whose label matured during the validation period (any basis grid), for the train+validation refit."""
    return (d['A'] > mi('2018-12')) & (d['A'] <= mi('2021-12'))


def stage_export():
    """Production fit on every mature row and the hand-off file for scripts/deploy_ranker.py."""
    d = load_dataset(); frozen = Frozen(d)
    X = d['X'][:, frozen.columns]
    final = read('final-evaluation.json')
    models, weights = frozen.fit(d, X, np.ones(len(d['B']), bool))
    from scoring import NumpyTreeEnsemble
    rng = np.random.default_rng(0); check = np.sort(rng.choice(len(X), size=5000, replace=False))
    exported = []; parts = []
    for model, weight in zip(models, weights):
        tmp = Path(tempfile.mkdtemp()) / 'model.json'
        model.model.save_model(str(tmp))
        document = json.loads(tmp.read_text(encoding='utf-8'))
        # The raw-margin intercept is measured, not read: a classifier stores its base score as a probability.
        gap = model.score(X[check]) - NumpyTreeEnsemble.from_xgboost_json(document, base_margin=0.0).predict_margin(X[check])
        if float(np.ptp(gap)) > 1e-3:
            raise SystemExit(f'numpy trees differ from xgboost beyond a constant: spread {np.ptp(gap)}')
        exported.append({'weight': float(weight), 'family': model.kind, 'base_margin': float(np.median(gap)), 'xgboost_model': document})
        parts.append((NumpyTreeEnsemble.from_xgboost_json(document, base_margin=float(np.median(gap))), weight))
    reference = Frozen.score(models, weights, X[check])
    max_diff = float(np.max(np.abs(reference - NumpyTreeEnsemble.combine(parts).predict_margin(X[check]))))
    if max_diff > 1e-3:
        raise SystemExit(f'combined numpy ensemble differs from xgboost: max abs diff {max_diff}')
    meta = panel_metadata()
    rows = frozen.rows
    record = {'combination': f'v2f:{frozen.name}', 'version': VERSION, 'members': [dict(family=f, params=p, seed=s, weight=w) for f, p, s, w in frozen.members],
              'features': frozen.features, 'training_rows': int(rows.sum()), 'training_bases': frozen.rows_label,
              'validation_' + PRIMARY: final['after']['validation'][PRIMARY], 'test_' + PRIMARY: final['after']['test'][PRIMARY],
              'validation_ndcg_at_5': final['after']['validation']['ndcg_at_5'], 'test_ndcg_at_5': final['after']['test']['ndcg_at_5'],
              'reason': 'benchmark v14: feature set v2 and tuned XGBoost, selected on inner folds and validation (selection.json); test read once (final-evaluation.json)'}
    export = {'exported_at_utc': now(), 'schema': 'btl-ranker-export-v2', 'version': VERSION, 'candidate': frozen.name, 'feature_set': 'v2f', 'features': frozen.features,
              'panel': meta.get('dataset_version'), 'models': exported, 'check_rows': check.tolist(), 'check_margins': reference.tolist(),
              'numpy_export_check': {'rows': int(len(check)), 'max_abs_diff': max_diff, 'trees': sum(len(e.trees) for e, _ in parts)},
              'training_row_filter': frozen.rows_label, 'final_evaluation': {k: final[k] for k in ('primary_metric', 'target', 'target_status', 'before', 'after', 'improvement', 'random_order', 'supplementary_test_protocols', 'protocol', 'scope')},
              'deployment': record}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / 'ranker-export.json').write_text(json.dumps(export, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps({'version': VERSION, 'members': len(exported), 'training_rows': record['training_rows'], 'features': len(frozen.features), 'numpy_export_check': export['numpy_export_check']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['evaluate', 'export', 'all', 'reference'])
    args = parser.parse_args()
    if args.stage == 'reference':
        stage_reference()
    if args.stage in ('evaluate', 'all'):
        stage_evaluate()
    if args.stage in ('export', 'all'):
        stage_export()
