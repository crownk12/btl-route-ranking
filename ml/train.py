"""Train every registered candidate on the same matrix and splits, apply the pre-fixed
selection rule, and deploy the chosen scorer (track A of docs/PLAN_MODEL_AND_SYNTHETIC.md).

Selection rule (fixed before training, see SELECTION below): the reference is the
current Extra Trees. A challenger is adopted only if its validation NDCG@5 beats the
reference by at least the margin, its test NDCG@5 is not lower than the reference by
more than the tolerance, and its mean score on thin-coverage validation rows is not
higher than the reference's. Ties go to the simpler candidate. Hybrids must also beat
their best component. Everything is recorded in artifacts/integration/model-selection.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import btl_paths  # noqa: E402,F401

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, time, datetime, shutil
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, ndcg_score
from common import *
from scoring import Sk, TwoHead, RuleScorer, GateHybrid, BlendHybrid, StackHybrid, logistic_two_head, COVERAGE

REFERENCE = 'extra_trees'
SELECTION = dict(metric='validation model_ndcg_at_5', margin=0.02, test_tolerance=0.01,
                 thin_coverage='validation rows with coverage < 0.6: mean predicted score must not exceed the reference',
                 simplicity_order=['logistic', 'rule', 'extra_trees', 'random_forest', 'hgb', 'two_head', 'gate_hybrid', 'blend_hybrid', 'stack_hybrid'],
                 hybrid_rule='a hybrid is eligible only if it beats every component on the validation metric',
                 inner_split='train rows with label maturity <= 2015-12 fit the components; train rows with basis >= 2016-01 set hybrid weights/meta-model')
LEGACY_VERSIONS = {'extra_trees': 'extra-trees-v2-china'}  # names used while the panel had no model_tag


def version_for(name, panel_tag):
    if not panel_tag:
        return LEGACY_VERSIONS.get(name, name.replace('_', '-') + '-v1-china')
    return name.replace('_', '-') + '-' + panel_tag
SEED = 42


def et():
    return Sk(ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=SEED, n_jobs=2), 'extra_trees')


def hgb():
    return TwoHead(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED),
                   MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)),
                   'hgb')


def candidates():
    """name -> factory of an unfitted scorer; hybrids are built from fresh components."""
    return {
        'extra_trees': et,
        'hgb': hgb,
        'random_forest': lambda: Sk(RandomForestRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=SEED, n_jobs=2), 'random_forest'),
        'logistic': lambda: logistic_two_head(SEED),
        'rule': RuleScorer,
        'gate_hybrid': lambda: GateHybrid(et()),
        'blend_hybrid': lambda: BlendHybrid([RuleScorer(), et(), hgb()]),
        'stack_hybrid': lambda: StackHybrid([RuleScorer(), et(), hgb()]),
        'two_head': lambda: TwoHead(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED),
                                    ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=SEED, n_jobs=2), 'two_head'),
    }


COMPONENTS = {'gate_hybrid': ['extra_trees'], 'blend_hybrid': ['rule', 'extra_trees', 'hgb'], 'stack_hybrid': ['rule', 'extra_trees', 'hgb']}


def evaluate(y, p, groups, baseline):
    out = {'rows': len(y), 'positive_rate': float(y[:, 0].mean()), 'roc_auc': float(roc_auc_score(y[:, 0], p[:, 0])),
           'brier': float(brier_score_loss(y[:, 0], np.clip(p[:, 0], 0, 1))), 'log_ratio_mae': float(mean_absolute_error(y[:, 1:], p[:, 1:]))}
    for name, score in [('model', p[:, 0]), ('baseline', baseline)]:
        ndcg = []; precision = []
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() < 5: continue
            yy = y[mask, 0]; ss = score[mask]; precision.append(yy[np.argsort(-ss)[:5]].mean()); ndcg.append(ndcg_score([yy], [ss], k=5))
        out[name + '_precision_at_5'] = float(np.mean(precision)); out[name + '_ndcg_at_5'] = float(np.mean(ndcg))
    return out


def build_matrix(p):
    X = []; Y = []; B = []; A = []; G = []
    # Same sampled horizons per group so ranking is evaluated across all nine candidates.
    for key in p.rows:
        final = {r['month']: r for r in p.rows[key] if r['revision'] == 1}
        for basis in range(60, p.end - 5, 4):
            for h in [3, 6, 12, 24, 60, 75]:
                endpoint = basis + h + 2
                if endpoint + 5 > p.end: continue
                x, cur, hist, events = p.features(key, basis, h)
                future = []
                for col in ['inbound', 'outbound']:
                    v = [final[m][col] for m in range(basis + h, basis + h + 3) if final[m][col] is not None]
                    future.append(np.mean(v) if v else np.nan)
                if min(cur) <= 0 or not np.isfinite(future).all(): continue
                ratios = np.array(future) / cur
                label = float(ratios[0] > 1.15 and ratios[1] > 1.20)
                X.append(x); Y.append([label, *np.clip(np.log(ratios), -2, 2)]); B.append(basis); A.append(endpoint + 5); G.append(f'{key[0]}:{key[1]}:{basis}:{h}')
    X = np.array(X); Y = np.array(Y); B = np.array(B); A = np.array(A); G = np.array(G)
    masks = {'train': A <= mi('2018-12'), 'validation': (B >= mi('2019-01')) & (A <= mi('2021-12')), 'test': (B >= mi('2022-01')) & (A <= p.end)}
    fit_part = masks['train'] & (A <= mi('2015-12')); hold_part = masks['train'] & (B >= mi('2016-01'))
    inner = (X[fit_part], Y[fit_part], X[hold_part], Y[hold_part], G[hold_part])
    return X, Y, B, A, G, masks, inner, {'inner_fit': int(fit_part.sum()), 'inner_holdout': int(hold_part.sum())}


def thin_coverage_mean(X, p, mask):
    thin = mask & (X[:, COVERAGE] < 0.6)
    return float(np.mean(np.clip(p[thin, 0], 0, 1))) if thin.any() else None, int(thin.sum())


def select(results):
    ref = results[REFERENCE]
    order = {n: i for i, n in enumerate(SELECTION['simplicity_order'])}
    verdicts = {}
    for name, r in results.items():
        reasons = []
        gain = r['validation']['model_ndcg_at_5'] - ref['validation']['model_ndcg_at_5']
        if name != REFERENCE:
            if gain < SELECTION['margin']: reasons.append(f'validation NDCG@5 gain {gain:+.4f} < margin {SELECTION["margin"]}')
            if r['test']['model_ndcg_at_5'] < ref['test']['model_ndcg_at_5'] - SELECTION['test_tolerance']: reasons.append('test NDCG@5 below reference beyond tolerance')
            if r['thin_coverage_mean_score'] is not None and ref['thin_coverage_mean_score'] is not None and r['thin_coverage_mean_score'] > ref['thin_coverage_mean_score'] + 1e-9:
                reasons.append('thin-coverage mean score above reference')
            for comp in COMPONENTS.get(name, []):
                if r['validation']['model_ndcg_at_5'] <= results[comp]['validation']['model_ndcg_at_5'] + 1e-12:
                    reasons.append(f'does not beat component {comp}')
        verdicts[name] = {'eligible': not reasons, 'validation_gain': gain, 'reasons': reasons}
    eligible = [n for n, v in verdicts.items() if v['eligible'] and n != REFERENCE]
    if not eligible:
        return REFERENCE, verdicts, 'no challenger met the pre-fixed rule; reference retained'
    best = max(results[n]['validation']['model_ndcg_at_5'] for n in eligible)
    tied = [n for n in eligible if abs(results[n]['validation']['model_ndcg_at_5'] - best) < 0.005]
    chosen = min(tied, key=lambda n: order.get(n, 99))
    return chosen, verdicts, f'{chosen} adopted: best validation NDCG@5 among eligible challengers (ties within 0.005 resolved by simplicity)'


def train():
    p = Panel(); beg = time.time()
    X, Y, B, A, G, masks, inner, inner_counts = build_matrix(p)
    results = {}; fitted = {}
    for name, factory in candidates().items():
        t0 = time.time(); scorer = factory().fit(X[masks['train']], Y[masks['train']], inner); fit_seconds = time.time() - t0
        entry = {'fit_seconds': round(fit_seconds, 1)}
        for split in ['validation', 'test']:
            mask = masks[split]; pred = scorer.predict(X[mask])
            entry[split] = evaluate(Y[mask], pred, G[mask], X[mask, 4]); entry[split]['basis_range'] = [mt(int(B[mask].min())), mt(int(B[mask].max()))]
        full_pred = scorer.predict(X[masks['validation']]); entry['thin_coverage_mean_score'], entry['thin_coverage_rows'] = thin_coverage_mean(X[masks['validation']], full_pred, np.ones(masks['validation'].sum(), bool))
        t1 = time.perf_counter(); [scorer.predict(X[:9]) for _ in range(20)]; entry['predict_ms_9_rows'] = round((time.perf_counter() - t1) / 20 * 1000, 3)
        if getattr(scorer, 'weights', None) is not None: entry['blend_weights'] = dict(zip(COMPONENTS[name], map(float, scorer.weights)))
        if getattr(scorer, 'meta', None) is not None and hasattr(scorer.meta, 'coef_'): entry['meta_coefficients'] = dict(zip(COMPONENTS[name], map(float, scorer.meta.coef_[0])))
        results[name] = entry; fitted[name] = scorer
        print(json.dumps({'candidate': name, 'validation_ndcg5': round(entry['validation']['model_ndcg_at_5'], 4), 'test_ndcg5': round(entry['test']['model_ndcg_at_5'], 4), 'fit_seconds': entry['fit_seconds']}), flush=True)
    chosen, verdicts, note = select(results)
    with connect() as db:
        panel_tag = (db.execute("select value from metadata where key='model_tag'").fetchone() or [None])[0]
        dataset_version = (db.execute("select value from metadata where key='dataset_version'").fetchone() or [None])[0]
    version = version_for(chosen, panel_tag)
    selection = {'schemaVersion': 1, 'selectedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reference': REFERENCE, 'rule': SELECTION,
                 'inner_split_counts': inner_counts, 'split_counts': {k: int(v.sum()) for k, v in masks.items()}, 'candidates': results, 'verdicts': verdicts,
                 'chosen': chosen, 'version': version, 'note': note, 'panel': {'dataset_version': dataset_version, 'model_tag': panel_tag},
                 'scope': 'generated trade panel; selection inside the demo panel, no real-world sanctions-evasion performance claim'}
    ROOT.joinpath('artifacts/integration').mkdir(parents=True, exist_ok=True)
    (ROOT / 'artifacts/integration/model-selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    scorer = fitted[chosen]; models = ROOT / 'models'; models.mkdir(exist_ok=True)
    previous = models / 'model.joblib'
    deploy = '--deploy' in sys.argv
    if previous.exists():
        old = joblib.load(previous)
        if old.get('deployed_by') == 'benchmark.export' and not deploy:
            print(json.dumps({'chosen': chosen, 'version': version, 'note': note, 'deployment': 'kept ' + str(old.get('version', '')) + ' (benchmark.export); pass --deploy to replace it'}, ensure_ascii=False))
            return
        if old.get('version') != version:
            archive = models / 'archive' / old.get('version', 'unknown'); archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, archive / 'model.joblib')
            if (models / 'metrics.json').exists(): shutil.copy2(models / 'metrics.json', archive / 'metrics.json')
    joblib.dump(scorer, models / 'evaluation.joblib', compress=3)
    r = results[chosen]; out = {'validation': r['validation'], 'test': r['test'], 'split_counts': selection['split_counts'], 'features': FEATURES}
    importance = getattr(scorer, 'feature_importances_', None)
    out['feature_importance'] = dict(zip(FEATURES, map(float, importance))) if importance is not None else None
    out['task'] = 'future 3-month-average import >15% and outbound >20% growth vs latest available 3-month average; not official designation labels'
    out['evaluation_scope'] = 'generated trade panel; no validated real-world sanctions-evasion performance'
    out['leakage_controls'] = ['features use available_month <= basis', 'labels mature after horizon+2+5 months', 'train label maturity <= 2018-12; validation basis >= 2019-01; test basis >= 2022-01', 'HS and country names excluded from model inputs', 'hybrid weights and meta-models fitted on an inner train holdout, never on validation or test']
    out['model_selection'] = {'chosen': chosen, 'version': version, 'note': note, 'panel': dataset_version, 'candidates': {n: {'validation_ndcg_at_5': v['validation']['model_ndcg_at_5'], 'test_ndcg_at_5': v['test']['model_ndcg_at_5']} for n, v in results.items()}}
    out['per_target'] = {}
    for t in TARGETS:
        tm = np.array([g.startswith(t + ':') for g in G]); out['per_target'][t] = {'production_rows': int(tm.sum())}
        for split in ['validation', 'test']:
            mask = tm & masks[split]; out['per_target'][t][split] = evaluate(Y[mask], scorer.predict(X[mask]), G[mask], X[mask, 4])
    # Deployment scorer is refit on all rows; hybrids keep the inner-split weights.
    deployed = candidates()[chosen]().fit(X, Y, inner)
    joblib.dump({'model': deployed, 'features': FEATURES, 'trained_through': mt(p.end), 'training_rows': len(X), 'version': version, 'candidate': chosen, 'panel': dataset_version}, previous, compress=3)
    out['production_training_rows'] = len(X); out['elapsed_seconds'] = round(time.time() - beg, 1)
    (models / 'metrics.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'chosen': chosen, 'version': version, 'note': note, 'elapsed_seconds': out['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    train()
