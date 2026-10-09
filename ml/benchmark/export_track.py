"""Export-surge track: a second, separate ranking task beside the coupled-growth one (user decision, 2026-10-05).

  coupled growth (benchmark v14, unchanged)   imports +15 % AND exports +20 %, horizons 3-75 months
  export surge (this module)                  the candidate's exports to the target +20 %, horizons 1 and 3 months

The coupled label keeps the transit signature (goods come in, then go on to the target) but asks for two noisy conditions at
once over long horizons; it tops out near 0.75 on the primary metric. The export leg alone, over the next months, is the
more predictable question and the one closest to the object of interest. The two are reported side by side and never mixed:
separate label, separate model, separate metric record.

Label: mean of the candidate's exports to the target over months b+h … b+h+2 above the mean of its latest three available
months by more than 20 %, for h = 1 and h = 3. Queries, features (FeatureBuilderV2.block), time splits, inner folds and the
primary metric (NDCG@5 over queries with a positive) are those of benchmark v14.

v2 (2026-10-05). On the inner folds and on validation the v1 model lost its points in the years after a policy event of
the target (Russia after 2014, North Korea after 2017), not elsewhere; an event lifts every route of the target at once
and the lift stays in the 60-month history that the cycle features read. v2 adds the export leg with the target-wide
common movement removed (FeatureBuilderV2(common=True), 35 columns). Inner folds: tied with v1; validation: +0.007
(three seeds, paired bootstrap over queries 95 % interval +0.002 … +0.012). The selection rule is unchanged and decides
between the two feature sets. The test split had been read once for v1; the v2 evaluation is its second reading and is
recorded as such, next to the v1 figures.

Stages
  build     monthly training bases, grid validation and test bases, horizons 1 and 3 -> cache/dataset-export.npz
  select    a few frozen candidates scored on the inner folds, ties broken on validation -> export-selection.json
  evaluate  train split -> validation and test (test read once), random and persistence references, yearly walk-forward
            -> export-evaluation.json
  deploy    fit on every mature row -> models/export-model.joblib (NumPy trees only) and models/export-metrics.json

Run: .venv-bench/Scripts/python -m benchmark.export_track build|select|evaluate|deploy|all   (in ml/)
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmark.lab import load_dataset, read, write, now, panel_metadata, CACHE
from common import ROOT, TARGETS, mi, mt
from benchmark.experiments import Model, folds, run_folds, score_split, log, PRIMARY
from benchmark.metrics import ranking_metrics

HORIZONS = (1, 3)
THRESHOLD = float(np.log(1.20))
VERSION = 'xgb-export-surge-v2-17items'
DATASET = 'dataset-export-v2.npz'
TIE = 0.002
KEYS = ('ndcg_at_5_pos', 'ndcg_at_5', 'ndcg_at_10_pos', 'precision_at_5', 'recall_at_5', 'hit_rate_at_5', 'map', 'mrr', 'roc_auc', 'pr_auc', 'queries', 'queries_with_positive', 'rows', 'positive_rate')


def _build_target(target):
    from benchmark.features_v2 import FeatureBuilderV2, build_dataset_v2
    d = build_dataset_v2(FeatureBuilderV2(common=True), basis_step=1, horizons=HORIZONS, targets=[target])
    return {k: d[k] for k in ('X', 'Y', 'R', 'B', 'A', 'G', 'H', 'T', 'HS', 'C', 'grid')} | {'features': d['features'], 'panel_end': d['panel_end'], 'masks': d['masks'], 'inner': d['inner']}


def stage_build():
    from joblib import Parallel, delayed
    t0 = time.time()
    parts = Parallel(n_jobs=len(TARGETS))(delayed(_build_target)(t) for t in TARGETS)
    d = {k: np.concatenate([p[k] for p in parts]) for k in ('X', 'Y', 'R', 'B', 'A', 'G', 'H', 'T', 'HS', 'C', 'grid')}
    masks = {k: np.concatenate([p['masks'][k] for p in parts]) for k in ('train', 'validation', 'test')}
    inner = {k: np.concatenate([p['inner'][k] for p in parts]) for k in ('fit', 'holdout')}
    np.savez_compressed(CACHE / DATASET, **d, features=np.array(parts[0]['features']), panel_end=int(parts[0]['panel_end']),
                        **{f'mask_{k}': v for k, v in masks.items()}, **{f'inner_{k}': v for k, v in inner.items()})
    y = d['R'][:, 1] > THRESHOLD
    print(json.dumps({k: {'rows': int(m.sum()), 'positive_rate': round(float(y[m].mean()), 4)} for k, m in masks.items()}), 'seconds', round(time.time() - t0, 1))


def load():
    """The export dataset with the export-surge label in Y[:, 0] (the coupled label moves to `coupled`)."""
    d = load_dataset(CACHE / DATASET)
    d['coupled'] = d['Y'][:, 0].copy()
    d['Y'] = d['Y'].copy(); d['Y'][:, 0] = (d['R'][:, 1] > THRESHOLD).astype(float)
    return d


def rows_of(d, label):
    return d['grid'] if label == 'grid' else (d['B'] % 2 == 0) if label == 'step2' else np.ones(len(d['B']), bool)


def stage_select():
    import optuna
    from benchmark.exp_round5 import SPACES
    from benchmark.features_v2 import COMMON_NAMES
    d = load(); names = d['features']
    v14 = read('selection.json')
    sets = {'v14_80': v14['features'], 'all_125': [n for n in names if n not in COMMON_NAMES], 'v14_80_common': v14['features'] + COMMON_NAMES}
    tuned = v14['chosen_record']['params']
    candidates = {}

    def add(name, feats, params, rows):
        X = d['X'][:, [names.index(n) for n in sets[feats]]]
        cv = run_folds(d, X, lambda: Model('xgb_clf', params), rows_of(d, rows))
        train = d['masks']['train'] & rows_of(d, rows); val = d['masks']['validation']
        model = Model('xgb_clf', params).fit(X, d, train)
        v = score_split(d, val, model.score(X[val]))[PRIMARY]
        candidates[name] = dict(features=feats, params=params, rows=rows, cv=cv['cv'], cv_min=cv['cv_min'], validation=v)
        log({'round': 'export', 'name': name, 'n_features': len(sets[feats]), 'cv_' + PRIMARY: cv['cv'], 'validation_' + PRIMARY: v, 'rows': rows})
        print(f"{name:26s} cv {cv['cv']:.4f}  val {v:.4f}", flush=True)

    add('v14_setup_grid', 'v14_80', tuned, 'grid')
    add('v14_setup_step2', 'v14_80', tuned, 'step2')
    add('v14_setup_monthly', 'v14_80', tuned, 'monthly')
    add('all_features_grid', 'all_125', tuned, 'grid')
    # a short search of its own: the horizons and the label differ from the task the v14 settings were tuned on
    X = d['X'][:, [names.index(n) for n in sets['v14_80']]]; trials = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = SPACES['xgb_clf'](trial)
        cv = run_folds(d, X, lambda: Model('xgb_clf', params), d['grid'])['cv']
        trials.append({'params': params, 'cv': cv}); print(f'trial {trial.number:2d} cv {cv:.4f}', flush=True)
        return cv
    optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0)).optimize(objective, n_trials=20)
    best = max(trials, key=lambda t: t['cv'])
    add('searched_grid', 'v14_80', best['params'], 'grid')
    add('searched_step2', 'v14_80', best['params'], 'step2')
    # v2: the same settings with the common-movement columns (listed last: on a full tie the smaller feature set stays)
    add('searched_grid_common', 'v14_80_common', best['params'], 'grid')
    add('searched_step2_common', 'v14_80_common', best['params'], 'step2')
    top = max(c['cv'] for c in candidates.values())
    tied = [k for k, c in candidates.items() if c['cv'] >= top - TIE]
    best_val = max(candidates[k]['validation'] for k in tied)
    final = [k for k in tied if candidates[k]['validation'] >= best_val - TIE]
    chosen = final[0]  # candidates were added simplest first
    write('export-selection.json', {'generated_at_utc': now(), 'metric': PRIMARY, 'label': 'export ratio > 1.20, horizons 1 and 3',
                                    'rule': f'candidates within {TIE} of the best inner-fold score are tied; among them the best validation score wins; a remaining tie ({TIE}) goes to the candidate listed first (simplest first)',
                                    'candidates': candidates, 'search_trials': trials, 'tied_on_folds': tied, 'tied_on_validation': final, 'chosen': chosen,
                                    'features': sets[candidates[chosen]['features']]})
    print('tied', tied, '| chosen', chosen)


def _frozen(d):
    sel = read('export-selection.json'); c = sel['candidates'][sel['chosen']]
    X = d['X'][:, [d['features'].index(n) for n in sel['features']]]
    return sel, c, X


def _metrics(d, mask, score):
    m = score_split(d, mask, score)
    return {k: m[k] for k in KEYS}


def stage_evaluate():
    d = load(); sel, c, X = _frozen(d)
    train, val, test = d['masks']['train'], d['masks']['validation'], d['masks']['test']
    rows = rows_of(d, c['rows']); rng = np.random.default_rng(0)
    try:
        earlier = read('export-evaluation.json')
    except FileNotFoundError:
        earlier = None
    if earlier and earlier.get('version') == VERSION:
        earlier = earlier.get('previous_version')  # a re-run of this version keeps the record of the one before it
    elif earlier:
        earlier = {'version': earlier['version'], 'selection': earlier['selection'], 'features': len(earlier['features']), 'generated_at_utc': earlier['generated_at_utc'],
                   'validation': earlier['validation'][PRIMARY], 'test': earlier['test'][PRIMARY], 'walk_forward_yearly': earlier['walk_forward_yearly'][PRIMARY]}
    t = time.perf_counter(); model = Model('xgb_clf', c['params']).fit(X, d, train & rows); fit_seconds = time.perf_counter() - t
    out = {'generated_at_utc': now(), 'version': VERSION, 'task': 'export surge: exports to the target +20 % over months b+h … b+h+2 against the latest three available months, h = 1 and 3',
           'primary_metric': PRIMARY, 'target': 0.85, 'selection': sel['chosen'], 'params': c['params'], 'training_rows': c['rows'], 'features': sel['features'],
           'protocol': 'fit on the train split (label maturity <= 2018-12); features, settings and training rows fixed on the inner folds and validation before this stage',
           'test_readings': ('second reading of the test split of this task: it was read once for ' + earlier['version'] + '; the change since (common-movement columns) was found and chosen on the inner folds and validation only'
                             if earlier else 'first reading of the test split of this task'),
           'previous_version': earlier,
           'scope': 'generated observation-calibrated panel; ranking of future export growth on the candidate-to-target leg, not real-world evasion accuracy',
           'train_seconds': round(fit_seconds, 1), 'train_rows': int((train & rows).sum())}
    persistence = d['X'][:, d['features'].index('export_growth')]
    for split, mask in (('validation', val), ('test', test)):
        s = model.score(X[mask])
        block = _metrics(d, mask, s)
        block['by_horizon'] = {str(h): _metrics(d, mask & (d['H'] == h), s[d['H'][mask] == h]) for h in HORIZONS}
        block['by_target'] = {t_: {k: v for k, v in _metrics(d, mask & (d['T'] == t_), s[d['T'][mask] == t_]).items() if k in ('ndcg_at_5_pos', 'ndcg_at_5', 'queries', 'queries_with_positive')} for t_ in TARGETS}
        block['random_order'] = _metrics(d, mask, rng.random(mask.sum()))[PRIMARY]
        block['persistence_export_growth'] = _metrics(d, mask, persistence[mask])[PRIMARY]
        coupled = ranking_metrics(d['coupled'][mask], s, d['G'][mask])[PRIMARY]
        block['same_scores_on_the_coupled_label'] = coupled  # how the export-surge order does on the other task (reference only)
        out[split] = block
        print(split, {k: round(block[k], 4) for k in ('ndcg_at_5_pos', 'random_order', 'persistence_export_growth', 'precision_at_5', 'recall_at_5', 'pr_auc')},
              {h: round(v['ndcg_at_5_pos'], 4) for h, v in block['by_horizon'].items()}, flush=True)
    out['target_status'] = 'achieved' if out['test'][PRIMARY] >= 0.85 else 'not_achieved'
    walk = np.full(test.sum(), np.nan); tb = d['B'][test]; refits = []
    for year in sorted(set((tb // 12).tolist())):
        cut = int(year) * 12 - 1
        m = Model('xgb_clf', c['params']).fit(X, d, (d['A'] <= cut) & rows)
        part = (tb // 12) == year; walk[part] = m.score(X[test][part])
        refits.append({'bases_in': 1990 + int(year), 'trained_on_maturity_through': mt(cut)})
    out['walk_forward_yearly'] = {'fit': 'refit at the start of each test year on every label matured by then', 'refits': refits, **_metrics(d, test, walk),
                                  'by_horizon': {str(h): _metrics(d, test & (d['H'] == h), walk[d['H'][test] == h])[PRIMARY] for h in HORIZONS}}
    print('walk-forward', round(out['walk_forward_yearly'][PRIMARY], 4), out['walk_forward_yearly']['by_horizon'], '| target', out['target_status'], flush=True)
    write('export-evaluation.json', out)


def stage_deploy():
    """Production fit on every mature row; the pack holds NumPy trees only, so the demo runtime needs nothing new."""
    import joblib
    from scoring import NumpyTreeEnsemble
    d = load(); sel, c, X = _frozen(d); final = read('export-evaluation.json')
    rows = rows_of(d, c['rows'])
    model = Model('xgb_clf', c['params']).fit(X, d, rows)
    tmp = Path(tempfile.mkdtemp()) / 'model.json'; model.model.save_model(str(tmp))
    document = json.loads(tmp.read_text(encoding='utf-8'))
    rng = np.random.default_rng(0); check = np.sort(rng.choice(len(X), size=5000, replace=False))
    reference = model.score(X[check])
    gap = reference - NumpyTreeEnsemble.from_xgboost_json(document, base_margin=0.0).predict_margin(X[check])
    ensemble = NumpyTreeEnsemble.from_xgboost_json(document, base_margin=float(np.median(gap)))
    max_diff = float(np.max(np.abs(reference - ensemble.predict_margin(X[check]))))
    if max_diff > 1e-3:
        raise SystemExit(f'numpy trees differ from xgboost: {max_diff}')
    meta = panel_metadata()
    pack = {'model': ensemble, 'features': sel['features'], 'horizons': list(HORIZONS), 'display_horizon': 1, 'trained_through': mt(int(d['panel_end'])), 'training_rows': int(rows.sum()),
            'version': VERSION, 'panel': meta.get('dataset_version'), 'score_kind': 'margin', 'label': 'exports to the target +20 % over the next months (h = 1: months b+1 … b+3)',
            'deployed_at_utc': now(), 'numpy_export_check': {'rows': int(len(check)), 'max_abs_diff': max_diff, 'trees': len(ensemble.trees)}}
    joblib.dump(pack, ROOT / 'models/export-model.joblib', compress=3)
    metrics = {'version': VERSION, 'task': final['task'], 'evaluation_scope': final['scope'], 'protocol': final['protocol'], 'features': sel['features'],
               'primary_metric': {'name': 'NDCG@5 over queries with at least one positive', 'target': final['target'], 'target_status': final['target_status'],
                                  'actual_validation': final['validation'][PRIMARY], 'actual_test': final['test'][PRIMARY],
                                  'random_validation': final['validation']['random_order'], 'random_test': final['test']['random_order'],
                                  'walk_forward_test': final['walk_forward_yearly'][PRIMARY]},
               'validation': final['validation'], 'test': final['test'], 'walk_forward_yearly': final['walk_forward_yearly'],
               'test_readings': final['test_readings'], 'previous_version': final['previous_version'],
               'production_training_rows': int(rows.sum()), 'deployed_at_utc': now(), 'benchmark': 'artifacts/model-benchmark-v14/export-evaluation.json',
               'note': 'a separate task from models/metrics.json (coupled growth); the two scores are never mixed'}
    (ROOT / 'models/export-metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding='utf-8', newline='\n')
    print(json.dumps({'version': VERSION, 'trees': len(ensemble.trees), 'max_abs_diff': max_diff, 'training_rows': int(rows.sum()), 'features': len(sel['features'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'select', 'evaluate', 'deploy', 'all'])
    args = parser.parse_args()
    for stage in (['build', 'select', 'evaluate', 'deploy'] if args.stage == 'all' else [args.stage]):
        globals()['stage_' + stage]()
