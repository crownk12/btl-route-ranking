"""Benchmark v14, round 6: freeze the configuration to deploy.

Candidates (all on the pruned feature set, tuned settings of round 5):
  single     the tuned XGBoost classifier, the tuned XGBoost ranker, the tuned LightGBM classifier (reference only: the demo
             runtime evaluates XGBoost trees, so LightGBM would need a margin of 0.005 on folds and validation to justify a converter)
  density    the best XGBoost single with training bases every 4 months (grid), every 2 months, every month
  seeds      the mean margin of three seeds
  blend      a fixed linear mix of the classifier and ranker margins (each divided by its standard deviation on the fit
             rows); the weight is chosen on the inner folds

Rule, fixed before this round was run: candidates within 0.002 of the best inner-fold score are tied; among the tied the best
validation score wins; a remaining tie (0.002) goes to the simpler candidate (single < density < seeds < blend).
The test split is not read. Writes selection.json.
Run: .venv-bench/Scripts/python -m benchmark.exp_round6   (in ml/)
"""
import numpy as np

from benchmark.lab import load_dataset, read, write, now
from benchmark.experiments import Model, folds, score_split, log, PRIMARY

TIE = 0.002
SIMPLICITY = ['single', 'density', 'seeds', 'blend']


def margins(d, X, family, params, seed, rows):
    """Scores of one model on the three folds and on validation: {split: (score-row mask, scores, sd of the scores on the fit rows)}."""
    out = {}
    splits = folds(d, rows) + [('validation', d['masks']['train'] if rows is None else d['masks']['train'] & rows, d['masks']['validation'])]
    for name, fit, score in splits:
        model = Model(family, params, seed=seed).fit(X, d, fit)
        sample = np.flatnonzero(fit)[::7]
        out[name] = (score, model.score(X[score]), float(np.std(model.score(X[sample]))))
    return out


def summarize(d, runs):
    """runs: list of {split: (mask, scores)} to average (already on one scale). Returns (cv, validation, per-split scores)."""
    per = {}
    for name in runs[0]:
        mask = runs[0][name][0]
        per[name] = score_split(d, mask, np.mean([r[name][1] for r in runs], axis=0))[PRIMARY]
    cv = float(np.mean([v for k, v in per.items() if k.startswith('fold')]))
    return cv, per['validation'], per


def main():
    d = load_dataset()
    feats = read('feature-set.json')['features']
    X = d['X'][:, [d['features'].index(n) for n in feats]]; grid = d['grid']
    best = {f: read(f'tuning-{f}.json')['best']['params'] for f in ('xgb_clf', 'xgb_rank', 'lgbm_clf')}
    candidates = {}

    def add(name, kind, cv, val, per, **extra):
        candidates[name] = dict(kind=kind, cv=cv, validation=val, scores=per, **extra)
        log({'round': 6, 'name': name, 'kind': kind, 'n_features': len(feats), 'cv_' + PRIMARY: cv, 'validation_' + PRIMARY: val, **extra})
        print(f'{name:34s} cv {cv:.4f}  val {val:.4f}', flush=True)

    base = {}
    for family in ('xgb_clf', 'xgb_rank', 'lgbm_clf'):
        base[family] = margins(d, X, family, best[family], 42, grid)
        add(f'single_{family}', 'single', *summarize(d, [base[family]]), family=family, params=best[family], rows='grid', deployable=family != 'lgbm_clf')
    top = max(('xgb_clf', 'xgb_rank'), key=lambda f: candidates[f'single_{f}']['cv'])
    for label, rows in (('step2', d['B'] % 2 == 0), ('monthly', None)):
        add(f'density_{top}_{label}', 'density', *summarize(d, [margins(d, X, top, best[top], 42, rows)]), family=top, params=best[top], rows=label, deployable=True)
    seeds = [base[top]] + [margins(d, X, top, best[top], s, grid) for s in (7, 2024)]
    add(f'seeds_{top}_x3', 'seeds', *summarize(d, seeds), family=top, params=best[top], rows='grid', seeds=[42, 7, 2024], deployable=True)
    # Linear blend of the two XGBoost margins, each scaled by its spread on the fit rows; the weight is set on the folds only.
    scaled = {f: {k: (m, s / sd) for k, (m, s, sd) in base[f].items()} for f in ('xgb_clf', 'xgb_rank')}
    weights = {}
    for w in np.round(np.arange(0.1, 1.0, 0.1), 1):
        mix = {k: (scaled['xgb_clf'][k][0], w * scaled['xgb_clf'][k][1] + (1 - w) * scaled['xgb_rank'][k][1]) for k in scaled['xgb_clf']}
        weights[float(w)] = summarize(d, [mix])
    w = max(weights, key=lambda k: weights[k][0])
    add(f'blend_clf{w:.1f}_rank{1 - w:.1f}', 'blend', *weights[w], weight_classifier=w, weights_cv={str(k): v[0] for k, v in weights.items()}, rows='grid', deployable=True)

    eligible = {k: v for k, v in candidates.items() if v['deployable']}
    best_cv = max(v['cv'] for v in eligible.values())
    tied = [k for k, v in eligible.items() if v['cv'] >= best_cv - TIE]
    best_val = max(eligible[k]['validation'] for k in tied)
    final = [k for k in tied if eligible[k]['validation'] >= best_val - TIE]
    chosen = min(final, key=lambda k: (SIMPLICITY.index(eligible[k]['kind']), -eligible[k]['validation']))
    lgbm = candidates['single_lgbm_clf']
    write('selection.json', {'generated_at_utc': now(), 'metric': PRIMARY, 'rule': __doc__.split('Rule, fixed before this round was run: ')[1].split('\n\n')[0].replace('\n', ' ').split(' The test split')[0],
                             'features': feats, 'tuned': best, 'candidates': candidates, 'tied_on_folds': tied, 'tied_on_validation': final, 'chosen': chosen, 'chosen_record': candidates[chosen],
                             'lightgbm_reference': {'cv': lgbm['cv'], 'validation': lgbm['validation'],
                                                    'beats_chosen_by_0.005_on_both': bool(lgbm['cv'] >= candidates[chosen]['cv'] + 0.005 and lgbm['validation'] >= candidates[chosen]['validation'] + 0.005)}})
    print('tied on folds', tied, '| tied on validation', final, '| chosen', chosen)


if __name__ == '__main__':
    main()
