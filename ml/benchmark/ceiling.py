"""Benchmark v14 ceiling diagnostic: how high can the primary metric go on this panel?

The panel is generated (ml/build_db.py), so its hidden state can be recovered by replaying the generator
(benchmark.replay; the replay is checked cell for cell against data/test.db first). The label depends on growth episodes:
a route climbs for 6-12 months, falls back over the same time, and the next episode starts 14-42 months after the last
crest. When the forecast window opens, an episode may not have started, or may not have reached a crest whose date nothing
in the history reveals. This script measures what that costs by giving a model knowledge no deployed model can have:

  model          the benchmark's own features (reference)
  state_known    + the exact state of every episode under way at the last known month (how long it has climbed, at what
                 pace; the full remaining path of an episode already past its crest). An optimistic bound for anything
                 inferred from history: a real model has to detect a climb through month-to-month noise of similar size.
  future_shape   + the exact future path of every episode under way (including a crest still to come), the exact seasonal
                 and level path, and the size of each policy response. Not knowable at the basis month.
  true_ratio     ranking by the realised future ratios themselves (sanity check of the metric; ~1.0).

Same learner and settings in every row (XGBoost classifier, defaults), scored on the three inner folds and on validation;
the test split is not used. Writes artifacts/model-benchmark-v14/ceiling.json.
Run: .venv-bench/Scripts/python -m benchmark.ceiling   (in ml/)
"""
import json

import numpy as np

from benchmark.lab import load_dataset, write, now, CACHE
from benchmark.experiments import Model, folds, score_split, PRIMARY
from benchmark.features_v2 import THRESHOLD
from benchmark.replay import replay, verify

STATE = ['fall_in', 'climb_pace_in', 'climb_months_in', 'climb_height_in', 'fall_out', 'climb_pace_out', 'climb_months_out', 'climb_height_out', 'months_since_crest']
SHAPE = ['level_season_in', 'episodes_in', 'policy_in', 'level_season_out', 'episodes_out', 'policy_out']


def hidden_state(d, latent, rows):
    """Privileged features for the given row indexes: (state matrix, shape matrix, true-ratio score)."""
    S = np.zeros((len(d['Y']), len(STATE)), np.float32); F = np.zeros((len(d['Y']), len(SHAPE)), np.float32); T = np.zeros(len(d['Y']), np.float32)
    for r in rows:
        z = latent[(str(d['T'][r]), str(d['HS'][r]), str(d['C'][r]))]; b = int(d['B'][r]); h = int(d['H'][r])
        t0 = b - z['report_lag']  # last month published by the basis
        future = np.arange(b + h, b + h + 3); current = np.arange(t0 - 2, t0 + 1)
        state = []; shape = []; ratio = []
        for j in range(2):
            fall = pace = months = height = 0.0
            started = np.zeros(len(z['final'][j]))
            for p in z['pulses']:
                onset = p['start'] - p['duration'] + (p['lag'] if j else 0); crest = p['start'] + (p['lag'] if j else 0)
                if onset > t0:
                    continue
                started = started + p['legs'][j]
                if crest <= t0 or (j == 1 and p['start'] <= t0):  # past the crest, or the import crest is seen so the export crest date follows
                    fall += p['legs'][j][future].mean() - p['legs'][j][current].mean()
                else:
                    pace = p['legs'][j][t0] - p['legs'][j][t0 - 1]; months = t0 - onset; height = p['legs'][j][t0]
            state += [fall, pace, months, height]
            deterministic = z['path'][j] + z['seasonal'][j]
            policy = sum((e['legs'][j] for e in z['events'] if e['published'] <= b), np.zeros(len(deterministic)))
            shape += [deterministic[future].mean() - deterministic[current].mean(), started[future].mean() - started[current].mean(), policy[future].mean() - policy[current].mean()]
            ratio.append(np.log(z['final'][j][future].mean() / z['final'][j][current].mean()) - THRESHOLD[j])
        last_crest = max([p['start'] for p in z['pulses'] if p['start'] <= t0], default=t0 - 60)
        S[r] = state + [min(t0 - last_crest, 60)]; F[r] = shape; T[r] = min(ratio)
    return S, F, T


def main():
    d = load_dataset()
    rows_latent, latent = replay('', latents=True)
    cells = verify(rows_latent)
    grid = d['grid']
    names = d['features']
    chosen = json.loads((CACHE.parent / 'feature-set.json').read_text(encoding='utf-8'))['features'] if (CACHE.parent / 'feature-set.json').exists() else names
    X = d['X'][:, [names.index(n) for n in chosen]]
    splits = folds(d, grid) + [('validation', d['masks']['train'] & grid, d['masks']['validation'])]
    needed = np.zeros(len(grid), bool)
    for _, fit, score in splits:
        needed |= fit | score
    S, F, T = hidden_state(d, latent, np.flatnonzero(needed))
    variants = {'model': X, 'state_known': np.column_stack([X, S]), 'future_shape': np.column_stack([X, S, F])}
    out = {'generated_at_utc': now(), 'replay_cells_equal_to_database': cells, 'metric': PRIMARY, 'learner': 'XGBoost binary classifier, default settings, seed 42, grid training rows',
           'features': len(chosen), 'levels': {}, 'note': 'state_known and future_shape use generator internals; they are bounds, not models that can be deployed'}
    for level, matrix in variants.items():
        scores = {}
        for name, fit, score in splits:
            model = Model('xgb_clf').fit(matrix, d, fit)
            scores[name] = score_split(d, score, model.score(matrix[score]))[PRIMARY]
        out['levels'][level] = {**scores, 'inner_folds_mean': float(np.mean([v for k, v in scores.items() if k.startswith('fold')]))}
        print(level, {k: round(v, 4) for k, v in out['levels'][level].items()}, flush=True)
    scores = {name: score_split(d, score, T[score])[PRIMARY] for name, fit, score in splits}
    out['levels']['true_ratio'] = {**scores, 'inner_folds_mean': float(np.mean([v for k, v in scores.items() if k.startswith('fold')]))}
    print('true_ratio', {k: round(v, 4) for k, v in out['levels']['true_ratio'].items()})
    write('ceiling.json', out)


if __name__ == '__main__':
    main()
