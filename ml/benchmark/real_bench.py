"""Real-data benchmark: the benchmark's label and ranking metrics on observed monthly declarations (no generated row).

Series (UN Comtrade monthly, as retrieved; a missing month is missing, never zero):
  outbound   reporter's exports of the item to the target    data/observed-exports.json, data/integration.sqlite, data/observed-wider.json
  inbound    reporter's imports of the item from the world   data/observed-imports.json, data/observed-wider.json
A route is target x item x reporter with at least 18 known months on both legs. The label is the benchmark's rule on these
series: the mean of months b+h … b+h+2 above the mean of the latest three known months by 15 % (imports) and 20 % (exports).
A query is target x item x basis x horizon; its documents are the reporters that have a row.

Two pools of reporters:
  app     the nine candidate countries of the application
  wider   those plus eight countries collected for evaluation only (scripts/collect_observed_exports.py --wider), which
          lengthens the candidate lists; the application's candidate set is not changed by this

Timing is an ASSUMPTION: a reporter's statistics are taken as known through basis-3. The retrieval dates of these rows are
all recent, so they say nothing about when each month was first published.

What is compared, on the same scored rows (bases from 2023-01, horizons 3/6/12):
  random          a random order (the floor of each metric at these list sizes)
  persistence     rank by the weaker leg's latest 3-month growth (what the v1 `coupled_growth` idea amounts to)
  panel_model     an XGBoost classifier fitted ONLY on the generated panel (label maturity <= 2018-12, before any observed
                  anchor enters the panel) with the v2 leg features, applied to the real rows unchanged: does what the model
                  learned on the panel carry over?
  real_model      the same learner fitted on real rows, walk-forward by year (labels matured before the scored year)
  real_plus_panel real_model with the panel model's margin as one more feature

Scores, each over the queries it can be computed on:
  pairs_ordered_correctly  share of positive/negative pairs in the right order   (at least 2 reporters, both classes)
  ndcg_at_3                                                                      (at least 4 reporters, both classes)
  ndcg_at_5_pos            the project's primary metric                          (at least 5 reporters, a positive)

Writes artifacts/model-benchmark-v14/real-benchmark.json. Run: .venv-bench/Scripts/python -m benchmark.real_bench   (in ml/)
"""
import json
import sqlite3

import numpy as np

from benchmark.lab import load_dataset, write, now  # sets the import path for `common`
from common import ROOT, TARGETS, CANDIDATES, mi, mt
from benchmark.features_v2 import leg_features, STATIC_LEG, DYNAMIC_LEG, THRESHOLD
from benchmark.metrics import per_query_indices, _ndcg

HORIZONS = [3, 6, 12]
OWN_LAG = 3
MIN_MONTHS = 18
FIRST_SCORED = '2023-01'
WIDER_MIN_MONTHS = 60  # months of both legs the wider pool needs before it is scored
PANEL_COLUMNS = ([f'ts_{n}_{leg}' for leg in ('in', 'out') for n in STATIC_LEG] + [f'ts_{n}_{leg}' for leg in ('in', 'out') for n in DYNAMIC_LEG]
                 + ['horizon', 'ts_min_exp'])
LEARNER = dict(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=5, reg_lambda=5, n_jobs=8)


def load_series(wider):
    """(outbound {(target, hs, reporter): {month: usd}}, inbound {(hs, reporter): {month: usd}}, reporters in the pool)."""
    outbound, inbound = {}, {}
    reporters = list(CANDIDATES)
    with sqlite3.connect((ROOT / 'data/integration.sqlite').as_uri() + '?mode=ro', uri=True) as c:
        for reporter, partner, hs, month, value in c.execute(
                "select reporter,partner,canonical_hs,month,value_usd from observations where canonical_hs is not null and flow='X' order by id"):
            if reporter in CANDIDATES and partner in TARGETS and reporter != partner and value and value > 0:
                outbound.setdefault((partner, '880330' if hs == '880730' else hs, reporter), {})[month] = value
    for r in json.loads((ROOT / 'data/observed-exports.json').read_text(encoding='utf-8'))['rows']:
        if r['usd'] > 0:
            outbound.setdefault((r['partner'], r['hs'], r['reporter']), {})[r['month']] = r['usd']  # the keyless collection is the later retrieval
    for r in json.loads((ROOT / 'data/observed-imports.json').read_text(encoding='utf-8'))['rows']:
        if r['partner'] == 'WLD' and r['usd'] > 0:
            inbound.setdefault((r['hs'], r['reporter']), {})[r['month']] = r['usd']
    if wider:
        extra = json.loads((ROOT / 'data/observed-wider.json').read_text(encoding='utf-8'))
        reporters += extra['reporters']
        for r in extra['exports']['rows']:
            if r['usd'] > 0:
                outbound.setdefault((r['partner'], r['hs'], r['reporter']), {})[r['month']] = r['usd']
        for r in extra['imports']['rows']:
            if r['partner'] == 'WLD' and r['usd'] > 0:
                inbound.setdefault((r['hs'], r['reporter']), {})[r['month']] = r['usd']
    return outbound, inbound, reporters


def build_rows(outbound, inbound):
    """Rows of every route, basis and horizon: dict(G, T, HS, C, B, H, y, X) with X over PANEL_COLUMNS."""
    events = json.loads((ROOT / 'data/events.json').read_text(encoding='utf-8'))
    last = max(max(v) for v in outbound.values())

    def window(s, upto):
        months = np.arange(max(min(s), upto - 59), upto + 1)
        raw = np.array([s.get(int(m), np.nan) for m in months], float)
        known = np.flatnonzero(np.isfinite(raw))
        return (raw[:known[-1] + 1], months[:known[-1] + 1]) if len(known) else (None, None)

    def mean(s, months):
        v = [s[m] for m in months if m in s]
        return float(np.mean(v)) if v else np.nan

    out = dict(G=[], T=[], HS=[], C=[], B=[], H=[], y=[], X=[])
    routes = 0
    exp = DYNAMIC_LEG.index('exp_base')
    for (target, hs, reporter), so_series in sorted(outbound.items()):
        si_series = inbound.get((hs, reporter))
        if si_series is None or len(so_series) < MIN_MONTHS or len(si_series) < MIN_MONTHS:
            continue
        routes += 1
        start = max(min(so_series), min(si_series)) + MIN_MONTHS
        for b in range(start, last - 4):
            rin, min_ = window(si_series, b - OWN_LAG); rout, mout = window(so_series, b - OWN_LAG)
            if rin is None or rout is None or len(rin) < 12 or len(rout) < 12:
                continue
            if not (np.isfinite(rin[-3:]).any() and np.isfinite(rout[-3:]).any()):
                continue
            cur = [float(np.nanmean(rin[-3:])), float(np.nanmean(rout[-3:]))]
            effective = [mi(e['effective']) for e in events if e['target'] == target and mi(e['published']) <= b and mi(e['effective']) <= b]
            event_month = max(effective) if effective else None
            s_in, d_in = leg_features(rin, min_, b, HORIZONS, event_month); s_out, d_out = leg_features(rout, mout, b, HORIZONS, event_month)
            for k, h in enumerate(HORIZONS):
                if b + h + 2 > last:
                    continue
                future = [mean(si_series, range(b + h, b + h + 3)), mean(so_series, range(b + h, b + h + 3))]
                if not np.isfinite(future).all():
                    continue
                ratio = np.log(np.array(future) / np.array(cur))
                x = np.r_[s_in, s_out, d_in[k], d_out[k], h, min(d_in[k][exp] - THRESHOLD[0], d_out[k][exp] - THRESHOLD[1])]
                out['G'].append(f'{target}:{hs}:{b}:{h}'); out['T'].append(target); out['HS'].append(hs); out['C'].append(reporter); out['B'].append(b); out['H'].append(h)
                out['y'].append(float(ratio[0] > THRESHOLD[0] and ratio[1] > THRESHOLD[1])); out['X'].append(x)
    d = {k: np.array(v) for k, v in out.items()}
    d['routes'] = routes
    return d


def query_scores(y, score, groups):
    """The three scores and, for standard errors, the per-query values of the pair score."""
    pairs, n3, n5 = [], [], []
    for idx in per_query_indices(groups):
        yy, ss = y[idx], score[idx]
        positives = yy.sum()
        if positives == 0:
            continue
        if len(idx) >= 5:
            n5.append(_ndcg(yy, ss, 5))
        if positives < len(yy):
            order = ss[yy > 0][:, None] - ss[yy == 0][None, :]
            pairs.append(float(np.mean((order > 0) + 0.5 * (order == 0))))
            if len(idx) >= 4:
                n3.append(_ndcg(yy, ss, 3))
    mean = lambda v: float(np.mean(v)) if v else None
    return {'pairs_ordered_correctly': mean(pairs), 'pair_queries': len(pairs), 'ndcg_at_3': mean(n3), 'ndcg_at_3_queries': len(n3),
            'ndcg_at_5_pos': mean(n5), 'ndcg_at_5_pos_queries': len(n5)}, np.array(pairs)


def run(pool, panel_model):
    import xgboost as xgb
    outbound, inbound, reporters = load_series(pool == 'wider')
    d = build_rows(outbound, inbound)
    y, G, B, H, X, T = d['y'], d['G'], d['B'], d['H'], d['X'], d['T']
    col = {n: i for i, n in enumerate(PANEL_COLUMNS)}
    panel_margin = panel_model.predict(X, output_margin=True)
    real = np.full(len(y), np.nan); stacked = np.full(len(y), np.nan); Xs = np.column_stack([X, panel_margin]); refits = []
    for year in range(int(FIRST_SCORED[:4]), 1990 + int(B.max()) // 12 + 1):
        cut = mi(f'{year}-01'); train = (B + H + 2) < cut; test = (B >= cut) & (B < cut + 12)
        if train.sum() < 300 or not test.any():
            continue
        for target, matrix in ((real, X), (stacked, Xs)):
            target[test] = np.mean([xgb.XGBClassifier(**LEARNER, random_state=s).fit(matrix[train], y[train]).predict(matrix[test], output_margin=True) for s in range(5)], axis=0)
        refits.append({'year': year, 'train_rows': int(train.sum()), 'scored_rows': int(test.sum())})
    scored = (B >= mi(FIRST_SCORED)) & np.isfinite(real)
    rng = np.random.default_rng(0)
    systems = {'random': rng.random(len(y)), 'persistence': np.minimum(X[:, col['ts_mom3_in']], X[:, col['ts_mom3_out']]),
               'panel_model': panel_margin, 'real_model': real, 'real_plus_panel': stacked}
    sizes = np.array([len(i) for i in per_query_indices(G[scored])])
    out = {'reporters': reporters, 'routes': int(d['routes']), 'routes_by_target': {t: len({(h, c) for tt, h, c in zip(T, d['HS'], d['C']) if tt == t}) for t in TARGETS},
           'reporters_with_routes': sorted(set(d['C'].tolist())), 'rows_all': int(len(y)), 'rows_scored': int(scored.sum()), 'positive_rate_scored': float(y[scored].mean()),
           'scored': f'bases from {FIRST_SCORED} to {mt(int(B[scored].max()))}, horizons {HORIZONS}', 'queries_scored': int(len(sizes)),
           'reporters_per_query': {str(int(k)): int((sizes == k).sum()) for k in np.unique(sizes)}, 'walk_forward': refits, 'systems': {}, 'by_target': {}}
    per_query = {}
    for name, score in systems.items():
        out['systems'][name], per_query[name] = query_scores(y[scored], score[scored], G[scored])
        print(f"{pool:6s} {name:16s}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['systems'][name].items()}, flush=True)
    for name, against in (('panel_model', 'random'), ('real_model', 'panel_model'), ('real_plus_panel', 'real_model')):
        diff = per_query[name] - per_query[against]
        out['systems'][name]['pairs_gain'] = {'against': against, 'mean': float(diff.mean()), 'standard_error': float(diff.std() / np.sqrt(len(diff)))}
    for t in TARGETS:
        m = scored & (T == t)
        if m.sum():
            out['by_target'][t] = {'rows': int(m.sum()), 'positive_rate': float(y[m].mean()), **{name: query_scores(y[m], score[m], G[m])[0] for name, score in systems.items()}}
            print(f'{pool:6s}', t, int(m.sum()), {n: (None if v['pairs_ordered_correctly'] is None else round(v['pairs_ordered_correctly'], 3), v['pair_queries'])
                                                 for n, v in out['by_target'][t].items() if isinstance(v, dict)}, flush=True)
    return out


def main():
    import xgboost as xgb
    panel = load_dataset(); rows = panel['masks']['train'] & panel['grid'] & (panel['H'] <= 12)
    Xp = panel['X'][:, [panel['features'].index(n) for n in PANEL_COLUMNS]]
    panel_model = xgb.XGBClassifier(**{**LEARNER, 'max_depth': 5, 'n_estimators': 400}, random_state=42).fit(Xp[rows], panel['Y'][rows, 0].astype(int))
    out = {'generated_at_utc': now(), 'label': 'benchmark rule on observed series', 'timing_assumption': f'reporter statistics known through basis-{OWN_LAG}; assumed, not recorded',
           'features': 'v2 leg features of the two legs, horizon and the joint margin (66 columns); no peer, v1 or country feature',
           'panel_model_training': {'rows': int(rows.sum()), 'rule': 'generated panel, train split (label maturity <= 2018-12), grid bases, horizons <= 12'}, 'pools': {}}
    out['pools']['app'] = run('app', panel_model)
    # The wider pool is scored only once both of its legs are collected: without the imports its reporters have no route.
    path = ROOT / 'data/observed-wider.json'
    months = {k: json.loads(path.read_text(encoding='utf-8'))[k]['months_received'] for k in ('exports', 'imports')} if path.exists() else {'exports': 0, 'imports': 0}
    if min(months.values()) >= WIDER_MIN_MONTHS:
        out['pools']['wider'] = run('wider', panel_model)
    else:
        out['wider_pool'] = {'status': 'collection incomplete', 'months_received': months, 'needed': WIDER_MIN_MONTHS,
                             'resume': 'python scripts/collect_observed_exports.py --wider --limit 120, then the same with --imports (the public endpoint has a daily request quota)'}
        print('wider pool not scored:', out['wider_pool'], flush=True)
    write('real-benchmark.json', out)


if __name__ == '__main__':
    main()
