"""Probe on the real monthly rows of data/integration.sqlite: does a supplier-side declaration help rank future coupled growth?

The benchmark panel is generated, so no outside series can tell when one of its growth episodes will start: a measured leading
indicator can only be tested where the labels are measured too. The repository holds such rows for one slice — Russia as
target, six candidates with observed China-sourced imports, six HS6 items, 2019-04 to 2026-06:

  inbound    candidate-reported imports from China
  outbound   candidate-reported exports to Russia
  supplier   China-reported exports to the candidate (the same goods, declared by the supplier)

Label: the benchmark's rule on the real series (future three-month mean above the current one by 15 % on imports and 20 % on
exports). Features: the v2 leg features of the two candidate-reported series ("own"), plus either two nowcast features from
the supplier series (its latest three months against its own level in the candidate's current window) or the full v2 leg
feature set of the supplier series.

Timing is an ASSUMPTION, not a recorded fact: candidate statistics are taken as known through basis-3 and China's through
basis-1 ("early") or basis-3 ("same"). The available_month column of these rows records Comtrade first-release dates of a
bulk retrieval and cannot serve as a publication calendar.

Scoring: walk-forward by year (2023, 2024, 2025: train on labels matured before the year). With six candidates a Top-5 is
nearly the whole list, so the scores are NDCG@3 and the share of correctly ordered positive/negative pairs within a query,
over queries that hold both. The sample is small (about 170 such queries); the standard error of a difference is about 0.01.

Writes artifacts/model-benchmark-v14/real-data-probe.json. Run: .venv-bench/Scripts/python -m benchmark.real_probe   (in ml/)
"""
import sqlite3

import numpy as np

from benchmark.lab import write, now  # sets the import path for `common`
from common import ROOT, mi, mt
from benchmark.features_v2 import leg_features, STATIC_LEG, DYNAMIC_LEG, THRESHOLD
from benchmark.metrics import per_query_indices, _ndcg

CANDIDATES = ['ARM', 'IND', 'KAZ', 'KGZ', 'TUR', 'UZB']
HORIZONS = [3, 6, 12]
OWN_LAG = 3
YEARS = (2023, 2024, 2025)


def load():
    series = {}
    with sqlite3.connect((ROOT / 'data/integration.sqlite').as_uri() + '?mode=ro', uri=True) as c:
        for reporter, partner, flow, hs, month, value in c.execute(
                'select reporter,partner,flow,canonical_hs,month,value_usd from observations where canonical_hs is not null order by id'):
            series.setdefault((reporter, partner, flow, hs), {})[month] = value  # a later row of the same month replaces the earlier one
    return series


def build(series, supplier_lag):
    """Rows (item, basis, horizon, candidate, label, own features, nowcast features, supplier features)."""
    first = min(min(v) for v in series.values()); last = max(max(v) for v in series.values())
    items = sorted({k[3] for k in series if k[0] in CANDIDATES and k[1] == 'RUS' and k[2] == 'X'})

    def window(key, upto):
        s = series.get(key, {})
        months = np.arange(max(first, upto - 59), upto + 1)
        raw = np.array([s.get(int(m), np.nan) for m in months], float); raw[raw <= 0] = np.nan
        known = np.flatnonzero(np.isfinite(raw))
        return (raw[:known[-1] + 1], months[:known[-1] + 1]) if len(known) else (None, None)

    def mean(key, months):
        s = series.get(key, {}); v = [s[m] for m in months if m in s and s[m] > 0]
        return float(np.mean(v)) if v else np.nan

    rows = []
    for hs in items:
        for b in range(first + 18, last - 4):
            for cand in CANDIDATES:
                kin, kout, ksup = (cand, 'CHN', 'M', hs), (cand, 'RUS', 'X', hs), ('CHN', cand, 'X', hs)
                rin, min_ = window(kin, b - OWN_LAG); rout, mout = window(kout, b - OWN_LAG); rsup, msup = window(ksup, b - supplier_lag)
                if rin is None or rout is None or len(rin) < 12 or len(rout) < 12:
                    continue
                cur = [float(np.nanmean(r[-3:])) if np.isfinite(r[-3:]).any() else np.nan for r in (rin, rout)]
                if not np.isfinite(cur).all():
                    continue
                si, di = leg_features(rin, min_, b, HORIZONS); so, do = leg_features(rout, mout, b, HORIZONS)
                sup_static = np.zeros(len(STATIC_LEG)); sup_dynamic = np.zeros((len(HORIZONS), len(DYNAMIC_LEG))); nowcast = np.zeros(2)
                if rsup is not None and len(rsup) >= 12 and np.isfinite(rsup[-3:]).any():
                    sup_static, sup_dynamic = leg_features(rsup, msup, b, HORIZONS)
                    for j, span in enumerate((3, 6)):  # the months the candidate has not published yet against the ones it has, on the supplier's series
                        a = np.nanmean(rsup[-span:]); ref = mean(ksup, [int(m) for m in min_[-span:]])
                        nowcast[j] = np.log(a / ref) if np.isfinite(a) and np.isfinite(ref) and ref > 0 else 0.0
                for k, h in enumerate(HORIZONS):
                    future = [mean(kin, range(b + h, b + h + 3)), mean(kout, range(b + h, b + h + 3))]
                    if b + h + 2 > last or not np.isfinite(future).all():
                        continue
                    ratio = np.log(np.array(future) / np.array(cur))
                    label = float(ratio[0] > THRESHOLD[0] and ratio[1] > THRESHOLD[1])
                    own = np.r_[si, so, di[k], do[k], h, min(di[k][5] - THRESHOLD[0], do[k][5] - THRESHOLD[1])]
                    rows.append((hs, b, h, cand, label, own, nowcast, np.r_[sup_static, sup_dynamic[k]]))
    return rows, (mt(first), mt(last)), items


def score(rows, parts):
    """Walk-forward scores of an XGBoost classifier (mean margin of five seeds) on the chosen feature parts (own=5, nowcast=6, supplier=7)."""
    import xgboost as xgb
    G = np.array([f'{r[0]}:{r[1]}:{r[2]}' for r in rows]); B = np.array([r[1] for r in rows]); H = np.array([r[2] for r in rows]); y = np.array([r[4] for r in rows])
    X = np.array([np.concatenate([r[p] for p in parts]) for r in rows])
    scores = np.full(len(y), np.nan)
    for year in YEARS:
        cut = mi(f'{year}-01'); train = (B + H + 2) < cut; test = (B >= cut) & (B < cut + 12)
        if train.sum() < 200 or not test.any():
            continue
        margins = [xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=5, reg_lambda=5,
                                     n_jobs=8, random_state=seed).fit(X[train], y[train]).predict(X[test], output_margin=True) for seed in range(5)]
        scores[test] = np.mean(margins, axis=0)
    ok = np.isfinite(scores); ndcg = []; pairs = []
    for idx in per_query_indices(G[ok]):
        yy = y[ok][idx]; ss = scores[ok][idx]
        if len(idx) < 4 or yy.sum() in (0, len(yy)):
            continue
        ndcg.append(_ndcg(yy, ss, 3)); pairs.append(float(np.mean(ss[yy > 0][:, None] > ss[yy == 0][None, :])))
    return {'rows_scored': int(ok.sum()), 'queries_with_both_classes': len(ndcg), 'ndcg_at_3': float(np.mean(ndcg)), 'pairs_ordered_correctly': float(np.mean(pairs))}, np.array(ndcg)


def main():
    series = load()
    out = {'generated_at_utc': now(), 'slice': 'target RUS; candidates ' + ', '.join(CANDIDATES) + '; supplier CHN', 'label': 'benchmark rule on the observed series',
           'timing_assumption': f'candidate statistics known through basis-{OWN_LAG}; supplier statistics through basis-1 (early) or basis-{OWN_LAG} (same); assumed, not recorded',
           'scoring': f'walk-forward by year {list(YEARS)}; XGBoost classifier depth 3, 200 trees, 5 seeds; queries with at least 4 candidates and both classes', 'regimes': {}}
    for regime, lag in (('early', 1), ('same', OWN_LAG)):
        rows, months, items = build(series, lag)
        own, base = score(rows, (5,))
        block = {'rows': len(rows), 'positive_rate': float(np.mean([r[4] for r in rows])), 'own_series': own}
        for name, parts in (('own_plus_supplier_nowcast', (5, 6)), ('own_plus_supplier_full', (5, 7))):
            if regime == 'same' and name == 'own_plus_supplier_nowcast':
                continue  # the nowcast compares the same months when both series end together
            result, values = score(rows, parts)
            diff = values - base
            block[name] = {**result, 'ndcg_at_3_gain': float(diff.mean()), 'gain_standard_error': float(diff.std() / np.sqrt(len(diff)))}
        out['regimes'][regime] = block; out['months'] = months; out['items'] = items
        print(regime, {k: (round(v['ndcg_at_3'], 4), round(v['pairs_ordered_correctly'], 4), round(v.get('ndcg_at_3_gain', 0), 4), round(v.get('gain_standard_error', 0), 4)) for k, v in block.items() if isinstance(v, dict)}, flush=True)
    write('real-data-probe.json', out)


if __name__ == '__main__':
    main()
