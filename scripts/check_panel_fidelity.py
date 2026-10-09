"""Compare the demo panel with the observed rows: distribution statistics and a discriminator (track B).

Usage: python scripts/check_panel_fidelity.py [--label before|after] [--output PATH]
Writes artifacts/integration/panel-fidelity[-label].json. This measures whether the panel
looks like the observations; it says nothing about prediction quality.

Statistics are computed on 24-month windows of log USD for routes that have an observed
counterpart (candidate imports from China = inbound leg, candidate exports to the target =
outbound leg). The discriminator is a gradient-boosting classifier on window features
(level, growth sd, lag-1 autocorrelation, seasonal amplitude, coefficient of variation,
month-to-month absolute change) with grouped 5-fold cross-validation; AUC near 0.5 means
windows are indistinguishable, the pass line is 0.6.
"""
import argparse
import datetime as dt
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));import btl_paths
from common import DB, TARGETS
from scripts.estimate_panel_parameters import load_series, interpolate, rolling_median

WINDOW = 24


def panel_series():
    """{(leg, target, hs, candidate): {month: log value}} from the final revision of the panel."""
    out = defaultdict(dict)
    with closing(sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)) as c:
        for target, hs, cand, month, inbound, outbound in c.execute('select target,hs,candidate,month,inbound,outbound from trade where revision=1'):
            if inbound and inbound > 0:
                out[('in', target, hs, cand)][month] = math.log(inbound)
            if outbound and outbound > 0:
                out[('out', target, hs, cand)][month] = math.log(outbound)
    return out


def window_features(values):
    x = np.asarray(values, dtype=float)
    trend = rolling_median(x, 12)
    dev = x - trend
    calendar = np.arange(len(x)) % 12
    profile = np.array([dev[calendar == k].mean() for k in range(12)])
    growth = np.diff(x)
    ar1 = float(np.corrcoef(dev[1:], dev[:-1])[0, 1]) if np.std(dev) > 1e-9 else 0.0
    return [float(np.mean(x)), float(np.std(growth)), ar1, float(np.std(profile)), float(np.std(x) / max(abs(np.mean(x)), 1e-9)),
            float(np.mean(np.abs(growth))), float(np.max(x) - np.min(x))]


def windows(series, step=6):
    for key, s in series.items():
        if len(s) < WINDOW:
            continue
        first, values = interpolate(s)
        for start in range(0, len(values) - WINDOW + 1, step):
            yield key, window_features(values[start:start + WINDOW])


def summarize(feature_rows):
    arr = np.array(feature_rows)
    names = ['level', 'growth_sd', 'ar1', 'seasonal_sd', 'cv', 'abs_change', 'range']
    return {n: {'median': round(float(np.median(arr[:, i])), 4), 'p25': round(float(np.percentile(arr[:, i], 25)), 4), 'p75': round(float(np.percentile(arr[:, i], 75)), 4)}
            for i, n in enumerate(names)}


def ks(a, b):
    a = np.sort(a); b = np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side='right') / len(a); fb = np.searchsorted(b, grid, side='right') / len(b)
    return float(np.max(np.abs(fa - fb)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / 'artifacts/integration' / ('panel-fidelity' + (f'-{args.label}' if args.label else '') + '.json')
    observed, zeros, _, _ = load_series()
    panel = panel_series()
    # Observed inbound series are target-independent; compare them with the panel's inbound leg of every target.
    obs_rows = []; obs_groups = []
    for key, feats in windows(observed):
        obs_rows.append(feats); obs_groups.append('obs:' + '|'.join(str(k) for k in key))
    matched = {}
    for (leg, target, hs, cand) in observed:
        if leg == 'in':
            for t in TARGETS:
                if cand != t and (leg, t, hs, cand) in panel:
                    matched[(leg, t, hs, cand)] = panel[(leg, t, hs, cand)]
        elif (leg, target, hs, cand) in panel:
            matched[(leg, target, hs, cand)] = panel[(leg, target, hs, cand)]
    pan_rows = []; pan_groups = []
    for key, feats in windows(matched):
        pan_rows.append(feats); pan_groups.append('pan:' + '|'.join(str(k) for k in key))
    X = np.array(obs_rows + pan_rows); y = np.array([1] * len(obs_rows) + [0] * len(pan_rows)); groups = np.array(obs_groups + pan_groups)
    aucs = []
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0).fit(X[train], y[train])
        if len(np.unique(y[test])) == 2:
            aucs.append(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1]))
    auc = float(np.mean(aucs)) if aucs else None
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0).fit(X, y)
    importance = None
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(clf, X, y, n_repeats=5, random_state=0, scoring='roc_auc')
        names = ['level', 'growth_sd', 'ar1', 'seasonal_sd', 'cv', 'abs_change', 'range']
        importance = {n: round(float(v), 4) for n, v in zip(names, imp.importances_mean)}
    except Exception:
        pass
    obs = np.array(obs_rows); pan = np.array(pan_rows)
    names = ['level', 'growth_sd', 'ar1', 'seasonal_sd', 'cv', 'abs_change', 'range']
    result = dict(
        schema_version='btl-v3-panel-fidelity-v1', label=args.label or None, computed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        panel=str(DB.relative_to(ROOT)), window_months=WINDOW,
        observed_windows=len(obs_rows), panel_windows=len(pan_rows), matched_routes=len(matched),
        observed=summarize(obs_rows), panel_matched=summarize(pan_rows),
        ks_by_feature={n: round(ks(obs[:, i], pan[:, i]), 4) for i, n in enumerate(names)},
        discriminator=dict(model='HistGradientBoosting, grouped 5-fold', auc=round(auc, 4) if auc is not None else None, pass_line=0.6,
                           passed=(auc is not None and auc <= 0.6), permutation_importance_auc=importance),
        note='Fidelity of the panel to observed rows; not a prediction-quality measure. Windows overlap (step 6 months) and are grouped by series in cross-validation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(label=args.label, matched_routes=len(matched), observed_windows=len(obs_rows), panel_windows=len(pan_rows),
                          discriminator_auc=result['discriminator']['auc'], passed=result['discriminator']['passed'],
                          ks=result['ks_by_feature'], output=str(output.relative_to(ROOT))), ensure_ascii=False))


if __name__ == '__main__':
    main()
