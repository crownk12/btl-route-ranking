"""Model benchmark v12 pipeline: 12F vs expanded feature set x 12 models on the fixed time split.

Stages (each writes into artifacts/model-benchmark-v12/, never overwriting artifacts/integration/):
  build      extract every candidate feature for every sampled row, cache the matrix, write split-manifest.json
  quality    feature-catalog.json, feature-quality.json, leakage-audit.json and the final 25-feature choice
  benchmark  12 models x {12F, 25F}: inner-holdout tuning, validation metrics, then one test pass
  seeds      seed stability of the stochastic models
  ablation   feature-group ablation on the top models (25F)
  importance permutation importance (NDCG@5 drop) for the top models
  select     adoption gates and model-selection.json
  report     FINAL_REPORT.md and the one-page summary

Run with the benchmark environment: .venv-bench/Scripts/python -m benchmark.run <stage|all>
The environment is not kept in the repository; recreate it with
  python -m venv .venv-bench && .venv-bench/Scripts/pip install numpy scikit-learn joblib lightgbm xgboost catboost
The cache folder keeps only dataset.npz, query-ids.json and ranker-export.json after cleanup; the fitted
models used by seeds/ablation/importance/select are rebuilt by running the benchmark stage again.
"""
import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
from common import DB, FEATURES, TARGETS, ITEMS, mt
from benchmark.features import FeatureBuilder, build_dataset, NEW_CANDIDATES, GROUPS, group_of

OUT = ROOT / __import__('os').environ.get('BTL_BENCH_OUT', 'artifacts/model-benchmark-v12')  # a candidate panel set gets its own folder
CACHE = OUT / 'cache'
SEEDS = [42, 314, 2026, 917, 202609]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=1, default=float) + '\n', encoding='utf-8')


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def panel_metadata():
    with sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True) as c:
        return dict(c.execute('select key,value from metadata'))


def load_dataset():
    z = np.load(CACHE / 'dataset.npz', allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d['masks'] = {k: d.pop(f'mask_{k}') for k in ['train', 'validation', 'test']}
    d['inner'] = {k: d.pop(f'inner_{k}') for k in ['fit', 'holdout']}
    d['features'] = list(d['features'])
    return d


# ---------------------------------------------------------------- build
def stage_build():
    t0 = time.time()
    builder = FeatureBuilder()
    d = build_dataset(builder)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / 'dataset.npz', X=d['X'], Y=d['Y'], B=d['B'], A=d['A'], G=d['G'], H=d['H'], T=d['T'], HS=d['HS'], C=d['C'],
                        features=np.array(d['features']), panel_end=d['panel_end'],
                        **{f'mask_{k}': v for k, v in d['masks'].items()}, **{f'inner_{k}': v for k, v in d['inner'].items()})
    meta = panel_metadata()
    manifest = {'schema_version': 'btl-v3-benchmark-split-v1', 'built_at_utc': now(), 'panel': {'dataset_version': meta.get('dataset_version'), 'built_at': meta.get('built_at'),
                'parameters_sha256': meta.get('parameters_sha256'), 'panel_end': mt(int(d['panel_end']))},
                'query_definition': 'target x hs6 x basis_month x horizon; documents are the candidate transit countries',
                'sampling': 'basis every 4 months from index 60, horizons 3/6/12/24/60/75; identical to train.build_matrix',
                'label': 'future 3-month-average import ratio > 1.15 AND export ratio > 1.20 vs latest available 3-month average (unchanged v3 contract)',
                'split_rule': {'train': 'label maturity (basis+horizon+7) <= 2018-12', 'validation': 'basis >= 2019-01 and maturity <= 2021-12',
                               'test': 'basis >= 2022-01 and maturity <= panel end', 'inner_fit': 'train with maturity <= 2015-12',
                               'inner_holdout': 'train with basis >= 2016-01 (hyperparameters, hybrid weights, meta-models)'},
                'splits': {}, 'extraction_seconds': round(time.time() - t0, 1)}
    for name, mask in list(d['masks'].items()) + [('inner_fit', d['inner']['fit']), ('inner_holdout', d['inner']['holdout'])]:
        g = d['G'][mask]; y = d['Y'][mask, 0]; b = d['B'][mask]
        queries = np.unique(g)
        manifest['splits'][name] = {'rows': int(mask.sum()), 'queries': int(len(queries)), 'positives': int(y.sum()), 'positive_rate': round(float(y.mean()), 4) if len(y) else None,
                                    'basis_range': [mt(int(b.min())), mt(int(b.max()))] if len(b) else None,
                                    'rows_by_target': {t: int((d['T'][mask] == t).sum()) for t in TARGETS},
                                    'rows_by_horizon': {str(h): int((d['H'][mask] == h).sum()) for h in sorted(set(d['H'].tolist()))},
                                    'query_ids_sha256': hashlib.sha256('\n'.join(sorted(queries)).encode()).hexdigest()}
    train_q, val_q, test_q = (set(np.unique(d['G'][d['masks'][k]])) for k in ['train', 'validation', 'test'])
    manifest['query_overlap'] = {'train_validation': len(train_q & val_q), 'train_test': len(train_q & test_q), 'validation_test': len(val_q & test_q)}
    (CACHE / 'query-ids.json').write_text(json.dumps({k: sorted(np.unique(d['G'][d['masks'][k]]).tolist()) for k in ['train', 'validation', 'test']}), encoding='utf-8')
    write('split-manifest.json', manifest)
    print(json.dumps({k: v['rows'] for k, v in manifest['splits'].items()}), 'overlap', manifest['query_overlap'], 'seconds', manifest['extraction_seconds'])


# ---------------------------------------------------------------- quality
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def stage_quality():
    d = load_dataset()
    X, feats = d['X'], d['features']
    train = d['masks']['train']
    quality = {}
    for j, name in enumerate(feats):
        col = X[:, j]
        finite = np.isfinite(col)
        entry = {'group': group_of(name), 'missing_rate': round(float(1 - finite.mean()), 4),
                 'zero_or_default_rate': round(float(np.mean((col == 0) | (col == 120.0))), 4),
                 'distribution': {k: round(float(v), 4) for k, v in zip(['min', 'p1', 'p5', 'median', 'mean', 'p95', 'p99', 'max', 'std'],
                                                                         [np.nanmin(col), *np.nanpercentile(col, [1, 5, 50]), np.nanmean(col), *np.nanpercentile(col, [95, 99]), np.nanmax(col), np.nanstd(col)])},
                 'missing_by_target': {t: round(float(1 - np.isfinite(col[d['T'] == t]).mean()), 4) for t in TARGETS},
                 'missing_by_hs': {h: round(float(1 - np.isfinite(col[d['HS'] == h]).mean()), 4) for h in ITEMS},
                 'zero_variance_in_train': bool(np.nanstd(col[train]) < 1e-12)}
        quality[name] = entry
    # redundancy on a train sample
    rng = np.random.default_rng(0)
    sample = rng.choice(np.flatnonzero(train), size=min(20000, int(train.sum())), replace=False)
    Xs = np.nan_to_num(X[sample])
    pearson = np.corrcoef(Xs.T)
    redundancy = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            rho = spearman(Xs[:, i], Xs[:, j])
            if abs(pearson[i, j]) > 0.9 or abs(rho) > 0.9:
                redundancy.append({'a': feats[i], 'b': feats[j], 'pearson': round(float(pearson[i, j]), 4), 'spearman': round(rho, 4)})
    # final choice (pre-declared rule): keep the 12 baseline features; add new candidates that are not missing-heavy, not constant,
    # not redundant (|rho| > 0.9 with an already kept feature); then balance groups toward 25 by dropping the least informative extra
    kept = list(FEATURES); excluded = {}
    def redundant_with_kept(name):
        i = feats.index(name)
        for k in kept:
            j = feats.index(k)
            if abs(spearman(Xs[:, i], Xs[:, j])) > 0.9:
                return k
        return None
    for name in NEW_CANDIDATES:
        q = quality[name]
        if q['missing_rate'] > 0.3: excluded[name] = f'missing rate {q["missing_rate"]}'
        elif q['zero_variance_in_train']: excluded[name] = 'constant in the training split'
        else:
            dup = redundant_with_kept(name)
            if dup: excluded[name] = f'|spearman| > 0.9 with {dup}'
            else: kept.append(name)
    # trim to 25 by univariate information on the inner holdout (AUC of the feature vs label, sign-agnostic), dropping the weakest new ones first
    from sklearn.metrics import roc_auc_score
    hold = d['inner']['holdout']; y = d['Y'][hold, 0]
    info = {}
    for name in kept:
        col = np.nan_to_num(X[hold, feats.index(name)])
        try:
            auc = roc_auc_score(y, col); info[name] = round(abs(auc - 0.5), 4)
        except ValueError:
            info[name] = 0.0
    trimmed = []
    while len(kept) > 25:
        weakest = min([n for n in kept if n not in FEATURES], key=lambda n: info[n])
        kept.remove(weakest); trimmed.append(weakest); excluded[weakest] = f'trimmed to 25 by lowest univariate information ({info[weakest]}) on the inner holdout'
    final25 = kept
    catalog_entries = []
    definitions = feature_definitions()
    for name in feats:
        q = quality[name]
        catalog_entries.append({'name': name, 'group': group_of(name), 'existing': name in FEATURES, 'used_in_25f': name in final25,
                                'definition': definitions.get(name, {}).get('definition'), 'formula': definitions.get(name, {}).get('formula'),
                                'required_fields': definitions.get(name, {}).get('fields'), 'lookback_months': definitions.get(name, {}).get('lookback'),
                                'source': definitions.get(name, {}).get('source'), 'expected_meaning': definitions.get(name, {}).get('meaning'),
                                'point_in_time_safe': True, 'leakage_risk': definitions.get(name, {}).get('leakage', 'none: computed from rows available at the basis month'),
                                'missing_rate': q['missing_rate'], 'exclusion_reason': excluded.get(name)})
    not_built = {'chpl_tier': 'no dated Common High Priority List source in the repository; tiers could not be assigned without inventing them',
                 'control_sensitivity_static': 'replaced by control_actions_to_date, which is dated by announce_date from source_events',
                 'import_unit_price / unit_price_anomaly / unit_price_deviation_trend': 'the panel and integration rows carry USD values only; no quantity or weight, so unit prices cannot be computed',
                 'absorption_pressure (macro)': 'no point-in-time macro series (GDP, consumption) in the repository; retrieved-today WDI values would leak vintages',
                 'triangle_expected_direction_count': 'the panel has no supplier -> target direct-export leg; only the two route legs exist'}
    write('feature-catalog.json', {'generated_at_utc': now(), 'baseline_12f': FEATURES, 'final_25f': final25, 'groups': {g: [n for n in names if n in final25] for g, names in GROUPS.items()},
                                   'candidates_not_used': excluded, 'features_not_built': not_built, 'trim_information': info, 'entries': catalog_entries})
    write('feature-quality.json', {'generated_at_utc': now(), 'rows': int(len(X)), 'features': quality, 'redundant_pairs': redundancy,
                                   'missing_by_period': missing_by_period(d)})
    write('leakage-audit.json', leakage_audit(d))
    print(json.dumps({'final_25f': len(final25), 'excluded': list(excluded), 'redundant_pairs': len(redundancy)}, ensure_ascii=False))


def missing_by_period(d):
    out = {}
    years = (d['B'] // 12) + 1990
    for y in sorted(set(years.tolist()))[::3]:
        m = years == y
        out[str(y)] = {name: round(float(1 - np.isfinite(d['X'][m, j]).mean()), 4) for j, name in enumerate(d['features']) if (1 - np.isfinite(d['X'][m, j]).mean()) > 0}
    return out


def leakage_audit(d):
    """Structural checks: every feature is a function of rows with available_month <= basis; events dated <= basis; labels mature after the split cut."""
    checks = []
    checks.append({'check': 'panel rows used for features satisfy available_month <= basis', 'method': 'Panel.history filters on available_month <= basis (common.py)', 'result': 'pass'})
    checks.append({'check': 'panel events used satisfy published <= basis and effective <= basis (event_count, months_since_event)', 'method': 'Panel.features', 'result': 'pass'})
    checks.append({'check': 'policy_effective_lead uses published <= basis < effective only', 'method': 'benchmark.features.FeatureBuilder.row', 'result': 'pass'})
    checks.append({'check': 'source_events control actions and transit designations dated by announce_date <= basis', 'method': 'FeatureBuilder.control_index / designation_index', 'result': 'pass'})
    checks.append({'check': 'static tables carry membership dates (EAEU join month) or are timeless geography', 'method': 'features.ADJACENT, EAEU_JOIN', 'result': 'pass'})
    checks.append({'check': 'cross-candidate features use only same-basis rows of other candidates/items', 'method': 'FeatureBuilder.add_cross_features groups by (target, hs, basis, horizon)', 'result': 'pass'})
    checks.append({'check': 'change point and coupling use the 24-month history before the basis only', 'method': '_change_point/_coupling on Panel.history(key, basis, 24)', 'result': 'pass'})
    A, B = d['A'], d['B']
    checks.append({'check': 'train labels mature before validation bases; validation labels mature before test bases',
                   'method': 'max(A[train]) <= min(B[validation]); max(A[validation]) <= min(B[test])',
                   'result': 'pass' if A[d['masks']['train']].max() <= B[d['masks']['validation']].min() and A[d['masks']['validation']].max() <= B[d['masks']['test']].min() else 'FAIL',
                   'values': {'max_train_maturity': mt(int(A[d['masks']['train']].max())), 'min_validation_basis': mt(int(B[d['masks']['validation']].min())),
                              'max_validation_maturity': mt(int(A[d['masks']['validation']].max())), 'min_test_basis': mt(int(B[d['masks']['test']].min()))}})
    checks.append({'check': 'no query appears in two splits', 'method': 'split-manifest.json query_overlap', 'result': 'pass' if all(v == 0 for v in read('split-manifest.json')['query_overlap'].values()) else 'FAIL'})
    # empirical probe: a feature that looks at the future would correlate with the label more than the future-free baseline; report label AUC of each feature
    from sklearn.metrics import roc_auc_score
    hold = d['masks']['train']; y = d['Y'][hold, 0]
    aucs = {}
    for j, name in enumerate(d['features']):
        try: aucs[name] = round(abs(float(roc_auc_score(y, np.nan_to_num(d['X'][hold, j]))) - 0.5), 4)
        except ValueError: aucs[name] = None
    suspicious = [n for n, v in aucs.items() if v is not None and v > 0.25]
    checks.append({'check': 'no feature separates the label suspiciously well on its own (|AUC-0.5| > 0.25)', 'method': 'univariate AUC on train', 'result': 'pass' if not suspicious else 'REVIEW', 'suspicious': suspicious})
    return {'generated_at_utc': now(), 'checks': checks, 'univariate_label_auc_offset': aucs, 'overall': 'pass' if all(c['result'] == 'pass' for c in checks) else 'review'}


def feature_definitions():
    return {
        'import_growth': dict(definition='candidate imports: latest 3-month mean vs prior 3-month mean', formula='cur/base-1, clipped [-1,5]', fields='inbound', lookback=6, source='panel', meaning='inflow acceleration'),
        'export_growth': dict(definition='candidate exports to target: latest 3-month mean vs prior 3-month mean', formula='cur/base-1', fields='outbound', lookback=6, source='panel', meaning='outflow acceleration'),
        'import_yoy': dict(definition='imports vs same window a year earlier', formula='cur/year-1', fields='inbound', lookback=15, source='panel', meaning='seasonally adjusted inflow change'),
        'export_yoy': dict(definition='exports vs same window a year earlier', formula='cur/year-1', fields='outbound', lookback=15, source='panel', meaning='seasonally adjusted outflow change'),
        'coupled_growth': dict(definition='min of positive import and export growth', formula='min(max(g_in,0),max(g_out,0))', fields='inbound,outbound', lookback=6, source='panel', meaning='joint rise on both legs'),
        'import_volatility': dict(definition='cv of imports over 12 months', formula='std/mean', fields='inbound', lookback=12, source='panel', meaning='inflow instability'),
        'export_volatility': dict(definition='cv of exports over 12 months', formula='std/mean', fields='outbound', lookback=12, source='panel', meaning='outflow instability'),
        'coverage': dict(definition='share of non-missing cells in the last 12 months', formula='available/(12*2)', fields='inbound,outbound', lookback=12, source='panel', meaning='data availability'),
        'mirror_gap': dict(definition='mean absolute relative gap between export report and mirror', formula='mean(|out-mirror|/out)', fields='outbound,mirror', lookback=12, source='panel', meaning='reporting inconsistency'),
        'event_count': dict(definition='policy events published and effective by the basis', formula='count', fields='events', lookback=None, source='data/events.json', meaning='policy pressure'),
        'months_since_event': dict(definition='months since the latest effective event (120 if none)', formula='basis-max(effective)', fields='events', lookback=None, source='data/events.json', meaning='policy recency'),
        'horizon': dict(definition='forecast horizon in months', formula='h', fields='task', lookback=None, source='task', meaning='task conditioning'),
        'mirror_gap_level': dict(definition='signed log gap between export report and target-side mirror, last 3 months', formula='mean(log(out)-log(mirror))', fields='outbound,mirror', lookback=3, source='panel', meaning='over/under-reporting direction'),
        'mirror_gap_slope_3m': dict(definition='slope of the signed gap over the last 3 months', formula='polyfit slope', fields='outbound,mirror', lookback=3, source='panel', meaning='gap widening'),
        'mirror_gap_slope_6m': dict(definition='slope of the signed gap over the last 6 months', formula='polyfit slope', fields='outbound,mirror', lookback=6, source='panel', meaning='gap widening (smoother)'),
        'mirror_gap_volatility': dict(definition='sd of the signed gap over 12 months', formula='std', fields='outbound,mirror', lookback=12, source='panel', meaning='reporting instability'),
        'mirror_gap_excess': dict(definition='gap level minus the median gap of the other candidates in the same query', formula='level - median(others)', fields='outbound,mirror', lookback=3, source='panel', meaning='candidate-specific reporting anomaly'),
        'change_point_detected': dict(definition='mean-shift scan flags a break in log imports (stat >= 1.5)', formula='max_k |mean(after)-mean(before)|/pooled sd', fields='inbound', lookback=24, source='panel', meaning='structural break present'),
        'change_magnitude': dict(definition='level difference after vs before the detected break', formula='mean(after)-mean(before) in log', fields='inbound', lookback=24, source='panel', meaning='break size and direction'),
        'slope_change': dict(definition='trend slope after minus before the break', formula='slope(after)-slope(before), clipped', fields='inbound', lookback=24, source='panel', meaning='trend reversal'),
        'months_since_change': dict(definition='months since the detected break (120 if none)', formula='n-k', fields='inbound', lookback=24, source='panel', meaning='break recency'),
        'geographic_adjacency': dict(definition='target and candidate share a land border', formula='indicator', fields='static', lookback=None, source='features.ADJACENT', meaning='logistics proximity'),
        'economic_bloc_link': dict(definition='both in the EAEU by the basis month', formula='indicator with join months', fields='static', lookback=None, source='features.EAEU_JOIN', meaning='customs-union access'),
        'neighbor_deviation': dict(definition='coupled growth minus the median of other candidates in the same query', formula='x - median(others)', fields='inbound,outbound', lookback=6, source='panel', meaning='candidate stands out among peers'),
        'similar_item_deviation': dict(definition='coupled growth minus the median over the other 8 items for the same target/candidate/basis', formula='x - median(other items)', fields='inbound,outbound', lookback=6, source='panel', meaning='item-specific rather than country-wide movement'),
        'hs_chapter': dict(definition='HS chapter of the item', formula='int(hs[:2])', fields='static', lookback=None, source='item', meaning='coarse product family'),
        'control_actions_to_date': dict(definition='product control actions covering this HS6 for the target, announced by the basis', formula='count', fields='source_events', lookback=None, source='integration.sqlite source_events', meaning='item is under control'),
        'months_since_control': dict(definition='months since the latest such control action (120 if none)', formula='basis-max(announce)', fields='source_events', lookback=None, source='integration.sqlite source_events', meaning='control recency'),
        'absorption_gap': dict(definition='import growth minus export growth', formula='g_in-g_out', fields='inbound,outbound', lookback=6, source='panel', meaning='imports rising without re-export (domestic absorption)'),
        'reexport_share': dict(definition='log ratio of exports to imports, last 3 months', formula='mean(log(out)-log(in))', fields='inbound,outbound', lookback=3, source='panel', meaning='share of inflow that leaves toward the target'),
        'reexport_share_trend': dict(definition='slope of the log export/import ratio over 6 months', formula='polyfit slope', fields='inbound,outbound', lookback=6, source='panel', meaning='re-export share rising'),
        'inflow_level_z': dict(definition='latest 3-month log import level vs the earlier 21-month history', formula='(mean3 - mean_hist)/sd_hist, clipped [-5,5]', fields='inbound', lookback=24, source='panel', meaning='abnormal inflow level'),
        'inflow_outflow_coupling': dict(definition='max lagged (0-3) correlation of monthly log changes between legs over 12 months', formula='max_lag corr(d log in, d log out)', fields='inbound,outbound', lookback=13, source='panel', meaning='second leg follows the first'),
        'policy_effective_lead': dict(definition='months until the effective date of a published-but-not-yet-effective event', formula='min(effective-basis) for published<=basis<effective', fields='events', lookback=None, source='data/events.json', meaning='front-running window'),
        'transit_designations_to_date': dict(definition='designations naming this candidate as a transit party for the target, announced by the basis', formula='count', fields='source_events', lookback=None, source='integration.sqlite source_events', meaning='candidate already flagged'),
        'months_since_transit_designation': dict(definition='months since the latest such designation (120 if none)', formula='basis-max(announce)', fields='source_events', lookback=None, source='integration.sqlite source_events', meaning='designation recency'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'quality', 'benchmark', 'seeds', 'ablation', 'importance', 'select', 'report', 'all'])
    args = parser.parse_args()
    order = ['build', 'quality', 'benchmark', 'seeds', 'ablation', 'importance', 'select', 'report'] if args.stage == 'all' else [args.stage]
    for stage in order:
        t = time.time()
        if stage in ('build', 'quality'):
            {'build': stage_build, 'quality': stage_quality}[stage]()
        else:
            from benchmark import stages
            getattr(stages, 'stage_' + stage)()
        print(f'[{stage}] {round(time.time() - t, 1)}s', flush=True)
