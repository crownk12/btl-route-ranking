"""Point-in-time feature extraction for the 12F baseline and the expanded candidate set (benchmark v12).

Everything is computed from panel rows whose available_month <= basis (Panel.history), the eight
panel events (published <= basis) and the source_events table of data/integration.sqlite
(announce_date <= basis). Static tables (land adjacency, EAEU membership with join months) are
dated so that nothing later than the basis month enters a row.

The 12 baseline features are produced by common.Panel.features unchanged; the expanded
candidates are appended, and cross-candidate features are derived per query afterwards.
"""
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing

import numpy as np

from common import ROOT, FEATURES, ITEMS, Panel, mi

INTEGRATION = ROOT / 'data/integration.sqlite'
M49 = {'CHN': '156', 'KAZ': '398', 'TUR': '792', 'ARE': '784', 'ARM': '051', 'UZB': '860', 'KGZ': '417', 'IND': '356', 'VNM': '704'}
ISO = {v: k for k, v in M49.items()}

# Land borders between target and candidate (static geography; no dating needed).
ADJACENT = {('RUS', 'CHN'), ('RUS', 'KAZ'), ('IRN', 'ARM'), ('IRN', 'TUR'), ('PRK', 'CHN'),
            ('CHN', 'KAZ'), ('CHN', 'KGZ'), ('CHN', 'IND'), ('CHN', 'VNM')}
# Eurasian Economic Union membership with the month it applies from (link only when both are members by the basis).
EAEU_JOIN = {'RUS': '2015-01', 'KAZ': '2015-01', 'ARM': '2015-01', 'KGZ': '2015-08'}

GROUPS = {
    'mirror_gap': ['mirror_gap', 'mirror_gap_level', 'mirror_gap_slope_3m', 'mirror_gap_slope_6m', 'mirror_gap_volatility', 'mirror_gap_excess'],
    'change_point': ['import_volatility', 'export_volatility', 'change_point_detected', 'change_magnitude', 'slope_change', 'months_since_change'],
    'graph': ['geographic_adjacency', 'economic_bloc_link', 'neighbor_deviation', 'similar_item_deviation'],
    'item': ['hs_chapter', 'control_actions_to_date', 'months_since_control'],
    'absorption': ['absorption_gap', 'reexport_share', 'reexport_share_trend', 'inflow_level_z'],
    'flow': ['import_growth', 'export_growth', 'import_yoy', 'export_yoy', 'coupled_growth', 'inflow_outflow_coupling'],
    'policy_timing': ['event_count', 'months_since_event', 'policy_effective_lead', 'transit_designations_to_date', 'months_since_transit_designation'],
    'quality_task': ['coverage', 'horizon'],
}
NEW_CANDIDATES = ['mirror_gap_level', 'mirror_gap_slope_3m', 'mirror_gap_slope_6m', 'mirror_gap_volatility', 'mirror_gap_excess',
                  'change_point_detected', 'change_magnitude', 'slope_change', 'months_since_change',
                  'geographic_adjacency', 'economic_bloc_link', 'neighbor_deviation', 'similar_item_deviation',
                  'hs_chapter', 'control_actions_to_date', 'months_since_control',
                  'absorption_gap', 'reexport_share', 'reexport_share_trend', 'inflow_level_z',
                  'inflow_outflow_coupling',
                  'policy_effective_lead', 'transit_designations_to_date', 'months_since_transit_designation']
ALL_FEATURES = FEATURES + NEW_CANDIDATES
NONE = 120.0  # "no such event" distance, same convention as months_since_event


def group_of(name):
    return next((g for g, names in GROUPS.items() if name in names), None)


def load_source_events():
    """Product control actions (target, announce month, HS6 set) and transit designations (target, candidate, announce month)."""
    controls = []; designations = []
    if not INTEGRATION.exists():
        return controls, designations
    with closing(sqlite3.connect(INTEGRATION.as_uri() + '?mode=ro', uri=True)) as c:
        for event_id, target, published_month, payload in c.execute('select event_id,target,published_month,payload from source_events'):
            d = json.loads(payload)
            announce = d.get('announce_date') or ''
            if len(announce) < 7:
                continue
            month = mi(announce[:7])
            if d.get('event_type') == 'product_control_action':
                mapped = d.get('hs6_mapped')
                codes = set(str(mapped).split(';')) if mapped and mapped != 'None' else set()
                if codes & set(ITEMS):
                    controls.append((target, month, codes & set(ITEMS), event_id))
            elif d.get('event_type') == 'transit_designation':
                cand = ISO.get(str(d.get('transit_m49')).zfill(3))
                if cand:
                    designations.append((target, cand, month, event_id))
    return controls, designations


def _slope(y):
    """Least-squares slope against 0..n-1 (closed form; identical to np.polyfit degree 1)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean(); ym = y.mean()
    return float(np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm))


def _change_point(x, min_seg=6, threshold=1.5):
    """Mean-shift scan on a log series using only rows before the basis: (detected, magnitude, slope change, months since)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2 * min_seg:
        return 0.0, 0.0, 0.0, NONE
    idx = np.where(ok)[0]
    x = np.interp(np.arange(len(x)), idx, x[idx])
    n = len(x)
    ks = np.arange(min_seg, n - min_seg + 1)
    if len(ks) == 0:
        return 0.0, 0.0, 0.0, NONE
    cs = np.cumsum(x); cs2 = np.cumsum(x * x); total = cs[-1]; total2 = cs2[-1]
    na = ks.astype(float); nb = n - na
    mean_a = cs[ks - 1] / na; mean_b = (total - cs[ks - 1]) / nb
    var_a = np.maximum(cs2[ks - 1] / na - mean_a ** 2, 0.0); var_b = np.maximum((total2 - cs2[ks - 1]) / nb - mean_b ** 2, 0.0)
    stats = np.abs(mean_b - mean_a) / (np.sqrt((var_a + var_b) / 2) + 1e-6)
    j = int(np.argmax(stats))
    if stats[j] < threshold:
        return 0.0, 0.0, 0.0, NONE
    k = int(ks[j])
    magnitude = float(x[k:].mean() - x[:k].mean())
    slope_change = _slope(x[k:]) - _slope(x[:k])
    return 1.0, magnitude, float(np.clip(slope_change, -1, 1)), float(n - k)


def _coupling(inbound, outbound, max_lag=3):
    """Max lagged correlation between monthly log changes of the two legs (12-month window)."""
    a = np.asarray(inbound, float); b = np.asarray(outbound, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return 0.0
    idx = np.where(ok)[0]
    a = np.interp(np.arange(len(a)), idx, a[idx]); b = np.interp(np.arange(len(b)), idx, b[idx])
    da, db = np.diff(a), np.diff(b)
    best = 0.0
    for lag in range(0, max_lag + 1):
        x, y = da[:len(da) - lag] if lag else da, db[lag:]
        if len(x) >= 6:
            xc = x - x.mean(); yc = y - y.mean()
            sx = math.sqrt(np.dot(xc, xc) / len(x)); sy = math.sqrt(np.dot(yc, yc) / len(y))
            if sx > 1e-9 and sy > 1e-9:
                best = max(best, float(np.dot(xc, yc) / len(x) / (sx * sy)))
    return float(np.clip(best, -1, 1))


class FeatureBuilder:
    def __init__(self, panel=None):
        self.panel = panel or Panel()
        self.controls, self.designations = load_source_events()
        self.control_index = defaultdict(list)     # (target, hs) -> [announce months]
        for target, month, codes, _ in self.controls:
            for hs in codes:
                self.control_index[(target, hs)].append(month)
        self.designation_index = defaultdict(list)  # (target, candidate) -> [announce months]
        for target, cand, month, _ in self.designations:
            self.designation_index[(target, cand)].append(month)

    def row(self, key, basis, horizon):
        """12 baseline features (from Panel.features) plus the per-row expanded candidates; returns (vector dict, cur)."""
        target, hs, candidate = key
        hist = self.panel.history(key, basis, 24)
        x, cur, latest, events = self.panel.features(key, basis, horizon, hist)
        base = dict(zip(FEATURES, x.tolist()))
        inb = np.array([math.log(r['inbound']) if r['inbound'] else np.nan for r in hist])
        outb = np.array([math.log(r['outbound']) if r['outbound'] else np.nan for r in hist])
        mirror = np.array([math.log(r['mirror']) if r.get('mirror') else np.nan for r in hist])
        gap = outb - mirror  # signed log gap between the candidate's export report and the target-side mirror
        last12 = gap[-12:]; last6 = gap[-6:]; last3 = gap[-3:]
        f = {}
        f['mirror_gap_level'] = float(np.nanmean(last3)) if np.isfinite(last3).any() else 0.0
        f['mirror_gap_slope_3m'] = _slope(last3[np.isfinite(last3)]) if np.isfinite(last3).sum() >= 2 else 0.0
        f['mirror_gap_slope_6m'] = _slope(last6[np.isfinite(last6)]) if np.isfinite(last6).sum() >= 3 else 0.0
        f['mirror_gap_volatility'] = float(np.nanstd(last12)) if np.isfinite(last12).sum() >= 3 else 0.0
        detected, magnitude, slope_change, since = _change_point(inb)
        f['change_point_detected'] = detected; f['change_magnitude'] = magnitude; f['slope_change'] = slope_change; f['months_since_change'] = since
        f['geographic_adjacency'] = float((target, candidate) in ADJACENT)
        both = target in EAEU_JOIN and candidate in EAEU_JOIN
        f['economic_bloc_link'] = float(both and basis >= max(mi(EAEU_JOIN[target]), mi(EAEU_JOIN[candidate])))
        f['hs_chapter'] = float(hs[:2])
        controls = [m for m in self.control_index.get((target, hs), []) if m <= basis]
        f['control_actions_to_date'] = float(len(controls))
        f['months_since_control'] = float(min(NONE, basis - max(controls))) if controls else NONE
        f['absorption_gap'] = float(base['import_growth'] - base['export_growth'])
        share = (outb - inb)[-3:]
        f['reexport_share'] = float(np.nanmean(share)) if np.isfinite(share).any() else 0.0
        share6 = (outb - inb)[-6:]
        f['reexport_share_trend'] = _slope(share6[np.isfinite(share6)]) if np.isfinite(share6).sum() >= 3 else 0.0
        hist_in = inb[:-3]
        if np.isfinite(hist_in).sum() >= 6 and np.isfinite(inb[-3:]).any():
            f['inflow_level_z'] = float(np.clip((np.nanmean(inb[-3:]) - np.nanmean(hist_in)) / (np.nanstd(hist_in) + 1e-6), -5, 5))
        else:
            f['inflow_level_z'] = 0.0
        f['inflow_outflow_coupling'] = _coupling(inb[-13:], outb[-13:])
        pending = [mi(e['effective']) - basis for e in self.panel.events if e['target'] == target and mi(e['published']) <= basis < mi(e['effective'])]
        f['policy_effective_lead'] = float(min(pending)) if pending else 0.0
        desig = [m for m in self.designation_index.get((target, candidate), []) if m <= basis]
        f['transit_designations_to_date'] = float(len(desig))
        f['months_since_transit_designation'] = float(min(NONE, basis - max(desig))) if desig else NONE
        # placeholders filled per query
        f['mirror_gap_excess'] = 0.0; f['neighbor_deviation'] = 0.0; f['similar_item_deviation'] = 0.0
        base.update(f)
        return base, cur

    @staticmethod
    def add_cross_features(rows, keys, bases, horizons):
        """mirror_gap_excess and neighbor_deviation: row minus the median of the other candidates in the same query;
        similar_item_deviation: row minus the median over the other HS items of the same target/candidate/basis/horizon."""
        by_query = defaultdict(list); by_item = defaultdict(list)
        for i, (key, basis, h) in enumerate(zip(keys, bases, horizons)):
            by_query[(key[0], key[1], basis, h)].append(i)
            by_item[(key[0], key[2], basis, h)].append(i)
        for name, source, index in [('mirror_gap_excess', 'mirror_gap_level', by_query), ('neighbor_deviation', 'coupled_growth', by_query),
                                    ('similar_item_deviation', 'coupled_growth', by_item)]:
            for members in index.values():
                values = np.array([rows[i][source] for i in members], float)
                for j, i in enumerate(members):
                    others = np.delete(values, j)
                    rows[i][name] = float(values[j] - np.median(others)) if len(others) else 0.0
        return rows


def build_dataset(builder, basis_step=4, horizons=(3, 6, 12, 24, 60, 75), start=60):
    """Same sampling, labels and split rules as train.build_matrix, but returning every candidate feature.

    Cross-candidate features are computed over every candidate and item at a basis/horizon BEFORE the
    label filter, so the peer set is exactly what the server sees and never depends on label availability.
    """
    p = builder.panel
    rows = []; keys = []; bases = []; hors = []; curs = []
    for key in p.rows:
        for basis in range(start, p.end - 5, basis_step):
            for h in horizons:
                if basis + h + 2 + 5 > p.end:
                    continue
                feats, cur = builder.row(key, basis, h)
                rows.append(feats); keys.append(key); bases.append(basis); hors.append(h); curs.append(cur)
    rows = FeatureBuilder.add_cross_features(rows, keys, bases, hors)
    kept = []; Y = []; B = []; A = []; G = []; K = []; H = []
    for feats, key, basis, h, cur in zip(rows, keys, bases, hors, curs):
        final = {r['month']: r for r in p.rows[key] if r['revision'] == 1}
        endpoint = basis + h + 2
        future = []
        for col in ['inbound', 'outbound']:
            v = [final[m][col] for m in range(basis + h, basis + h + 3) if final[m][col] is not None]
            future.append(np.mean(v) if v else np.nan)
        if min(cur) <= 0 or not np.isfinite(future).all():
            continue
        ratios = np.array(future) / cur
        label = float(ratios[0] > 1.15 and ratios[1] > 1.20)
        kept.append(feats); Y.append([label, *np.clip(np.log(ratios), -2, 2)]); B.append(basis); A.append(endpoint + 5)
        G.append(f'{key[0]}:{key[1]}:{basis}:{h}'); K.append(key); H.append(h)
    rows = kept
    X = np.array([[r[name] for name in ALL_FEATURES] for r in rows], float)
    Y = np.array(Y); B = np.array(B); A = np.array(A); G = np.array(G); H = np.array(H)
    T = np.array([k[0] for k in K]); HS = np.array([k[1] for k in K]); C = np.array([k[2] for k in K])
    masks = {'train': A <= mi('2018-12'), 'validation': (B >= mi('2019-01')) & (A <= mi('2021-12')), 'test': (B >= mi('2022-01')) & (A <= p.end)}
    inner = {'fit': masks['train'] & (A <= mi('2015-12')), 'holdout': masks['train'] & (B >= mi('2016-01'))}
    return dict(X=X, Y=Y, B=B, A=A, G=G, H=H, T=T, HS=HS, C=C, masks=masks, inner=inner, features=ALL_FEATURES, panel_end=p.end)
