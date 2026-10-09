"""Point-in-time feature set v2 (benchmark v14): the 36 v1 candidates plus time-series, horizon and peer features.

The v1 features describe the last 24 months with a handful of growth ratios. The label asks something more specific:
whether the mean of months b+h … b+h+2 will exceed the latest available three-month mean by 15 % (imports) and 20 %
(exports). Three things decide that and v1 measured none of them directly:

  seasonality   where the forecast window sits in the yearly cycle compared with the current window;
  cycle state   whether the route is climbing, at a crest, or falling back (level against its own history, slopes,
                time since the highest and lowest point, both legs and the lead of imports over exports);
  current gap   how far the latest three months stand from the route's own trend (the label's denominator).

Everything is computed from Panel.history(key, basis, 60): rows with available_month <= basis only. One code path,
`FeatureBuilderV2.block`, builds a whole target x basis block (every item, candidate and horizon) for both the
training matrix and the server, so peer deviations always see the same peers and nothing depends on label availability.
`extended=True` adds the experimental families of features_v2_extra (benchmark round 2; not deployed).

`common=True` (export-surge track v2) adds the export leg seen without the movement its target's routes share. A policy
response lifts every route of a target over the same months, each by its own amount, and for years afterwards the level,
range and slope features of the 60-month history read that lift as a growth episode. The common movement is the
month-by-month median, across all routes of the target, of each route's centred deseasonalised log series; a route's
loading on it is a ridge regression towards 1; the leg features are recomputed on the series with loading x common
movement removed (`adj_*_out`), next to the loading and the recent course of the removed part (`fac_*_out`).
"""
import math

import numpy as np

from common import ITEMS, candidates_for, mi
from benchmark.features import FeatureBuilder, ALL_FEATURES as V1_FEATURES, _slope

HISTORY = 60
THRESHOLD = (math.log(1.15), math.log(1.20))  # the label's import and export thresholds, in log ratio
LEGS = ('in', 'out')
OWN_SEASON_GAP = 0.15  # a leg whose seasonal coefficients differ from the item's shared ones by more than this keeps its own
_KERNEL = np.r_[0.5, np.ones(11), 0.5] / 12.0  # centred 2x12 moving average: removes a 12-month cycle exactly

STATIC_LEG = ['amp', 'dev6', 'dev12', 'dev24', 'dev48', 'slope3', 'slope6', 'slope12', 'slope_long', 'accel', 'mom3', 'mom6', 'mom12',
              'since_max', 'since_min', 'pos', 'drop', 'rise', 'height', 'run_up', 'run_dn', 'vol', 'last_dev', 'ncur', 'miss12', 'ev_gain']
DYNAMIC_LEG = ['seas_delta', 'snaive_mean', 'snaive_first', 'snaive_n', 'trend_h', 'exp_base']
STATIC_PAIR = ['lead_gap', 'slope_gap6', 'mom_gap3', 'report_lag']
DYNAMIC_PAIR = ['min_exp', 'min_seas', 'event_phase_h']
# Deviation from the median of the other candidates in the same query (target x item x basis x horizon).
RELATIVE = ['exp_base_in', 'exp_base_out', 'min_exp', 'snaive_mean_in', 'snaive_mean_out', 'dev12_in', 'dev12_out', 'dev48_in', 'dev48_out',
            'slope6_in', 'slope6_out', 'mom3_in', 'mom3_out', 'height_in', 'height_out', 'pos_in', 'pos_out', 'lead_gap']
V1_STATIC = [n for n in V1_FEATURES if n != 'horizon']
# Export leg with the target-wide common movement removed (FeatureBuilderV2(common=True)).
COMMON_LEG = 1
COMMON_MIN_ROUTES = 5      # months reported by fewer routes take the common movement from their neighbours
COMMON_RIDGE = 0.03 ** 2   # per month; with a flat common movement the loading stays at 1
COMMON_FACTOR = ['load', 'now', 'slope6', 'slope3']
COMMON_STATIC = [n for n in STATIC_LEG if n != 'ev_gain']  # no event month is passed for the adjusted series
COMMON_NAMES = [f'adj_{n}_out' for n in COMMON_STATIC + DYNAMIC_LEG] + [f'fac_{n}_out' for n in COMMON_FACTOR]


def layout(extended=False, common=False):
    """Name lists of one build: per-leg static and dynamic names, pair names, peer-deviation sources, and the full feature list."""
    static_leg, dynamic_leg, dynamic_pair, relative = list(STATIC_LEG), list(DYNAMIC_LEG), list(DYNAMIC_PAIR), list(RELATIVE)
    if extended:
        from benchmark import features_v2_extra as extra
        static_leg += extra.STATIC_LEG; dynamic_leg += extra.DYNAMIC_LEG; dynamic_pair += extra.DYNAMIC_PAIR; relative = extra.RELATIVE + relative
    static = [f'ts_{n}_{leg}' for leg in LEGS for n in static_leg] + [f'ts_{n}' for n in STATIC_PAIR]
    dynamic = [f'ts_{n}_{leg}' for leg in LEGS for n in dynamic_leg] + [f'ts_{n}' for n in dynamic_pair]
    adjusted = COMMON_NAMES if common else []
    return dict(static_leg=static_leg, dynamic_leg=dynamic_leg, dynamic_pair=dynamic_pair, relative=relative, static=static, dynamic=dynamic, common=adjusted,
                names=list(V1_FEATURES) + static + dynamic + [f'rel_{n}' for n in relative] + adjusted)


ALL_FEATURES_V2 = layout()['names']
RELATIVE_NAMES = [f'rel_{n}' for n in RELATIVE]

GROUPS_V2 = {
    'seasonal': [f'ts_{n}_{leg}' for leg in LEGS for n in ('amp', 'seas_delta', 'snaive_mean', 'snaive_first', 'snaive_n')] + ['ts_min_seas'],
    'cycle_state': [f'ts_{n}_{leg}' for leg in LEGS for n in ('slope3', 'slope6', 'slope12', 'slope_long', 'accel', 'mom3', 'mom6', 'mom12', 'since_max', 'since_min',
                                                               'pos', 'drop', 'rise', 'height', 'run_up', 'run_dn', 'vol', 'trend_h')],
    'current_gap': [f'ts_{n}_{leg}' for leg in LEGS for n in ('dev6', 'dev12', 'dev24', 'dev48', 'last_dev', 'exp_base')] + ['ts_min_exp'],
    'leg_lead': ['ts_lead_gap', 'ts_slope_gap6', 'ts_mom_gap3'],
    'policy_response': ['ts_ev_gain_in', 'ts_ev_gain_out', 'ts_event_phase_h'],
    'reporting': ['ts_ncur_in', 'ts_ncur_out', 'ts_miss12_in', 'ts_miss12_out', 'ts_report_lag'],
    'peer_relative': RELATIVE_NAMES,
    'common_movement_removed': COMMON_NAMES,
}


def group_of_v2(name):
    from benchmark.features import group_of
    return next((g for g, names in GROUPS_V2.items() if name in names), None) or group_of(name)


def _season(m, coef):
    return coef[0] * np.sin(m * np.pi / 6) + coef[1] * np.cos(m * np.pi / 6)


def _log_filled(raw):
    """(log series, finite mask, log series with gaps interpolated) or None when fewer than six months are known."""
    with np.errstate(invalid='ignore', divide='ignore'):
        x = np.log(np.where(raw > 0, raw, np.nan))
    fin = np.isfinite(x)
    if fin.sum() < 6:
        return None
    known = np.flatnonzero(fin)
    return x, fin, np.interp(np.arange(len(x)), known, x[known])


def seasonal_residual(raw, months):
    """Log series minus its centred 12-month moving average, on the months where both exist: {month: residual}."""
    filled = _log_filled(raw)
    n = len(raw)
    if filled is None or n < 37:
        return {}
    x, fin, xf = filled
    centre = slice(6, n - 6)
    resid = xf[centre] - np.convolve(xf, _KERNEL, 'valid'); use = fin[centre]
    return dict(zip(months[centre][use].tolist(), resid[use].tolist()))


def harmonic(residual):
    """First-harmonic (12-month) coefficients of a {month: residual} map; zeros when there are fewer than 24 points."""
    if len(residual) < 24:
        return np.zeros(2)
    mm = np.array(list(residual)); r = np.array(list(residual.values())); r = r - r.mean()
    design = np.column_stack([np.sin(mm * np.pi / 6), np.cos(mm * np.pi / 6)])
    return np.linalg.lstsq(design, r, rcond=None)[0]


def pooled_harmonic(residuals):
    """Seasonal coefficients shared by the routes of one item and leg: the harmonic of the month-by-month median residual.
    The median keeps one route's growth episode, or one route with a seasonality of its own, from bending the shared cycle."""
    by_month = {}
    for residual in residuals:
        for m, v in residual.items():
            by_month.setdefault(m, []).append(v)
    return harmonic({m: float(np.median(v)) for m, v in by_month.items() if len(v) >= 3})


def leg_features(raw, months, basis, horizons, event_month=None, coef=None, extended=False):
    """One leg of one route. raw: as-of values up to the last known month (NaN where missing); months: their indexes.
    coef: seasonal coefficients to deseasonalise with (default: the leg's own harmonic).

    Returns (static vector, dynamic matrix len(horizons) x dynamic names) in the order of layout(extended). A leg with
    fewer than six known months, or none in the current three-month window, returns zeros (the same default v1 uses).
    """
    extra = None
    static_names, dynamic_names = STATIC_LEG, DYNAMIC_LEG
    if extended:
        from benchmark import features_v2_extra as extra
        static_names = STATIC_LEG + extra.STATIC_LEG; dynamic_names = DYNAMIC_LEG + extra.DYNAMIC_LEG
    static = np.zeros(len(static_names)); dynamic = np.zeros((len(horizons), len(dynamic_names)))
    n = len(raw)
    filled = _log_filled(raw)
    if filled is None:
        return static, dynamic
    x, fin, xf = filled
    window = np.arange(max(0, n - 3), n)
    cur_idx = window[fin[window]]
    if not len(cur_idx):
        return static, dynamic
    if coef is None:
        coef = harmonic(seasonal_residual(raw, months))
    season = _season(months, coef)
    y = xf - season
    cur_log = math.log(float(np.mean(raw[cur_idx])))
    s_cur = float(season[cur_idx].mean())
    cur_d = cur_log - s_cur
    f = dict.fromkeys(STATIC_LEG, 0.0)
    f['amp'] = float(np.hypot(*coef))
    for k in (6, 12, 24, 48):
        f[f'dev{k}'] = cur_d - float(y[-k:].mean())
    f['slope3'] = _slope(y[-3:]); f['slope6'] = _slope(y[-6:]); f['slope12'] = _slope(y[-12:]); f['slope_long'] = _slope(y[-48:])
    f['accel'] = f['slope3'] - f['slope12']
    f['mom3'] = float(y[-3:].mean() - y[-6:-3].mean()) if n >= 6 else 0.0
    f['mom6'] = float(y[-6:].mean() - y[-12:-6].mean()) if n >= 12 else 0.0
    f['mom12'] = float(y[-12:].mean() - y[-24:-12].mean()) if n >= 24 else 0.0
    ys = np.convolve(y, np.ones(3) / 3, 'valid') if n >= 3 else y
    w = ys[-36:]
    top, bottom = int(np.argmax(w)), int(np.argmin(w))
    f['since_max'] = float(len(w) - 1 - top); f['since_min'] = float(len(w) - 1 - bottom)
    f['pos'] = float((w[-1] - w[bottom]) / (w[top] - w[bottom] + 1e-6))
    f['drop'] = float(w[-1] - w[top]); f['rise'] = float(w[-1] - w[bottom])
    f['height'] = float(ys[-1] - np.percentile(ys[-48:], 20))
    steps = np.diff(ys[-25:])
    up = 0
    while up < len(steps) and steps[-1 - up] > 0: up += 1
    down = 0
    while down < len(steps) and steps[-1 - down] < 0: down += 1
    f['run_up'] = float(up); f['run_dn'] = float(down)
    f['vol'] = float(np.std(np.diff(y[-13:]))) if n >= 3 else 0.0
    f['last_dev'] = float(y[-1] - y[-3:].mean())
    f['ncur'] = float(len(cur_idx)); f['miss12'] = float(1 - fin[-12:].mean())
    if event_month is not None:
        e = int(event_month - months[0])
        if 3 <= e < n:
            f['ev_gain'] = float(y[-3:].mean() - y[e - 3:e].mean())
    state = None
    if extra is not None:
        more, state = extra.static(y, ys, cur_d)
        f.update(more)
    static[:] = [f[k] for k in static_names]
    cur_months = months[cur_idx]; first = int(months[0]); last = int(months[-1])
    for i, h in enumerate(horizons):
        future = basis + h + np.arange(3)
        seas_delta = float(_season(future, coef).mean()) - s_cur
        naive = []  # the same change of calendar position in earlier years
        for years in range(1, 5):
            fm = future - 12 * years; cm = cur_months - 12 * years
            if fm.max() > last:
                continue
            if fm.min() < first or cm.min() < first:
                break
            fv = raw[fm - first]; cv = raw[cm - first]
            if np.isfinite(fv).any() and np.isfinite(cv).any():
                a, b = float(np.nanmean(fv)), float(np.nanmean(cv))
                if a > 0 and b > 0:
                    naive.append(math.log(a / b))
            if len(naive) == 3:
                break
        ahead = float(basis + h + 1 - cur_months.mean())  # months between the centres of the current and the forecast window
        values = [seas_delta, float(np.mean(naive)) if naive else 0.0, naive[0] if naive else 0.0, float(len(naive)), f['slope_long'] * ahead, seas_delta - f['dev24']]
        if extra is not None:
            values += extra.dynamic(f, state, ahead, future - last, cur_idx - (n - 1), seas_delta)
        dynamic[i] = values
    return static, dynamic


def common_movement(routes, basis, leg=COMMON_LEG):
    """Common movement of one leg across the routes of a target, and each route's centred series.

    routes: [(values [months, 2] or None, month indexes, seasonal coefficients per leg)]. Returns (common [HISTORY] for the
    months basis-HISTORY+1 … basis, centred [routes, HISTORY] with NaN where a route has no value).
    """
    first = basis - HISTORY + 1
    centred = np.full((len(routes), HISTORY), np.nan)
    for r, (a, months, coefs) in enumerate(routes):
        filled = _log_filled(a[:, leg]) if a is not None else None
        if filled is None:
            continue
        x, fin, xf = filled
        y = (xf - _season(months, coefs[leg]))[fin]
        centred[r, months[fin] - first] = y - y.mean()
    count = np.isfinite(centred).sum(0)
    enough = np.flatnonzero(count >= COMMON_MIN_ROUTES)
    if not len(enough):
        return np.zeros(HISTORY), centred
    median = np.nanmedian(centred[:, enough], axis=0)
    return np.interp(np.arange(HISTORY), enough, median), centred


def common_features(raw, months, basis, horizons, coef, common, centred):
    """One route's leg without the common movement: [horizons, COMMON_NAMES] (zeros for a route without a series)."""
    out = np.zeros((len(horizons), len(COMMON_NAMES)))
    if raw is None:
        return out
    at = months - (basis - HISTORY + 1)
    f = common[at]; z = centred[at]; ok = np.isfinite(z)
    load = 1.0
    if ok.sum() >= 6:
        fc = f[ok] - f[ok].mean(); zc = z[ok] - z[ok].mean(); ridge = COMMON_RIDGE * ok.sum()
        load = float(np.clip((zc @ fc + ridge) / (fc @ fc + ridge), 0.0, 3.0))
    removed = load * f
    static, dynamic = leg_features(raw * np.exp(-removed), months, basis, horizons, None, coef)
    keep = [STATIC_LEG.index(n) for n in COMMON_STATIC]; ns, nd = len(keep), len(DYNAMIC_LEG)
    factor = [load, float(removed[-3:].mean() - removed[-15:-3].mean()) if len(removed) >= 15 else 0.0, _slope(removed[-6:]), _slope(removed[-3:])]
    out[:, :ns] = static[keep]; out[:, ns:ns + nd] = dynamic; out[:, ns + nd:] = factor
    return out


def _others_median_deviation(values, axis):
    """values minus the median of the other members along `axis` (0 when a member has no peer)."""
    n = values.shape[axis]
    out = np.zeros_like(values)
    if n < 2:
        return out
    moved = np.moveaxis(values, axis, 0); result = np.moveaxis(out, axis, 0)
    for j in range(n):
        result[j] = moved[j] - np.median(np.delete(moved, j, axis=0), axis=0)
    return out


class FeatureBuilderV2:
    """v1 rows (FeatureBuilder) plus the v2 time-series block; `block` is the only entry point used for training and serving."""

    def __init__(self, panel=None, seasonal='pooled', extended=False, common=False):
        self.v1 = FeatureBuilder(panel)
        self.panel = self.v1.panel
        self.seasonal = seasonal  # 'own': each leg's harmonic; 'pooled': the item's shared harmonic unless the leg clearly has its own
        self.extended = extended
        self.common = common  # adds the export leg without the target-wide common movement (COMMON_NAMES)
        self.layout = layout(extended, common)
        self.names = self.layout['names']

    def _series(self, key, basis):
        """As-of import and export values up to the last known month: (values [months, 2], month indexes), or (None, None)."""
        hist = self.panel.history(key, basis, HISTORY)
        a = np.array([[r['inbound'] if r['inbound'] is not None else np.nan, r['outbound'] if r['outbound'] is not None else np.nan] for r in hist])
        months = np.array([r['month'] for r in hist])
        known = np.flatnonzero(np.isfinite(a).any(axis=1))
        if not len(known):
            return None, None
        return a[:known[-1] + 1], months[:known[-1] + 1]

    def _prepare(self, target, hs, candidates, basis):
        """As-of series and seasonal coefficients of every candidate of one item: [(values, month indexes, coefficients per leg)]."""
        series = [self._series((target, hs, candidate), basis) for candidate in candidates]
        residuals = [[seasonal_residual(a[:, j], months) if a is not None else {} for j in range(2)] for a, months in series]
        own = [[harmonic(r[j]) for j in range(2)] for r in residuals]
        shared = [pooled_harmonic([r[j] for r in residuals]) for j in range(2)]
        out = []
        for c in range(len(candidates)):
            coefs = own[c]
            if self.seasonal == 'pooled':
                # A leg keeps its own cycle only when it stands clearly apart from the item's shared one.
                coefs = [own[c][j] if np.hypot(*(own[c][j] - shared[j])) > OWN_SEASON_GAP else shared[j] for j in range(2)]
            out.append((*series[c], coefs))
        return out

    def _route(self, key, basis, horizons, a, months, coefs):
        """(v1 dict, cur, static vector, dynamic matrix) for one route; coefs: seasonal coefficients per leg."""
        target = key[0]; lay = self.layout
        v1, cur = self.v1.row(key, basis, horizons[0])
        static = np.zeros(len(lay['static'])); dynamic = np.zeros((len(horizons), len(lay['dynamic'])))
        if a is None:
            return v1, cur, static, dynamic
        effective = [mi(e['effective']) for e in self.panel.events if e['target'] == target and mi(e['published']) <= basis and mi(e['effective']) <= basis]
        event_month = max(effective) if effective else None
        nl, nd = len(lay['static_leg']), len(lay['dynamic_leg'])
        legs = [leg_features(a[:, j], months, basis, horizons, event_month, coefs[j], self.extended) for j in range(2)]
        for j, (s, d) in enumerate(legs):
            static[j * nl:(j + 1) * nl] = s; dynamic[:, j * nd:(j + 1) * nd] = d
        col = {n: i for i, n in enumerate(lay['static_leg'])}
        (si, di), (so, do) = legs
        static[2 * nl:] = [si[col['height']] - so[col['height']], si[col['slope6']] - so[col['slope6']], si[col['mom3']] - so[col['mom3']], float(basis - months[-1])]
        dcol = {n: i for i, n in enumerate(lay['dynamic_leg'])}
        age = float(basis - event_month) if event_month is not None else 120.0
        joint = lambda name: np.minimum(di[:, dcol[name]] - THRESHOLD[0], do[:, dcol[name]] - THRESHOLD[1])  # the weaker leg against its threshold
        pair = [joint('exp_base'), joint('seas_delta'), min(age, 120.0) + np.array(horizons, float)]
        if self.extended:
            pair.append(joint('exp_ep'))
        dynamic[:, 2 * nd:] = np.column_stack(pair)
        return v1, cur, static, dynamic

    def block(self, target, basis, horizons, items=None):
        """Every item x candidate x horizon of one target at one basis month.

        Returns dict(X=[items, candidates, horizons, features], cur=[items, candidates, 2], items, candidates, names).
        """
        items = list(items or ITEMS); candidates = candidates_for(target); horizons = list(horizons); lay = self.layout
        ni, nc, nh = len(items), len(candidates), len(horizons)
        index = {n: i for i, n in enumerate(self.names)}
        X = np.zeros((ni, nc, nh, len(self.names))); cur = np.zeros((ni, nc, 2))
        v1_cols = [index[n] for n in V1_STATIC]; s0 = index[lay['static'][0]]; d0 = index[lay['dynamic'][0]]
        prepared = {hs: self._prepare(target, hs, candidates, basis) for hs in items}
        if self.common:
            # The common movement always comes from the active item set, whichever items the caller asked for.
            pool = [route for hs in ITEMS for route in (prepared.get(hs) or self._prepare(target, hs, candidates, basis))]
            common, centred = common_movement(pool, basis)
            position = {hs: k * nc for k, hs in enumerate(ITEMS)}; a0 = index[lay['common'][0]]
        for i, hs in enumerate(items):
            for c, candidate in enumerate(candidates):
                a, months, coefs = prepared[hs][c]
                v1, cur[i, c], static, dynamic = self._route((target, hs, candidate), basis, horizons, a, months, coefs)
                X[i, c, :, v1_cols] = np.array([v1[n] for n in V1_STATIC])[:, None]
                X[i, c, :, s0:s0 + len(lay['static'])] = static
                X[i, c, :, d0:d0 + len(lay['dynamic'])] = dynamic
                if self.common and hs in position:
                    X[i, c, :, a0:a0 + len(lay['common'])] = common_features(a[:, COMMON_LEG] if a is not None else None, months, basis, horizons, coefs[COMMON_LEG],
                                                                             common, centred[position[hs] + c])
        X[..., index['horizon']] = np.array(horizons, float)
        # v1 peer features (same definitions as FeatureBuilder.add_cross_features)
        X[..., index['mirror_gap_excess']] = _others_median_deviation(X[..., index['mirror_gap_level']], 1)
        X[..., index['neighbor_deviation']] = _others_median_deviation(X[..., index['coupled_growth']], 1)
        X[..., index['similar_item_deviation']] = _others_median_deviation(X[..., index['coupled_growth']], 0)
        for name in lay['relative']:
            X[..., index[f'rel_{name}']] = _others_median_deviation(X[..., index[f'ts_{name}']], 1)
        return dict(X=X, cur=cur, items=items, candidates=candidates, horizons=horizons, names=self.names)


def build_dataset_v2(builder, basis_step=4, horizons=(3, 6, 12, 24, 60, 75), start=60, targets=None):
    """Same sampling, label and split rules as v1 build_dataset (so the validation and test queries are the same rows),
    with the v2 feature block. basis_step < 4 only adds training bases: validation and test keep the every-4-months grid."""
    from common import TARGETS
    p = builder.panel
    horizons = list(horizons)
    final = {key: {r['month']: (r['inbound'], r['outbound']) for r in rows if r['revision'] == 1} for key, rows in p.rows.items()}
    X = []; Y = []; R = []; B = []; A = []; G = []; H = []; T = []; HS = []; C = []
    for target in (targets or TARGETS):
        for basis in range(start, p.end - 5, basis_step):
            hs_ok = [h for h in horizons if basis + h + 2 + 5 <= p.end]
            if not hs_ok:
                continue
            blk = builder.block(target, basis, hs_ok)
            for i, hs in enumerate(blk['items']):
                for c, candidate in enumerate(blk['candidates']):
                    cur = blk['cur'][i, c]
                    if min(cur) <= 0:
                        continue
                    series = final[(target, hs, candidate)]
                    for k, h in enumerate(hs_ok):
                        future = []
                        for col in (0, 1):
                            v = [series[m][col] for m in range(basis + h, basis + h + 3) if series[m][col] is not None]
                            future.append(np.mean(v) if v else np.nan)
                        if not np.isfinite(future).all():
                            continue
                        ratios = np.array(future) / cur
                        label = float(ratios[0] > 1.15 and ratios[1] > 1.20)
                        X.append(blk['X'][i, c, k].astype(np.float32)); Y.append([label, *np.clip(np.log(ratios), -2, 2)]); R.append(np.log(ratios))
                        B.append(basis); A.append(basis + h + 7); G.append(f'{target}:{hs}:{basis}:{h}'); H.append(h); T.append(target); HS.append(hs); C.append(candidate)
    X = np.array(X, np.float32); Y = np.array(Y); R = np.array(R); B = np.array(B); A = np.array(A); G = np.array(G); H = np.array(H)
    T = np.array(T); HS = np.array(HS); C = np.array(C)
    grid = (B - start) % 4 == 0  # the v1 evaluation grid
    masks = {'train': A <= mi('2018-12'), 'validation': (B >= mi('2019-01')) & (A <= mi('2021-12')) & grid, 'test': (B >= mi('2022-01')) & (A <= p.end) & grid}
    inner = {'fit': masks['train'] & (A <= mi('2015-12')), 'holdout': masks['train'] & (B >= mi('2016-01')) & grid}
    return dict(X=X, Y=Y, R=R, B=B, A=A, G=G, H=H, T=T, HS=HS, C=C, masks=masks, inner=inner, features=list(builder.names), panel_end=p.end, grid=grid)
