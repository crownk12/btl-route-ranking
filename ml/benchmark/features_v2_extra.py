"""Experimental feature families of benchmark v14 (FeatureBuilderV2(extended=True)); not used by the deployed model.

Round 2 of the benchmark (benchmark.exp_round2) tested three more ways of reading where a route stands in a growth episode:

  swing     turning points of the deseasonalised series (zigzag), a two-knot piecewise-linear fit of the last 30 months and a
            flat-then-linear fit of the last 14;
  episode   rise-and-fall templates matched to the last 18 months and averaged by how well each one fits;
  lags      the last 24 deseasonalised monthly levels against the current window.

None of them moved the primary metric beyond seed noise over the core set (inner folds 0.8172 core, 0.8180-0.8190 with any
of them), so the deployed feature set leaves them out. The code stays so the comparison can be rerun.
"""
import math

import numpy as np

SWING = 0.12       # a turning point is confirmed once the smoothed series has moved this far (log) the other way
PL_WINDOW = 30     # months of the two-knot trend fit
HINGE_WINDOW = 14  # months of the flat-then-linear fit
EP_WINDOW = 18                 # months matched against the episode templates
EP_LENGTHS = range(5, 14)      # months from the start of a climb to its crest (and from the crest back down)
EP_SIZE = (-0.6, 0.8)          # admissible crest size in log terms (a dip on the export leg is a negative episode)
EP_NOISE = 0.06                # month-to-month noise of a deseasonalised log series; floor of the fit's residual scale
EP_AHEAD = 80                  # months past the last known month for which a template is tabulated
LAGS = 24

STATIC_LEG = ['zz_state', 'zz_since', 'zz_move', 'zz_prev_size', 'zz_prev_len', 'zz_left', 'zz_time_left', 'zz_since_peak', 'zz_since_trough', 'zz_swings', 'zz_gap',
              'pl_s1', 'pl_s2', 'pl_s3', 'pl_since', 'pl_mid_len', 'pl_mid_rise', 'pl_last_move', 'pl_resid', 'pl_gain', 'hinge_slope', 'hinge_since', 'hinge_gain',
              'ep_height', 'ep_rising', 'ep_falling', 'ep_to_peak', 'ep_since_peak', 'ep_gain', 'ep_size', 'ep_peak', 'ep_length', 'ep_flat', 'ep_resid'] + [f'lag{k}' for k in range(1, LAGS + 1)]
DYNAMIC_LEG = ['zz_proj', 'pl_proj', 'hinge_proj', 'ep_proj', 'ep_proj_best', 'exp_ep']
DYNAMIC_PAIR = ['min_exp_ep']
RELATIVE = ['ep_proj_in', 'ep_proj_out', 'exp_ep_in', 'exp_ep_out', 'min_exp_ep', 'zz_proj_in', 'zz_proj_out', 'pl_proj_in', 'pl_proj_out']


def zigzag(ys, threshold=SWING):
    """Turning points of a series: [(index, value, +1 crest / -1 trough)], the last one being the latest confirmed turn."""
    pivots = []
    hi = lo = 0; direction = 0
    for i in range(1, len(ys)):
        if ys[i] > ys[hi]: hi = i
        if ys[i] < ys[lo]: lo = i
        if direction >= 0 and ys[hi] - ys[i] >= threshold and hi != i:
            if direction == 0 and lo < hi and ys[hi] - ys[lo] >= threshold: pivots.append((lo, float(ys[lo]), -1))
            pivots.append((hi, float(ys[hi]), 1)); direction = -1; lo = i
        elif direction <= 0 and ys[i] - ys[lo] >= threshold and lo != i:
            if direction == 0 and hi < lo and ys[hi] - ys[lo] >= threshold: pivots.append((hi, float(ys[hi]), 1))
            pivots.append((lo, float(ys[lo]), -1)); direction = 1; hi = i
    return pivots


def _bases(window, knots):
    """Orthonormal bases of the continuous piecewise-linear models on a fixed window, one per knot choice."""
    t = np.arange(window, dtype=float); out = []
    for ks in knots:
        design = np.column_stack([np.ones(window), t] + [np.maximum(0.0, t - k) for k in ks])
        q, _ = np.linalg.qr(design)
        out.append(q)
    return np.array(out)


_PL_KNOTS = [(a, b) for a in range(3, PL_WINDOW - 6) for b in range(a + 3, PL_WINDOW - 2)]
_PL_Q = _bases(PL_WINDOW, _PL_KNOTS)
_PL_LINE = _bases(PL_WINDOW, [()])[0]
_HINGE_KNOTS = list(range(2, HINGE_WINDOW - 2))


def _hinge_bases():
    t = np.arange(HINGE_WINDOW, dtype=float); out = []
    for k in _HINGE_KNOTS:  # flat until the knot, linear after it
        q, _ = np.linalg.qr(np.column_stack([np.ones(HINGE_WINDOW), np.maximum(0.0, t - k)]))
        out.append(q)
    return np.array(out)


_HINGE_Q = _hinge_bases()


def _episode_templates():
    """Triangular rise-and-fall shapes that are under way at the last known month (index 0): (peak offsets, lengths,
    centred unit shapes on the window, their norms, the shapes from the window start to EP_AHEAD months ahead)."""
    t = np.arange(-EP_WINDOW + 1, EP_AHEAD + 1, dtype=float)
    peaks, lengths, units, norms, shapes = [], [], [], [], []
    for length in EP_LENGTHS:
        for peak in range(-length - 5, length - 1):  # the climb has at least two months inside the window
            shape = np.maximum(0.0, 1.0 - np.abs(t - peak) / length)
            centred = shape[:EP_WINDOW] - shape[:EP_WINDOW].mean()
            norm = float(np.sqrt(centred @ centred))
            if norm < 1e-6:
                continue
            peaks.append(peak); lengths.append(length); units.append(centred / norm); norms.append(norm); shapes.append(shape)
    return np.array(peaks, float), np.array(lengths, float), np.array(units), np.array(norms), np.array(shapes)


_EP_PEAK, _EP_LENGTH, _EP_UNIT, _EP_NORM, _EP_SHAPE = _episode_templates()
_EP_NOW = EP_WINDOW - 1  # column of the last known month in _EP_SHAPE


def episode_fit(y):
    """Match the episode templates to the last EP_WINDOW points of a deseasonalised log series.

    Each template is fitted by least squares (level + size x shape); templates outside the admissible size are dropped and
    the rest, with the flat model, are weighted by exp(explained sum of squares / 2 sigma^2). Returns the weights, the
    fitted sizes, the best template and the summary dict; None when the series is shorter than the window.
    """
    if len(y) < EP_WINDOW:
        return None
    w = y[-EP_WINDOW:]; centred = w - w.mean()
    score = _EP_UNIT @ centred                      # signed projection on each unit template
    size = score / _EP_NORM
    ok = (size >= EP_SIZE[0]) & (size <= EP_SIZE[1])
    explained = np.where(ok, score ** 2, 0.0)
    total = float(centred @ centred)
    best = int(np.argmax(explained))
    sigma2 = max(EP_NOISE ** 2, (total - float(explained[best])) / EP_WINDOW)
    logw = np.where(ok, explained / (2 * sigma2), -np.inf)
    top = max(0.0, float(logw.max()))
    weight = np.exp(logw - top); flat = math.exp(-top)
    norm = float(weight.sum()) + flat
    weight /= norm; flat /= norm
    now = _EP_SHAPE[:, _EP_NOW]; step = now - _EP_SHAPE[:, _EP_NOW - 1]
    move = size * step
    active = now > 0
    out = {'ep_height': float(weight @ (size * now)), 'ep_rising': float(weight @ (move > 0.004)), 'ep_falling': float(weight @ (move < -0.004)),
           'ep_to_peak': float(weight @ np.maximum(0.0, _EP_PEAK)), 'ep_since_peak': float(weight @ (np.maximum(0.0, -_EP_PEAK) * active)),
           'ep_gain': float(explained[best] / (total + 1e-9)), 'ep_size': float(size[best]), 'ep_peak': float(_EP_PEAK[best]), 'ep_length': float(_EP_LENGTH[best]),
           'ep_flat': flat, 'ep_resid': math.sqrt(sigma2)}
    return weight, size, best, out


def episode_projection(fit, future, current):
    """Expected change of the episode's contribution between the current window and the forecast window, with no new episode:
    (weighted over the templates, best template alone). future/current: month offsets from the last known month."""
    weight, size, best, _ = fit
    f = np.clip(np.asarray(future) + _EP_NOW, 0, _EP_SHAPE.shape[1] - 1); c = np.asarray(current) + _EP_NOW
    change = size * (_EP_SHAPE[:, f].mean(axis=1) - _EP_SHAPE[:, c].mean(axis=1))
    return float(weight @ change), float(change[best])


def trend_fit(y):
    """Best two-knot continuous piecewise-linear fit of the last PL_WINDOW points:
    (slopes s1, s2, s3, knot indexes, fitted values, residual sd, gain over one line)."""
    w = y[-PL_WINDOW:]
    proj = np.einsum('kwj,w->kj', _PL_Q, w)
    best = int(np.argmax((proj ** 2).sum(axis=1)))
    fitted = _PL_Q[best] @ proj[best]
    a, b = _PL_KNOTS[best]
    s1 = fitted[1] - fitted[0]; s2 = fitted[a + 1] - fitted[a]; s3 = fitted[-1] - fitted[-2]
    sse = float(((w - fitted) ** 2).sum()); line = _PL_LINE @ (_PL_LINE.T @ w); sse_line = float(((w - line) ** 2).sum())
    return float(s1), float(s2), float(s3), a, b, fitted, math.sqrt(sse / PL_WINDOW), 1.0 - sse / (sse_line + 1e-9)


def hinge_fit(y):
    """Flat-then-linear fit of the last HINGE_WINDOW points: (slope after the knot, months since the knot, gain over a constant)."""
    w = y[-HINGE_WINDOW:]
    proj = np.einsum('kwj,w->kj', _HINGE_Q, w)
    best = int(np.argmax((proj ** 2).sum(axis=1)))
    fitted = _HINGE_Q[best] @ proj[best]
    sse = float(((w - fitted) ** 2).sum()); flat = float(((w - w.mean()) ** 2).sum())
    return float(fitted[-1] - fitted[-2]), float(HINGE_WINDOW - 1 - _HINGE_KNOTS[best]), 1.0 - sse / (flat + 1e-9)


def static(y, ys, cur_d):
    """Static values over STATIC_LEG for one leg, plus the state the horizon projections need."""
    n = len(y)
    f = dict.fromkeys(STATIC_LEG, 0.0)
    pivots = zigzag(ys); end = len(ys) - 1
    crests = [p for p in pivots if p[2] > 0]; troughs = [p for p in pivots if p[2] < 0]
    f['zz_since_peak'] = float(min(60, end - crests[-1][0])) if crests else 60.0
    f['zz_since_trough'] = float(min(60, end - troughs[-1][0])) if troughs else 60.0
    f['zz_swings'] = float(len(pivots))
    if len(crests) >= 2:
        f['zz_gap'] = float(np.mean(np.diff([p[0] for p in crests])))
    zz_left = zz_time_left = 0.0
    if pivots:
        at, value, kind = pivots[-1]
        f['zz_state'] = float(-kind)                      # +1 climbing from a trough, -1 falling from a crest
        f['zz_since'] = float(end - at); f['zz_move'] = float(ys[-1] - value)
        if len(pivots) >= 2:
            f['zz_prev_size'] = float(value - pivots[-2][1]); f['zz_prev_len'] = float(at - pivots[-2][0])
            if kind > 0:  # falling back after a climb: what is left of the climb to give back, and for how long at the climb's pace
                zz_left = max(0.0, f['zz_prev_size'] + f['zz_move']); zz_time_left = max(0.0, f['zz_prev_len'] - f['zz_since'])
    f['zz_left'] = zz_left; f['zz_time_left'] = zz_time_left
    last_slope = None
    if n >= PL_WINDOW:
        s1, s2, s3, ka, kb, fitted, resid, gain = trend_fit(y)
        last_slope = s3
        f['pl_s1'] = s1; f['pl_s2'] = s2; f['pl_s3'] = s3; f['pl_since'] = float(PL_WINDOW - 1 - kb); f['pl_mid_len'] = float(kb - ka)
        f['pl_mid_rise'] = float(fitted[kb] - fitted[ka]); f['pl_last_move'] = float(fitted[-1] - fitted[kb]); f['pl_resid'] = resid; f['pl_gain'] = gain
    if n >= HINGE_WINDOW:
        f['hinge_slope'], f['hinge_since'], f['hinge_gain'] = hinge_fit(y)
    episode = episode_fit(y)
    if episode is not None:
        f.update(episode[3])
    for k in range(1, min(LAGS, n) + 1):
        f[f'lag{k}'] = float(y[-k] - cur_d)
    return f, dict(pivots=pivots, zz_left=zz_left, last_slope=last_slope, episode=episode)


def dynamic(f, state, ahead, future_offsets, current_offsets, seas_delta):
    """Horizon values over DYNAMIC_LEG. A fall is carried on at its pace until the preceding climb is given back; a climb
    is carried on for at most six months; the episode projection assumes no new episode."""
    pivots = state['pivots']
    zz_proj = 0.0
    if pivots and f['zz_since'] > 0:
        pace = f['zz_move'] / f['zz_since']
        zz_proj = -min(state['zz_left'], -pace * ahead) if pivots[-1][2] > 0 else pace * min(ahead, 6.0)
    pl_proj = 0.0
    s3 = state['last_slope']
    if s3 is not None:
        if s3 < 0 and f['pl_mid_rise'] > 0:
            pl_proj = -min(max(0.0, f['pl_mid_rise'] + f['pl_last_move']), -s3 * ahead)
        else:
            pl_proj = s3 * min(ahead, 6.0)
    hinge_proj = f['hinge_slope'] * min(ahead, 6.0)
    ep_proj = ep_best = 0.0
    if state['episode'] is not None:
        ep_proj, ep_best = episode_projection(state['episode'], future_offsets, current_offsets)
    return [zz_proj, pl_proj, hinge_proj, ep_proj, ep_best, seas_delta + ep_proj]
