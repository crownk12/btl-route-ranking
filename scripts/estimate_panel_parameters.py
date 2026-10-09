"""Estimate generator parameters for the demo panel from the observed rows only (track B-1).

Reads data/integration.sqlite (kind='observed', deduplicated to the latest retrieval per
reporter/partner/hs/month/flow) and writes data/panel-parameters.json. Nothing here
touches the panel, the model or the source databases.

Two legs of every panel route are calibrated:
  in  = candidate imports from China (reporter=candidate, partner=CHN, flow M). China is
        the only supplier with observed reports; the map's supplier label is a separate
        screen definition (docs/SUPPLIER_DEFINITION_DECISION.md).
  out = candidate exports to the target (reporter=candidate, partner=target, flow X).

Per series with enough months: a smooth level path (12-month centered rolling median of
log USD) used as an anchor, plus seasonal amplitude/phase, AR(1), residual sd and drift.
Pooled by HS and leg for routes without observations. Event responses, missingness,
mirror gaps and growth episodes are pooled across series. Parameters that cannot be
estimated keep the legacy generator value and are listed under `unestimated`.

Registered evaluation cohorts are excluded from estimation; none exists yet, which is
recorded in the output.
"""
import datetime as dt
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));import btl_paths
from common import TARGETS, CANDIDATES, ITEMS, mi

SOURCE = ROOT / 'data/integration.sqlite'
OUTPUT = ROOT / 'data/panel-parameters.json'
READINESS = ROOT / 'artifacts/integration/next-stage/evaluation-readiness.json'
MIN_ANCHOR_MONTHS = 12
MIN_STRUCTURE_MONTHS = 24
LEGACY = dict(level_mean=10.0, level_sd=0.6, out_in_ratio=math.log(0.25), drift=0.00115, seasonal_amp_in=0.08, seasonal_amp_out=0.06,
              ar1=0.8, resid_sd=0.045, missing=0.05, missing_prk=0.18, mirror_log_sd=0.13, event_amp_in=(0.05, 0.7), event_amp_out=(-0.3, 0.9),
              event_lag=(1, 10), pulse_gap=(28, 55), pulse_duration=(10, 24), pulse_amp=(0.25, 1.2), pulse_kinds=(0.55, 0.3, 0.15))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_series():
    """{(leg, target, hs, candidate): {month: log_value}} using the latest retrieval per cell; zeros excluded from levels."""
    rows = defaultdict(dict)
    zeros = defaultdict(set)
    with closing(sqlite3.connect(SOURCE.as_uri() + '?mode=ro', uri=True)) as c:
        c.row_factory = sqlite3.Row
        query = ("select reporter, partner, hs, month, flow, value_usd, retrieved from observations where kind='observed' "
                 "order by reporter, partner, hs, month, flow, retrieved")
        latest = {}
        for r in c.execute(query):
            latest[(r['reporter'], r['partner'], r['hs'], r['month'], r['flow'])] = r['value_usd']
        count = c.execute("select count(*) from observations where kind='observed'").fetchone()[0]
    for (reporter, partner, hs, month, flow), value in latest.items():
        if hs not in ITEMS:
            continue
        if flow == 'M' and partner == 'CHN' and reporter in CANDIDATES:
            key = ('in', None, hs, reporter)
        elif flow == 'X' and partner in TARGETS and reporter in CANDIDATES and reporter != partner:
            key = ('out', partner, hs, reporter)
        else:
            continue
        if value > 0:
            rows[key][month] = math.log(value)
        else:
            zeros[key].add(month)
    return rows, zeros, count, len(latest)


def interpolate(series):
    months = sorted(series)
    first, last = months[0], months[-1]
    xs = np.arange(first, last + 1)
    known = np.array(months); vals = np.array([series[m] for m in months])
    return first, np.interp(xs, known, vals)


def rolling_median(values, window=12):
    half = window // 2
    out = np.empty_like(values)
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def first_harmonic(profile):
    m = np.arange(12)
    A = np.column_stack([np.sin(m * np.pi / 6), np.cos(m * np.pi / 6)])
    coef, *_ = np.linalg.lstsq(A, profile, rcond=None)
    amp = float(math.hypot(coef[0], coef[1]))
    phase = float(math.atan2(coef[1], coef[0]))  # amp*sin(m*pi/6 + phase)
    return amp, phase


def describe(series):
    """Anchor path and structure statistics for one observed series."""
    first, values = interpolate(series)
    n = len(series)
    out = dict(n_months=n, first_month=first, last_month=first + len(values) - 1, coverage=round(n / len(values), 3),
               level_mean=float(np.mean(values)))
    path = rolling_median(values)
    out['anchor'] = [round(float(v), 4) for v in path]
    if len(values) >= MIN_STRUCTURE_MONTHS:
        trend = rolling_median(values, 12)
        deviation = values - trend
        calendar = (np.arange(first, first + len(values)) % 12)
        profile = np.array([deviation[calendar == k].mean() if (calendar == k).any() else 0.0 for k in range(12)])
        profile -= profile.mean()
        resid = deviation - profile[calendar]
        amp, phase = first_harmonic(profile)
        ar1 = float(np.corrcoef(resid[1:], resid[:-1])[0, 1]) if np.std(resid) > 1e-9 else 0.0
        slope = float(np.polyfit(np.arange(len(trend)), trend, 1)[0])
        out.update(seasonal_amp=round(amp, 4), seasonal_phase=round(phase, 4), ar1=round(max(-0.95, min(0.95, ar1)), 4),
                   resid_sd=round(float(np.std(resid)), 4), drift=round(slope, 6))
    return out


def circular_mean(phases):
    return float(math.atan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))))


def pool(descriptions):
    keys = ['level_mean', 'seasonal_amp', 'ar1', 'resid_sd', 'drift']
    values = {k: [d[k] for d in descriptions if k in d] for k in keys}
    result = {'n_series': len(descriptions), 'n_structured': sum('ar1' in d for d in descriptions)}
    for k, v in values.items():
        if v:
            result[k] = round(float(np.median(v)), 6)
    if len(values['level_mean']) >= 2:
        result['level_sd'] = round(float(np.std(values['level_mean'])), 4)
    phases = [d['seasonal_phase'] for d in descriptions if 'seasonal_phase' in d]
    if phases:
        result['seasonal_phase'] = round(circular_mean(phases), 4)
    return result


def event_responses(series, events):
    curves = {'in': [], 'out': []}
    horizon = 18
    for e in events:
        eff = mi(e['effective'])
        for (leg, target, hs, cand), s in series.items():
            if leg == 'out' and target != e['target']:
                continue
            if leg == 'in':
                continue  # inbound response to a target's event is attributed through the out leg only
            first, values = interpolate(s)
            last = first + len(values) - 1
            if first > eff - 12 or last < eff + 6:
                continue
            pre = values[eff - 12 - first: eff - first].mean()
            post = [values[eff + k - first] - pre if eff + k <= last else np.nan for k in range(1, horizon + 1)]
            curves[leg].append(post)
    result = {}
    for leg, rows in curves.items():
        if not rows:
            result[leg] = {'pairs': 0}
            continue
        arr = np.array(rows, dtype=float)
        median = np.nanmedian(arr, axis=0)
        peak = int(np.nanargmax(median)) + 1
        result[leg] = {'pairs': len(rows), 'median_log_response': [round(float(v), 4) if np.isfinite(v) else None for v in median],
                       'amplitude': round(float(max(0.0, np.nanmax(median))), 4), 'peak_lag_months': peak,
                       'iqr_at_peak': [round(float(np.nanpercentile(arr[:, peak - 1], q)), 4) for q in (25, 75)]}
    return result


def missingness(series, zeros):
    by_target = defaultdict(list); by_candidate = defaultdict(list)
    for (leg, target, hs, cand), s in series.items():
        months = sorted(s)
        span = months[-1] - months[0] + 1
        if span < MIN_STRUCTURE_MONTHS:
            continue
        reported_zero = len(zeros.get((leg, target, hs, cand), set()) & set(range(months[0], months[-1] + 1)))
        missing = 1 - (len(months) + reported_zero) / span
        (by_target[target] if leg == 'out' else by_candidate[cand]).append(missing)
    return {'out_by_target': {t: round(float(np.median(v)), 4) for t, v in by_target.items()},
            'in_by_candidate': {c: round(float(np.median(v)), 4) for c, v in by_candidate.items()},
            'out_pooled': round(float(np.median([x for v in by_target.values() for x in v])), 4) if by_target else None,
            'in_pooled': round(float(np.median([x for v in by_candidate.values() for x in v])), 4) if by_candidate else None}


def mirror_gap():
    """Log gap between the candidate's export report and the target's import report of the same flow."""
    gaps = []
    with closing(sqlite3.connect(SOURCE.as_uri() + '?mode=ro', uri=True)) as c:
        latest = {}
        for reporter, partner, hs, month, flow, value, retrieved in c.execute(
                "select reporter, partner, hs, month, flow, value_usd, retrieved from observations where kind='observed' and value_usd>0 order by retrieved"):
            latest[(reporter, partner, hs, month, flow)] = value
    for (reporter, partner, hs, month, flow), value in latest.items():
        if flow == 'X' and partner in TARGETS and reporter in CANDIDATES:
            mirror = latest.get((partner, reporter, hs, month, 'M'))
            if mirror:
                gaps.append(math.log(mirror) - math.log(value))
    if len(gaps) < 10:
        return {'pairs': len(gaps)}
    gaps = np.array(gaps)
    return {'pairs': len(gaps), 'log_sd': round(float(np.std(gaps)), 4), 'log_median': round(float(np.median(gaps)), 4)}


def growth_episodes(series):
    """Inbound 3-month growth episodes and how the outbound leg moved within 1-4 months (coupled / domestic / substitution)."""
    stats = dict(routes=0, months=0, episodes=0, coupled=0, domestic=0, substitution=0, durations=[], amplitudes=[], lags=[], gaps=[])
    for (leg, target, hs, cand), s in series.items():
        if leg != 'out':
            continue
        inbound = series.get(('in', None, hs, cand))
        if not inbound:
            continue
        f_out, v_out = interpolate(s); f_in, v_in = interpolate(inbound)
        first, last = max(f_out, f_in), min(f_out + len(v_out), f_in + len(v_in)) - 1
        if last - first + 1 < 36:
            continue
        x_in = v_in[first - f_in: last - f_in + 1]; x_out = v_out[first - f_out: last - f_out + 1]
        def growth(x):
            g = np.full(len(x), np.nan)
            for t in range(6, len(x)):
                g[t] = np.exp(np.mean(x[t - 2:t + 1]) - np.mean(x[t - 5:t - 2])) - 1
            return g
        g_in, g_out = growth(x_in), growth(x_out)
        stats['routes'] += 1; stats['months'] += len(x_in)
        t = 6; previous_start = None
        while t < len(x_in):
            if g_in[t] > 0.25:
                start = t
                while t < len(x_in) and g_in[t] > 0.10:
                    t += 1
                duration = t - start
                window = g_out[start + 1: min(len(x_out), start + 5)]
                window = window[np.isfinite(window)]
                best = float(np.max(window)) if len(window) else 0.0
                kind = 'coupled' if best > 0.20 else 'substitution' if (len(window) and float(np.min(window)) < -0.15) else 'domestic'
                stats['episodes'] += 1; stats[kind] += 1; stats['durations'].append(duration)
                stats['amplitudes'].append(float(np.nanmax(g_in[start:t])))
                if best > 0.20:
                    stats['lags'].append(int(np.argmax(window)) + 1)
                if previous_start is not None:
                    stats['gaps'].append(start - previous_start)
                previous_start = start
            else:
                t += 1
    n = stats['episodes']
    if n < 10:
        return {'episodes': n, 'routes': stats['routes']}
    return {'routes': stats['routes'], 'months': stats['months'], 'episodes': n, 'episodes_per_100_months': round(100 * n / stats['months'], 3),
            'kind_probabilities': [round(stats[k] / n, 3) for k in ('coupled', 'domestic', 'substitution')],
            'duration_median': int(np.median(stats['durations'])), 'duration_iqr': [int(np.percentile(stats['durations'], q)) for q in (25, 75)],
            'amplitude_median': round(float(np.median(stats['amplitudes'])), 3), 'amplitude_iqr': [round(float(np.percentile(stats['amplitudes'], q)), 3) for q in (25, 75)],
            'lag_median': int(np.median(stats['lags'])) if stats['lags'] else None,
            'gap_median': int(np.median(stats['gaps'])) if stats['gaps'] else None}


def main():
    series, zeros, observed_rows, dedup_rows = load_series()
    events = json.loads((ROOT / 'data/events.json').read_text(encoding='utf-8'))
    cohort = 'none registered (evaluation-readiness independent_cohort_registered=false); nothing excluded'
    if READINESS.exists():
        readiness = json.loads(READINESS.read_text(encoding='utf-8'))
        if readiness.get('independent_cohort_registered'):
            raise SystemExit('A registered cohort exists; add its exclusion before estimating')
    anchors = {}
    descriptions = defaultdict(list)
    for (leg, target, hs, cand), s in series.items():
        if len(s) < MIN_ANCHOR_MONTHS:
            continue
        d = describe(s)
        key = f'{leg}|{hs}|{cand}' if leg == 'in' else f'{leg}|{target}|{hs}|{cand}'
        anchors[key] = d
        descriptions[(leg, hs)].append(d)
    pooled = {}
    for hs in ITEMS:
        entry = {}
        for leg in ('in', 'out'):
            if descriptions.get((leg, hs)):
                entry[leg] = pool(descriptions[(leg, hs)])
        ratios = []
        for key, d in anchors.items():
            if key.startswith(f'out|') and key.split('|')[2] == hs:
                cand = key.split('|')[3]
                partner = anchors.get(f'in|{hs}|{cand}')
                if partner:
                    ratios.append(d['level_mean'] - partner['level_mean'])
        if ratios:
            entry['out_in_log_ratio'] = round(float(np.median(ratios)), 4); entry['ratio_pairs'] = len(ratios)
        if entry:
            pooled[hs] = entry
    fallback = {leg: pool([d for (l, hs), ds in descriptions.items() if l == leg for d in ds]) for leg in ('in', 'out')}
    all_ratios = [e['out_in_log_ratio'] for e in pooled.values() if 'out_in_log_ratio' in e]
    fallback['out_in_log_ratio'] = round(float(np.median(all_ratios)), 4) if all_ratios else None
    result = dict(
        schema_version='btl-v3-panel-parameters-v1',
        estimated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source=dict(path='data/integration.sqlite', sha256=sha(SOURCE), observed_rows=observed_rows, deduplicated_cells=dedup_rows,
                    series_used=len(series), anchors=len(anchors)),
        evaluation_cohort_exclusion=cohort,
        legs=dict(**{'in': 'candidate imports from China (reporter=candidate, partner=CHN, flow M); calibration source, not the map supplier label',
                     'out': 'candidate exports to the target (reporter=candidate, partner=target, flow X)'}),
        thresholds=dict(min_anchor_months=MIN_ANCHOR_MONTHS, min_structure_months=MIN_STRUCTURE_MONTHS, anchor='12-month centered rolling median of log USD over the observed span'),
        anchors=anchors, pooled_by_hs=pooled, fallback=fallback,
        events=event_responses(series, events), missingness=missingness(series, zeros), mirror=mirror_gap(), growth_episodes=growth_episodes(series),
        legacy_defaults={k: v for k, v in LEGACY.items()},
    )
    unestimated = ['revision_noise (kept 3.5% first-release factor)', 'inbound event response (attributed through the out leg only)']
    if result['mirror'].get('pairs', 0) < 10: unestimated.append('mirror_log_sd')
    if result['growth_episodes'].get('episodes', 0) < 10: unestimated.append('growth_episodes')
    for leg in ('in', 'out'):
        if result['events'][leg].get('pairs', 0) < 5: unestimated.append(f'event_response_{leg}')
    missing_targets = [t for t in TARGETS if t not in result['missingness']['out_by_target']]
    if missing_targets: unestimated.append('missingness for targets ' + ','.join(missing_targets) + ' (pooled or legacy value)')
    for hs in ITEMS:
        if hs not in pooled: unestimated.append(f'all parameters for HS {hs} (fallback pooled across items)')
    result['unestimated'] = unestimated
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    summary = dict(anchors=len(anchors), anchored_routes_out=sum(k.startswith('out|') for k in anchors), anchored_candidates_in=sum(k.startswith('in|') for k in anchors),
                   hs_with_pooled=sorted(pooled), event_pairs=result['events']['out'].get('pairs'), mirror_pairs=result['mirror'].get('pairs'),
                   episodes=result['growth_episodes'].get('episodes'), unestimated=len(unestimated), output=str(OUTPUT.relative_to(ROOT)))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
