"""Replay of the panel generator (ml/build_db.py) in memory, for two diagnostics of benchmark v14.

  latents   with the panel's own seeds the replay reproduces data/test.db cell for cell (`verify`) and keeps what the
            database does not store: the seasonal, noise, growth-episode and event components of every route. The ceiling
            diagnostic (benchmark.ceiling) uses them to ask how well any model could rank if it knew a route's state exactly.
  replicas  with another seed tag the same generator draws an independent panel of the same routes (`ReplicaPanel`), used to
            measure how much more training data of the same kind would help. A replica shares no random draw with the panel.

The loop below mirrors build_db.build() statement for statement so the random streams stay aligned; `verify` fails if the
generator and this replay ever drift apart. Nothing here writes to the database.
"""
import json
import sqlite3

import numpy as np

import build_db as B
from common import DB, ROOT, TARGETS, ITEMS_ALL, candidates_for, mi, seed


def replay(tag='', end=None, latents=False):
    """Generate the panel in memory. tag='' uses the panel's seeds; any other tag gives an independent replica.

    Returns (rows, latent): rows[key] is the list of trade rows in Panel.rows layout; latent[key] holds the components
    (only when latents=True).
    """
    P = B.load_parameters(); DEMO = B.DEMO
    events = json.loads((ROOT / 'data/events.json').read_text(encoding='utf-8'))
    end = mi(B.PANEL_END) if end is None else end
    months = np.arange(end + 1); t_ref = mi(DEMO['level_reference_month'])
    episodes = P['growth_episodes']; ev_out = P['events']['out']; missing_p = P['missingness']; mirror_p = P['mirror']
    rows = {}; latent = {}
    for target in TARGETS:
        shared = np.random.default_rng(seed(target + tag)).normal(0, DEMO['shared_shock_sd'], end + 1)
        for hs in ITEMS_ALL:
            p_in, p_out = B.leg_parameters(P, hs, 'in'), B.leg_parameters(P, hs, 'out')
            ratio = P['pooled_by_hs'].get(hs, {}).get('out_in_log_ratio', P['fallback']['out_in_log_ratio'])
            for candidate in candidates_for(target):
                rng = np.random.default_rng(seed(target + hs + candidate + tag))
                a_in = P['anchors'].get(f'in|{hs}|{candidate}'); a_out = P['anchors'].get(f'out|{target}|{hs}|{candidate}')
                drift_in = min(p_in.get('drift', 0.0), DEMO['drift_cap']); drift_out = min(p_out.get('drift', 0.0), DEMO['drift_cap'])
                level_in = p_in['level_mean'] + rng.normal(0, min(p_in.get('level_sd', 0.6), DEMO['level_sd_cap']))
                path_in, kind_in = B.level_path(a_in, months, drift_in, level_in, t_ref)
                base_out = (float(np.mean(a_in['anchor'])) if a_in else level_in) + ratio + rng.normal(0, 0.5)
                path_out, kind_out = B.level_path(a_out, months, drift_out, base_out, t_ref)
                parts = {}

                def texture(p, anchor, leg):
                    amp = min((anchor or {}).get('seasonal_amp', p.get('seasonal_amp', 0.1)), DEMO['seasonal_amp_cap'])
                    phase = (anchor or {}).get('seasonal_phase', p.get('seasonal_phase', rng.uniform(0, 6)))
                    ar1 = float(np.clip((anchor or {}).get('ar1', p.get('ar1', 0.3)), 0.0, 0.9))
                    sd = min((anchor or {}).get('resid_sd', p.get('resid_sd', 0.1)), DEMO['resid_sd_cap'])
                    ar = np.zeros(end + 1)
                    for m in range(1, end + 1): ar[m] = ar1 * ar[m - 1] + rng.normal(0, sd)
                    parts[leg] = dict(seasonal=amp * np.sin(months * np.pi / 6 + phase), noise=ar, ar1=ar1)
                    return amp * np.sin(months * np.pi / 6 + phase) + ar
                incoming = np.exp(path_in + texture(p_in, a_in, 'in') + shared)
                outgoing = np.exp(path_out + texture(p_out, a_out, 'out') + 0.7 * shared)
                pulses = []
                if episodes.get('episodes', 0) < 10:
                    raise RuntimeError('replay covers the calibrated-episode branch of build_db only')
                kinds_p = episodes['kind_probabilities']; gap = episodes['gap_median'] or 7
                dur_lo, dur_hi = [max(2, d * DEMO['pulse_duration_scale']) for d in episodes['duration_iqr']]
                amp_lo, amp_hi = [min(DEMO['pulse_amp_cap'], a * DEMO['pulse_amp_scale']) for a in episodes['amplitude_iqr']]
                lag_med = episodes['lag_median'] or 3
                start = int(rng.integers(12, 30))
                while start < end:
                    duration = int(rng.integers(dur_lo, dur_hi + 1)); amp = rng.uniform(amp_lo, amp_hi); lag = int(max(1, lag_med + rng.integers(-1, 2)))
                    kind = rng.choice(['coupled', 'domestic', 'substitution'], p=kinds_p)
                    pulse = lambda l: np.maximum(0, 1 - abs((months - start - l) / duration))
                    out_amp = (amp * rng.uniform(.7, 1.4) if kind == 'coupled' else .03 if kind == 'domestic' else -.4)
                    incoming = incoming * (1 + amp * pulse(0))
                    outgoing = outgoing * (1 + out_amp * pulse(lag))
                    if latents:
                        pulses.append(dict(start=start, duration=duration, lag=lag, kind=str(kind), legs=(np.log(1 + amp * pulse(0)), np.log(1 + out_amp * pulse(lag)))))
                    start += int(rng.integers(gap * DEMO['pulse_gap_scale'][0], gap * DEMO['pulse_gap_scale'][1] + 1))
                responses = []
                for e in events:
                    if e['target'] != target: continue
                    eff = mi(e['effective'])
                    covered_out = a_out and a_out['first_month'] <= eff - 12 and a_out['last_month'] >= eff + 6
                    covered_in = a_in and a_in['first_month'] <= eff - 12 and a_in['last_month'] >= eff + 6
                    log_in = np.zeros(end + 1); log_out = np.zeros(end + 1)
                    if ev_out.get('pairs', 0) < 5:
                        raise RuntimeError('replay covers the calibrated-event branch of build_db only')
                    peak = ev_out['peak_lag_months']; amp_out = min(ev_out['amplitude'] * DEMO['event_damping'], DEMO['event_amp_cap'])
                    start = eff + int(rng.integers(1, 4)); ramp = np.clip((months - start) / peak, 0, 1) * np.exp(-np.maximum(0, months - start - 30) / 45)
                    if not covered_out:
                        log_out = rng.uniform(0.5, 1.0) * amp_out * ramp; outgoing = outgoing * np.exp(log_out)
                    if not covered_in:
                        start = eff + int(rng.integers(1, 10)); ramp = np.clip((months - start) / 8, 0, 1) * np.exp(-np.maximum(0, months - start - 30) / 45)
                        factor = 1 + rng.uniform(.05, .7) * ramp
                        log_in = np.log(factor); incoming = incoming * factor
                    responses.append(dict(published=mi(e['published']), effective=eff, legs=(log_in, log_out)))
                miss_out = missing_p['out_by_target'].get(target, missing_p.get('out_pooled'))
                if miss_out is None or target in ('PRK', 'CHN'): miss_out = .18 if target == 'PRK' else .05
                miss_in = missing_p['in_by_candidate'].get(candidate, missing_p.get('in_pooled'))
                if miss_in is None: miss_in = .05
                miss_out = float(np.clip(miss_out, DEMO['missing_floor'], DEMO['missing_cap'])); miss_in = float(np.clip(miss_in, DEMO['missing_floor'], DEMO['missing_cap']))
                mirror_sd = min(mirror_p.get('log_sd', .13), DEMO['mirror_sd_cap']); mirror_shift = mirror_p.get('log_median', 0.0)
                lag = int(rng.integers(1, 4)); mirror = outgoing * np.exp(rng.normal(mirror_shift, mirror_sd, end + 1))
                drops = np.column_stack([rng.random(end + 1) < miss_in, rng.random(end + 1) < miss_out])
                route = []
                for m in months:
                    for revision, delay in [(0, lag), (1, lag + 2)]:
                        factor = rng.normal(1, .035, 2) if revision == 0 else [1, 1]
                        vals = [None if drops[m, j] else round(float(v * factor[j]), 2) for j, v in enumerate([incoming[m], outgoing[m]])]
                        route.append(dict(target=target, hs=hs, candidate=candidate, month=int(m), revision=revision, available_month=int(m + delay),
                                          inbound=vals[0], outbound=vals[1], mirror=round(float(mirror[m]), 2), inbound_kind=str(kind_in[m]), outbound_kind=str(kind_out[m])))
                key = (target, hs, candidate)
                rows[key] = route
                if latents:
                    latent[key] = dict(path=(path_in, path_out), seasonal=(parts['in']['seasonal'], parts['out']['seasonal']), noise=(parts['in']['noise'], parts['out']['noise']),
                                       ar1=(parts['in']['ar1'], parts['out']['ar1']), pulses=pulses, events=responses, final=(incoming, outgoing), report_lag=lag)
    return rows, latent


def verify(rows, every=1):
    """Compare the replayed rows with data/test.db (every `every`-th month); returns the number of cells compared and raises
    on the first difference."""
    cells = 0
    with sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True) as c:
        for target, hs, candidate, month, revision, available, inbound, outbound, mirror in c.execute(
                'select target,hs,candidate,month,revision,available_month,inbound,outbound,mirror from trade where month % ? = 0', (every,)):
            r = rows[(target, hs, candidate)][2 * month + revision]
            if (r['month'], r['revision'], r['available_month'], r['inbound'], r['outbound'], r['mirror']) != (month, revision, available, inbound, outbound, mirror):
                raise AssertionError(f'replay differs from the database at {(target, hs, candidate, month, revision)}')
            cells += 3
    if every == 1 and cells != 3 * sum(len(v) for v in rows.values()):
        raise AssertionError('replay and database hold a different number of rows')
    return cells


class ReplicaPanel:
    """An independent draw of the panel (same routes, parameters and events; different random streams), with the Panel interface."""

    def __init__(self, tag, end=None):
        if not tag:
            raise ValueError('a replica needs a seed tag; the empty tag is the panel itself')
        from common import Panel
        self.rows, _ = replay(tag, end)
        self.end = mi(B.PANEL_END) if end is None else end
        self.events = json.loads((ROOT / 'data/events.json').read_text(encoding='utf-8'))
        self.history = Panel.history.__get__(self); self.features = Panel.features.__get__(self)
