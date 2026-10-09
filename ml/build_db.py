"""Build the demo trade panel (data/test.db) from generator parameters estimated on observed rows.

Every route (target, HS, candidate) has an inbound leg (candidate imports) and an outbound
leg (candidate exports to the target). Where data/panel-parameters.json holds an anchor
for a leg, the level path follows the observed series' 12-month rolling median; other legs
take pooled per-HS parameters. Seasonality, AR(1) texture, growth episodes, event responses,
missingness and mirror gaps are all calibrated, but several are damped by DEMO caps so the
demonstration curves stay smooth; the caps are recorded in the metadata table. Each cell
carries its kind (anchored / calibrated). The review queue survives a rebuild.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import btl_paths  # noqa: E402,F401

import sqlite3, json, datetime, hashlib, os
import numpy as np
from common import *

PARAMETERS = ROOT / 'data/panel-parameters.json'
PANEL_TAG = 'v3-calibrated'
DATASET_VERSION = 'btl-test-3-calibrated'
# Last month of the panel. It is fixed, not today's month: the random streams depend on the panel length, so a
# rolling end would give a fresh installation a different panel from the one the deployed model was fitted on.
# BTL_PANEL_END=YYYY-MM builds a longer panel on purpose (then retrain and redeploy, docs/ML.md).
PANEL_END = '2026-09'
# Demo damping: observed monthly trade is far more jagged than a demonstration screen should be.
DEMO = dict(resid_sd_cap=0.05, drift_cap=0.004, level_sd_cap=1.5, seasonal_amp_cap=0.35, event_amp_cap=1.0, event_damping=0.3,
            mirror_sd_cap=0.4, missing_floor=0.02, missing_cap=0.2, pulse_amp_scale=0.5, pulse_amp_cap=0.8, pulse_duration_scale=3,
            pulse_gap_scale=(2, 6), level_reference_month='2023-01', shared_shock_sd=0.025)


def load_parameters():
    return json.loads(PARAMETERS.read_text(encoding='utf-8'))


def leg_parameters(P, hs, leg):
    pooled = P['pooled_by_hs'].get(hs, {}).get(leg) or P['fallback'][leg]
    return pooled


def level_path(anchor, months, drift, level0, t_ref):
    """Log level per month: anchored inside the observed span, extrapolated with the capped drift outside; else a straight line."""
    if anchor:
        first, values = anchor['first_month'], np.array(anchor['anchor'])
        last = first + len(values) - 1
        path = np.empty(len(months))
        path[:] = np.nan
        inside = (months >= first) & (months <= last)
        path[inside] = values[months[inside] - first]
        before = months < first; after = months > last
        path[before] = values[0] - drift * (first - months[before])
        path[after] = values[-1] + drift * (months[after] - last)
        kinds = np.where(inside, 'anchored', 'calibrated')
        return path, kinds
    return level0 + drift * (months - t_ref), np.full(len(months), 'calibrated', dtype=object)


def build():
    P = load_parameters()
    parameters_sha = hashlib.sha256(PARAMETERS.read_bytes()).hexdigest()
    DB.parent.mkdir(exist_ok=True)
    # Rebuild in place: a running demo server may hold the file open, and the review queue must survive.
    c = sqlite3.connect(DB, timeout=60)
    c.executescript('''PRAGMA journal_mode=DELETE;
 DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS items; DROP TABLE IF EXISTS trade; DROP TABLE IF EXISTS metadata;
 CREATE TABLE events(id TEXT PRIMARY KEY,target TEXT,title TEXT,announced TEXT,effective TEXT,published TEXT,authority TEXT,url TEXT,summary TEXT,scope TEXT);
 CREATE TABLE items(hs TEXT PRIMARY KEY,name TEXT,classification TEXT);
 CREATE TABLE trade(target TEXT,hs TEXT,candidate TEXT,month INTEGER,revision INTEGER,available_month INTEGER,inbound REAL,outbound REAL,mirror REAL,inbound_kind TEXT,outbound_kind TEXT,PRIMARY KEY(target,hs,candidate,month,revision));
 CREATE INDEX trade_asof ON trade(target,hs,candidate,month,available_month);
 CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
 CREATE TABLE IF NOT EXISTS review_queue(id TEXT PRIMARY KEY,payload TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
 ''')
    queue_rows = c.execute('select id from review_queue').fetchall()
    events = json.loads((ROOT / 'data/events.json').read_text(encoding='utf-8'))
    c.executemany('insert into events values (:id,:target,:title,:announced,:effective,:published,:authority,:url,:summary,:scope)', events)
    c.executemany('insert into items values (?,?,?)', [(hs, name, 'HS 2017') for hs, name in zip(ITEMS_ALL, NAMES_ALL)])  # the panel always holds every item; the active set only decides what is served
    now = datetime.date.today(); end = mi(os.getenv('BTL_PANEL_END') or PANEL_END); months = np.arange(end + 1); allrows = []
    t_ref = mi(DEMO['level_reference_month'])
    episodes = P['growth_episodes']; ev_out = P['events']['out']; missing_p = P['missingness']; mirror_p = P['mirror']
    kind_counts = {'anchored': 0, 'calibrated': 0}
    anchored_routes = 0
    for target in TARGETS:
        shared = np.random.default_rng(seed(target)).normal(0, DEMO['shared_shock_sd'], end + 1)
        for hs in ITEMS_ALL:
            p_in, p_out = leg_parameters(P, hs, 'in'), leg_parameters(P, hs, 'out')
            ratio = P['pooled_by_hs'].get(hs, {}).get('out_in_log_ratio', P['fallback']['out_in_log_ratio'])
            for candidate in candidates_for(target):
                rng = np.random.default_rng(seed(target + hs + candidate))
                a_in = P['anchors'].get(f'in|{hs}|{candidate}'); a_out = P['anchors'].get(f'out|{target}|{hs}|{candidate}')
                drift_in = min(p_in.get('drift', 0.0), DEMO['drift_cap']); drift_out = min(p_out.get('drift', 0.0), DEMO['drift_cap'])
                level_in = p_in['level_mean'] + rng.normal(0, min(p_in.get('level_sd', 0.6), DEMO['level_sd_cap']))
                path_in, kind_in = level_path(a_in, months, drift_in, level_in, t_ref)
                base_out = (float(np.mean(a_in['anchor'])) if a_in else level_in) + ratio + rng.normal(0, 0.5)
                path_out, kind_out = level_path(a_out, months, drift_out, base_out, t_ref)
                if a_in or a_out: anchored_routes += 1
                def texture(p, anchor, leg):
                    amp = min((anchor or {}).get('seasonal_amp', p.get('seasonal_amp', 0.1)), DEMO['seasonal_amp_cap'])
                    phase = (anchor or {}).get('seasonal_phase', p.get('seasonal_phase', rng.uniform(0, 6)))
                    ar1 = float(np.clip((anchor or {}).get('ar1', p.get('ar1', 0.3)), 0.0, 0.9))
                    sd = min((anchor or {}).get('resid_sd', p.get('resid_sd', 0.1)), DEMO['resid_sd_cap'])
                    ar = np.zeros(end + 1)
                    for m in range(1, end + 1): ar[m] = ar1 * ar[m - 1] + rng.normal(0, sd)
                    return amp * np.sin(months * np.pi / 6 + phase) + ar
                incoming = np.exp(path_in + texture(p_in, a_in, 'in') + shared)
                outgoing = np.exp(path_out + texture(p_out, a_out, 'out') + 0.7 * shared)
                # Growth episodes: calibrated kind mix, lag and duration; amplitude damped for the demonstration.
                if episodes.get('episodes', 0) >= 10:
                    kinds_p = episodes['kind_probabilities']; gap = episodes['gap_median'] or 7
                    dur_lo, dur_hi = [max(2, d * DEMO['pulse_duration_scale']) for d in episodes['duration_iqr']]
                    amp_lo, amp_hi = [min(DEMO['pulse_amp_cap'], a * DEMO['pulse_amp_scale']) for a in episodes['amplitude_iqr']]
                    lag_med = episodes['lag_median'] or 3
                    start = int(rng.integers(12, 30))
                    while start < end:
                        duration = int(rng.integers(dur_lo, dur_hi + 1)); amp = rng.uniform(amp_lo, amp_hi); lag = int(max(1, lag_med + rng.integers(-1, 2)))
                        kind = rng.choice(['coupled', 'domestic', 'substitution'], p=kinds_p)
                        pulse = lambda l: np.maximum(0, 1 - abs((months - start - l) / duration))
                        incoming = incoming * (1 + amp * pulse(0))
                        outgoing = outgoing * (1 + (amp * rng.uniform(.7, 1.4) if kind == 'coupled' else .03 if kind == 'domestic' else -.4) * pulse(lag))
                        start += int(rng.integers(gap * DEMO['pulse_gap_scale'][0], gap * DEMO['pulse_gap_scale'][1] + 1))
                else:
                    for start in range(int(rng.integers(12, 30)), end, int(rng.integers(28, 55))):
                        duration = int(rng.integers(10, 24)); amp = rng.uniform(.25, 1.2); lag = int(rng.integers(1, 5)); kind = rng.choice(['coupled', 'domestic', 'substitution'], p=[.55, .3, .15])
                        pulse = lambda l: np.maximum(0, 1 - abs((months - start - l) / duration))
                        incoming = incoming * (1 + amp * pulse(0)); outgoing = outgoing * (1 + (amp * rng.uniform(.7, 1.4) if kind == 'coupled' else .03 if kind == 'domestic' else -.4) * pulse(lag))
                # Event responses: anchored outbound legs already contain the observed response inside their span.
                for e in events:
                    if e['target'] != target: continue
                    eff = mi(e['effective'])
                    covered_out = a_out and a_out['first_month'] <= eff - 12 and a_out['last_month'] >= eff + 6
                    covered_in = a_in and a_in['first_month'] <= eff - 12 and a_in['last_month'] >= eff + 6
                    if ev_out.get('pairs', 0) >= 5:
                        peak = ev_out['peak_lag_months']; amp_out = min(ev_out['amplitude'] * DEMO['event_damping'], DEMO['event_amp_cap'])
                        start = eff + int(rng.integers(1, 4)); ramp = np.clip((months - start) / peak, 0, 1) * np.exp(-np.maximum(0, months - start - 30) / 45)
                        if not covered_out: outgoing = outgoing * np.exp(rng.uniform(0.5, 1.0) * amp_out * ramp)
                    else:
                        start = eff + int(rng.integers(1, 10)); ramp = np.clip((months - start) / 8, 0, 1) * np.exp(-np.maximum(0, months - start - 30) / 45)
                        if not covered_out: outgoing = outgoing * (1 + rng.uniform(-.3, .9) * np.concatenate((np.zeros(2), ramp[:-2])))
                    if not covered_in:
                        start = eff + int(rng.integers(1, 10)); ramp = np.clip((months - start) / 8, 0, 1) * np.exp(-np.maximum(0, months - start - 30) / 45)
                        incoming = incoming * (1 + rng.uniform(.05, .7) * ramp)
                miss_out = missing_p['out_by_target'].get(target, missing_p.get('out_pooled'))
                if miss_out is None or target in ('PRK', 'CHN'): miss_out = .18 if target == 'PRK' else .05
                miss_in = missing_p['in_by_candidate'].get(candidate, missing_p.get('in_pooled'))
                if miss_in is None: miss_in = .05
                miss_out = float(np.clip(miss_out, DEMO['missing_floor'], DEMO['missing_cap'])); miss_in = float(np.clip(miss_in, DEMO['missing_floor'], DEMO['missing_cap']))
                mirror_sd = min(mirror_p.get('log_sd', .13), DEMO['mirror_sd_cap']); mirror_shift = mirror_p.get('log_median', 0.0)
                lag = int(rng.integers(1, 4)); mirror = outgoing * np.exp(rng.normal(mirror_shift, mirror_sd, end + 1))
                drops = np.column_stack([rng.random(end + 1) < miss_in, rng.random(end + 1) < miss_out])
                for m in months:
                    for revision, delay in [(0, lag), (1, lag + 2)]:
                        factor = rng.normal(1, .035, 2) if revision == 0 else [1, 1]
                        vals = [None if drops[m, j] else round(float(v * factor[j]), 2) for j, v in enumerate([incoming[m], outgoing[m]])]
                        allrows.append((target, hs, candidate, int(m), revision, int(m + delay), *vals, round(float(mirror[m]), 2), str(kind_in[m]), str(kind_out[m])))
                        if revision == 1:
                            kind_counts[str(kind_in[m])] += 1; kind_counts[str(kind_out[m])] += 1
    c.executemany('insert into trade values (?,?,?,?,?,?,?,?,?,?,?)', allrows)
    c.executemany('insert into metadata values (?,?)', [('panel_end', str(end)), ('built_at', now.isoformat()), ('seed', 'sha256-route-v2-calibrated'),
        ('dataset_version', DATASET_VERSION), ('model_tag', PANEL_TAG), ('parameters_sha256', parameters_sha), ('demo_caps', json.dumps(DEMO)),
        ('kind_counts', json.dumps(kind_counts)), ('anchored_routes', str(anchored_routes)), ('review_queue_preserved', str(len(queue_rows)))])
    c.commit()
    try:
        c.execute('VACUUM')
    except sqlite3.OperationalError:
        pass  # a reader holds the file; the rebuilt tables are complete without compaction
    c.close()
    print(json.dumps({'trade_rows': len(allrows), 'events': len(events), 'routes': sum(len(candidates_for(t)) * len(ITEMS_ALL) for t in TARGETS), 'panel_end': mt(end),
                      'dataset_version': DATASET_VERSION, 'anchored_routes': anchored_routes, 'cell_kinds': kind_counts, 'review_queue_preserved': len(queue_rows)}))


if __name__ == '__main__': build()
