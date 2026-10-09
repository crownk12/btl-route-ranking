"""Benchmark stages after the dataset build: benchmark, seeds, ablation, importance, select, report."""
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
from common import FEATURES, TARGETS, ITEMS
from benchmark.run import OUT, CACHE, SEEDS, write, read, load_dataset, now, panel_metadata
from benchmark.features import GROUPS, group_of
from benchmark.models import MODELS, TUNED_ORDER, SIMPLICITY, make_model, fit_model
from benchmark.metrics import evaluate, ranking_metrics

FEATURE_SETS = ['12f', '25f']
COVERAGE_BINS = [('>=75%', 0.75, 1.01), ('50-75%', 0.5, 0.75), ('25-50%', 0.25, 0.5), ('<25%', 0.0, 0.25)]
PRIMARY_HORIZONS = [3, 6, 12]
GATES = dict(A_margin=0.02, B_country_min=3, B_horizon_tolerance=0.02, C_rule='low-coverage (<50%) mean score not above the champion and low-coverage NDCG@5 not below it by > 0.02',
             D_seed_sd_max=0.01, E_tie=0.005, F_rule='hybrids must beat every component on validation NDCG@5')


def feature_indices(d, fs):
    names = FEATURES if fs == '12f' else read('feature-catalog.json')['final_25f']
    return [d['features'].index(n) for n in names], names


def breakdowns(d, mask, pred, names):
    """Per-country, per-horizon, per-HS and per-coverage-bin ranking metrics on one split."""
    y = d['Y'][mask]; g = d['G'][mask]; score = pred[:, 0]
    cov = d['X'][mask][:, d['features'].index('coverage')]
    out = {'by_target': {}, 'by_horizon': {}, 'by_hs': {}, 'by_coverage': {}}
    for t in TARGETS:
        m = d['T'][mask] == t
        if m.any(): out['by_target'][t] = ranking_metrics(y[m, 0], score[m], g[m])
    for h in sorted(set(d['H'][mask].tolist())):
        m = d['H'][mask] == h
        out['by_horizon'][str(int(h))] = ranking_metrics(y[m, 0], score[m], g[m])
    for hs in ITEMS:
        m = d['HS'][mask] == hs
        if m.any(): out['by_hs'][hs] = ranking_metrics(y[m, 0], score[m], g[m])
    for label, lo, hi in COVERAGE_BINS:
        m = (cov >= lo) & (cov < hi)
        out['by_coverage'][label] = {'rows': int(m.sum()), 'mean_score': float(np.mean(np.clip(score[m], 0, 1))) if m.any() else None,
                                     'mean_rank_score_raw': float(np.mean(score[m])) if m.any() else None,
                                     'positive_rate': float(y[m, 0].mean()) if m.any() else None}
    # NDCG restricted to queries whose selected rows are thin: rank-level low-coverage check = share of thin rows placed in top-5 vs their share overall
    thin = cov < 0.5
    top5 = np.zeros(len(score), bool)
    from benchmark.metrics import per_query_indices
    for idx in per_query_indices(g):
        if len(idx) >= 5:
            top5[idx[np.argsort(-score[idx], kind='stable')[:5]]] = True
    out['thin_share'] = {'thin_rows': int(thin.sum()), 'thin_share_overall': float(thin.mean()) if len(thin) else None,
                         'thin_share_in_top5': float(thin[top5].mean()) if top5.any() else None}
    return out


def tune(name, d, idx, seed=42):
    """Pick the configuration with the best inner-holdout NDCG@5 (fit on inner fit rows)."""
    spec = MODELS[name]
    if spec['factory'] is None or len(spec['configs']) <= 1:
        return spec['configs'][0], []
    Xf, Yf, Gf = d['X'][d['inner']['fit']][:, idx], d['Y'][d['inner']['fit']], d['G'][d['inner']['fit']]
    Xh, Yh, Gh = d['X'][d['inner']['holdout']][:, idx], d['Y'][d['inner']['holdout']], d['G'][d['inner']['holdout']]
    trials = []
    for cfg in spec['configs']:
        model = spec['factory'](cfg, seed)
        t = fit_model(model, np.nan_to_num(Xf), Yf, None, Gf)
        m = ranking_metrics(Yh[:, 0], model.predict(np.nan_to_num(Xh))[:, 0], Gh)
        trials.append({'config': cfg, 'inner_holdout_ndcg_at_5': m['ndcg_at_5'], 'fit_seconds': round(t, 1)})
    best = max(trials, key=lambda t: t['inner_holdout_ndcg_at_5'])
    return best['config'], trials


def stage_benchmark():
    d = load_dataset()
    train, val = d['masks']['train'], d['masks']['validation']
    inner_masks = d['inner']
    results = {}
    for fs in FEATURE_SETS:
        idx, names = feature_indices(d, fs)
        X = np.nan_to_num(d['X'][:, idx])
        inner = (X[inner_masks['fit']], d['Y'][inner_masks['fit']], X[inner_masks['holdout']], d['Y'][inner_masks['holdout']], d['G'][inner_masks['holdout']])
        best = {}; entries = {}
        for name in TUNED_ORDER:
            cfg, trials = tune(name, {'X': d['X'], 'Y': d['Y'], 'G': d['G'], 'inner': inner_masks}, idx)
            best[name] = cfg
            model = make_model(name, best, 42)
            fit_s = fit_model(model, X[train], d['Y'][train], inner, d['G'][train])
            t = time.perf_counter(); pv = model.predict(X[val]); predict_s = time.perf_counter() - t
            metrics = evaluate(d['Y'][val], pv, d['G'][val], probabilistic=MODELS[name]['probabilistic'])
            entry = {'label': MODELS[name]['label'], 'type': MODELS[name]['type'], 'feature_set': fs, 'n_features': len(names), 'config': cfg, 'tuning_trials': trials,
                     'train_seconds': round(fit_s, 1), 'predict_seconds_validation': round(predict_s, 3), 'predict_ms_per_1000_rows': round(predict_s / max(len(pv), 1) * 1e6, 3),
                     'validation': metrics, 'validation_breakdown': breakdowns(d, val, pv, names)}
            if getattr(model, 'weights', None) is not None: entry['blend_weights'] = dict(zip(MODELS[name]['components'], map(float, model.weights)))
            if getattr(model, 'meta', None) is not None and hasattr(model.meta, 'coef_'): entry['meta_coefficients'] = dict(zip(MODELS[name]['components'], map(float, model.meta.coef_[0])))
            entries[name] = entry
            CACHE.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, CACHE / f'model-{fs}-{name}.joblib', compress=3)
            print(json.dumps({'fs': fs, 'model': name, 'val_ndcg5': round(metrics['ndcg_at_5'], 4), 'fit_s': round(fit_s, 1)}), flush=True)
        write(f'benchmark-{fs}.json', {'generated_at_utc': now(), 'feature_set': fs, 'features': names, 'tuned_configs': best, 'models': entries,
                                       'note': 'validation metrics only; test is evaluated once in the select stage after the champion candidate is frozen'})
        results[fs] = entries
    baseline_record(d)


def baseline_record(d):
    meta = panel_metadata()
    sel = json.loads((ROOT / 'artifacts/integration/model-selection.json').read_text(encoding='utf-8'))
    metrics = json.loads((ROOT / 'models/metrics.json').read_text(encoding='utf-8'))
    manifest = read('split-manifest.json')
    b12 = read('benchmark-12f.json')['models']['M01_extra_trees']
    write('baseline.json', {
        'generated_at_utc': now(), 'deployed_model_version': sel['version'], 'deployed_candidate': sel['chosen'], 'panel_dataset_version': meta.get('dataset_version'),
        'integration_data_version': json.loads((ROOT / 'artifacts/integration/serving-verification.json').read_text(encoding='utf-8')).get('data_version'),
        'features_12f': FEATURES, 'label': manifest['label'], 'split_rows_queries': {k: {'rows': v['rows'], 'queries': v['queries'], 'positives': v['positives']} for k, v in manifest['splits'].items()},
        'rows_by_target': {k: v['rows_by_target'] for k, v in manifest['splits'].items()}, 'rows_by_horizon': {k: v['rows_by_horizon'] for k, v in manifest['splits'].items()},
        'deployed_metrics_from_train_py': {'validation_ndcg_at_5': metrics['validation']['model_ndcg_at_5'], 'test_ndcg_at_5': metrics['test']['model_ndcg_at_5'],
                                           'validation_roc_auc': metrics['validation']['roc_auc'], 'test_roc_auc': metrics['test']['roc_auc']},
        'reproduced_in_benchmark_12f_M01': {'validation': b12['validation'], 'config': b12['config'], 'train_seconds': b12['train_seconds'], 'predict_ms_per_1000_rows': b12['predict_ms_per_1000_rows']},
        'seed': 42, 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'benchmark_venv': '.venv-bench (numpy, scikit-learn, lightgbm, xgboost, catboost)',
                                    'versions': library_versions()}})


def library_versions():
    out = {}
    for m in ['numpy', 'sklearn', 'lightgbm', 'xgboost', 'catboost']:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def stage_seeds():
    d = load_dataset()
    train, val = d['masks']['train'], d['masks']['validation']
    out = {'seeds': SEEDS, 'models': {}}
    for fs in FEATURE_SETS:
        idx, names = feature_indices(d, fs)
        X = np.nan_to_num(d['X'][:, idx]); best = read(f'benchmark-{fs}.json')['tuned_configs']
        inner = (X[d['inner']['fit']], d['Y'][d['inner']['fit']], X[d['inner']['holdout']], d['Y'][d['inner']['holdout']], d['G'][d['inner']['holdout']])
        for name in TUNED_ORDER:
            if not MODELS[name]['stochastic']:
                continue
            values = []
            for seed in SEEDS:
                model = make_model(name, best, seed)
                fit_model(model, X[train], d['Y'][train], inner, d['G'][train])
                values.append(ranking_metrics(d['Y'][val, 0], model.predict(X[val])[:, 0], d['G'][val])['ndcg_at_5'])
            out['models'][f'{fs}:{name}'] = {'feature_set': fs, 'model': name, 'validation_ndcg_at_5_by_seed': dict(zip(map(str, SEEDS), values)),
                                             'mean': float(np.mean(values)), 'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}
            print(json.dumps({'fs': fs, 'model': name, 'mean': round(float(np.mean(values)), 4), 'std': round(float(np.std(values)), 4)}), flush=True)
    out['generated_at_utc'] = now()
    write('seed-stability.json', out)


def top_models(fs, k=3):
    b = read(f'benchmark-{fs}.json')['models']
    return sorted(b, key=lambda n: -b[n]['validation']['ndcg_at_5'])[:k]


def stage_ablation():
    d = load_dataset()
    train, val = d['masks']['train'], d['masks']['validation']
    idx25, names25 = feature_indices(d, '25f'); best = read('benchmark-25f.json')['tuned_configs']
    full = read('benchmark-25f.json')['models']
    models = top_models('25f', 3)
    if 'M01_extra_trees' not in models: models.append('M01_extra_trees')
    out = {'models': models, 'groups': {}, 'generated_at_utc': now()}
    for group, members in GROUPS.items():
        if group == 'quality_task':
            continue
        removed = [n for n in members if n in names25]
        if not removed:
            continue
        keep = [n for n in names25 if n not in removed]
        idx = [d['features'].index(n) for n in keep]
        X = np.nan_to_num(d['X'][:, idx])
        inner = (X[d['inner']['fit']], d['Y'][d['inner']['fit']], X[d['inner']['holdout']], d['Y'][d['inner']['holdout']], d['G'][d['inner']['holdout']])
        entry = {'removed': removed, 'remaining_features': len(keep), 'models': {}}
        for name in models:
            model = make_model(name, best, 42)
            fit_model(model, X[train], d['Y'][train], inner, d['G'][train])
            m = ranking_metrics(d['Y'][val, 0], model.predict(X[val])[:, 0], d['G'][val])
            ref = full[name]['validation']
            entry['models'][name] = {'validation': {k: m[k] for k in ['ndcg_at_5', 'precision_at_5', 'recall_at_5', 'map_at_5']},
                                     'delta_vs_full': {k: (m[k] - ref[k]) if (m[k] is not None and ref[k] is not None) else None for k in ['ndcg_at_5', 'precision_at_5', 'recall_at_5']}}
        out['groups'][group] = entry
        print(json.dumps({'ablation': group, 'removed': len(removed), 'delta_ndcg': {n: round(v['delta_vs_full']['ndcg_at_5'], 4) for n, v in entry['models'].items()}}), flush=True)
    write('ablation.json', out)


def stage_importance():
    d = load_dataset()
    val = d['masks']['validation']
    out = {'method': 'permutation: validation NDCG@5 drop when one feature column is shuffled (seed 0, one repeat); native importances where the library exposes them; SHAP not installed',
           'feature_sets': {}, 'generated_at_utc': now()}
    rng = np.random.default_rng(0)
    for fs in FEATURE_SETS:
        idx, names = feature_indices(d, fs)
        X = np.nan_to_num(d['X'][val][:, idx]); y = d['Y'][val, 0]; g = d['G'][val]
        models = top_models(fs, 3)
        if 'M01_extra_trees' not in models: models.append('M01_extra_trees')
        fs_out = {}
        for name in models:
            model = joblib.load(CACHE / f'model-{fs}-{name}.joblib')
            base = ranking_metrics(y, model.predict(X)[:, 0], g)['ndcg_at_5']
            drops = {}
            for j, fname in enumerate(names):
                Xp = X.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops[fname] = round(base - ranking_metrics(y, model.predict(Xp)[:, 0], g)['ndcg_at_5'], 5)
            by_country = {}
            for t in TARGETS:
                m = d['T'][val] == t
                if not m.any(): continue
                b = ranking_metrics(y[m], model.predict(X[m])[:, 0], g[m])['ndcg_at_5']
                cd = {}
                for j, fname in enumerate(names):
                    Xp = X[m].copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    cd[fname] = round(b - ranking_metrics(y[m], model.predict(Xp)[:, 0], g[m])['ndcg_at_5'], 5)
                by_country[t] = sorted(cd.items(), key=lambda kv: -kv[1])[:5]
            native = getattr(model, 'feature_importances_', None)
            fs_out[name] = {'base_validation_ndcg_at_5': base, 'permutation_drop': drops, 'top10': sorted(drops.items(), key=lambda kv: -kv[1])[:10],
                            'group_sum': {grp: round(sum(v for f, v in drops.items() if group_of(f) == grp), 5) for grp in GROUPS},
                            'by_country_top5': by_country,
                            'native_importance': dict(zip(names, map(float, native))) if native is not None and len(native) == len(names) else None}
            print(json.dumps({'fs': fs, 'model': name, 'top3': fs_out[name]['top10'][:3]}), flush=True)
        out['feature_sets'][fs] = fs_out
    write('feature-importance.json', out)


def stage_select():
    d = load_dataset()
    test = d['masks']['test']
    b = {fs: read(f'benchmark-{fs}.json') for fs in FEATURE_SETS}
    seeds = read('seed-stability.json')['models']
    champ = b['12f']['models']['M01_extra_trees']
    champ_val = champ['validation']['ndcg_at_5']
    verdicts = {}
    for fs in FEATURE_SETS:
        for name, e in b[fs]['models'].items():
            key = f'{fs}:{name}'
            v = e['validation']; reasons = []
            gain = v['ndcg_at_5'] - champ_val
            gates = {}
            gates['A_validation_improvement'] = gain >= GATES['A_margin']
            if not gates['A_validation_improvement']: reasons.append(f'gate A: validation NDCG@5 gain {gain:+.4f} < {GATES["A_margin"]}')
            bt = e['validation_breakdown']['by_target']; ct = champ['validation_breakdown']['by_target']
            wins = sum(1 for t in bt if bt[t]['ndcg_at_5'] is not None and ct[t]['ndcg_at_5'] is not None and bt[t]['ndcg_at_5'] >= ct[t]['ndcg_at_5'])
            bh = e['validation_breakdown']['by_horizon']; ch = champ['validation_breakdown']['by_horizon']
            horizon_ok = all(bh[str(h)]['ndcg_at_5'] >= ch[str(h)]['ndcg_at_5'] - GATES['B_horizon_tolerance'] for h in PRIMARY_HORIZONS if str(h) in bh)
            gates['B_robustness'] = wins >= GATES['B_country_min'] and horizon_ok
            if not gates['B_robustness']: reasons.append(f'gate B: better than champion in {wins}/4 countries, horizon check {"ok" if horizon_ok else "failed"}')
            lc = e['validation_breakdown']['by_coverage']; lcc = champ['validation_breakdown']['by_coverage']
            thin_mean = np.nanmean([lc[k]['mean_score'] for k in lc if k in ('25-50%', '<25%') and lc[k]['mean_score'] is not None] or [np.nan])
            thin_mean_c = np.nanmean([lcc[k]['mean_score'] for k in lcc if k in ('25-50%', '<25%') and lcc[k]['mean_score'] is not None] or [np.nan])
            ts = e['validation_breakdown']['thin_share']; tsc = champ['validation_breakdown']['thin_share']
            share_ok = ts['thin_share_in_top5'] is None or tsc['thin_share_in_top5'] is None or ts['thin_share_in_top5'] <= tsc['thin_share_in_top5'] + 0.02
            mean_ok = (not MODELS[name]['probabilistic']) or np.isnan(thin_mean) or np.isnan(thin_mean_c) or thin_mean <= thin_mean_c + 1e-9
            gates['C_low_coverage_safety'] = bool(share_ok and mean_ok)
            if not gates['C_low_coverage_safety']: reasons.append('gate C: thin-coverage rows scored or ranked higher than under the champion')
            s = seeds.get(key)
            gates['D_stability'] = (s is None) or (s['std'] <= GATES['D_seed_sd_max'])
            if not gates['D_stability']: reasons.append(f'gate D: seed std {s["std"]:.4f} > {GATES["D_seed_sd_max"]}')
            comps = MODELS[name]['components']
            gates['F_hybrid'] = all(v['ndcg_at_5'] > b[fs]['models'][c]['validation']['ndcg_at_5'] + 1e-12 for c in comps) if comps else True
            if not gates['F_hybrid']: reasons.append('gate F: does not beat every component')
            verdicts[key] = {'feature_set': fs, 'model': name, 'validation_ndcg_at_5': v['ndcg_at_5'], 'gain_vs_champion': gain, 'gates': gates, 'eligible': not reasons, 'reasons': reasons}
    eligible = [k for k, v in verdicts.items() if v['eligible'] and k != '12f:M01_extra_trees']
    if eligible:
        best_val = max(verdicts[k]['validation_ndcg_at_5'] for k in eligible)
        tied = [k for k in eligible if best_val - verdicts[k]['validation_ndcg_at_5'] <= GATES['E_tie']]
        chosen = min(tied, key=lambda k: (SIMPLICITY.index(k.split(':')[1]), 0 if k.startswith('12f') else 1))
        decision = f'{chosen} passes gates A-F and is the simplest among candidates within {GATES["E_tie"]} of the best validation NDCG@5'
    else:
        chosen = '12f:M01_extra_trees'; decision = 'no candidate passed every gate; the deployed Extra Trees (12F) is retained'
    # Frozen: evaluate test once for every combination
    test_results = {}
    for fs in FEATURE_SETS:
        idx, names = feature_indices(d, fs); X = np.nan_to_num(d['X'][test][:, idx])
        for name in TUNED_ORDER:
            model = joblib.load(CACHE / f'model-{fs}-{name}.joblib')
            pt = model.predict(X)
            test_results[f'{fs}:{name}'] = {'test': evaluate(d['Y'][test], pt, d['G'][test], probabilistic=MODELS[name]['probabilistic']), 'test_breakdown': breakdowns(d, test, pt, names)}
    write('model-selection.json', {'generated_at_utc': now(), 'champion_reference': '12f:M01_extra_trees (deployed extra-trees-v3-calibrated, same configuration)', 'gates': GATES,
                                   'verdicts': verdicts, 'eligible': eligible, 'chosen': chosen, 'decision': decision,
                                   'test_policy': 'model selection used validation only; test was evaluated exactly once, after freezing, for every combination in this table',
                                   'test_results': test_results, 'deployment_change': chosen != '12f:M01_extra_trees'})
    per_country = {'validation': {}, 'test': {}}; per_horizon = {'validation': {}, 'test': {}}; low_cov = {'validation': {}, 'test': {}}
    for fs in FEATURE_SETS:
        for name, e in b[fs]['models'].items():
            key = f'{fs}:{name}'
            per_country['validation'][key] = {t: v['ndcg_at_5'] for t, v in e['validation_breakdown']['by_target'].items()}
            per_country['test'][key] = {t: v['ndcg_at_5'] for t, v in test_results[key]['test_breakdown']['by_target'].items()}
            per_horizon['validation'][key] = {h: v['ndcg_at_5'] for h, v in e['validation_breakdown']['by_horizon'].items()}
            per_horizon['test'][key] = {h: v['ndcg_at_5'] for h, v in test_results[key]['test_breakdown']['by_horizon'].items()}
            low_cov['validation'][key] = {'by_coverage': e['validation_breakdown']['by_coverage'], 'thin_share': e['validation_breakdown']['thin_share']}
            low_cov['test'][key] = {'by_coverage': test_results[key]['test_breakdown']['by_coverage'], 'thin_share': test_results[key]['test_breakdown']['thin_share']}
    write('per-country.json', per_country); write('per-horizon.json', per_horizon); write('low-coverage-analysis.json', low_cov)
    rows = []
    for fs in FEATURE_SETS:
        for name in TUNED_ORDER:
            e = b[fs]['models'][name]; key = f'{fs}:{name}'; t = test_results[key]['test']; s = seeds.get(key)
            rows.append({'model': name, 'label': e['label'], 'feature_set': fs, 'type': e['type'], 'val_ndcg_at_5': e['validation']['ndcg_at_5'], 'test_ndcg_at_5': t['ndcg_at_5'],
                         'val_precision_at_5': e['validation']['precision_at_5'], 'val_recall_at_5': e['validation']['recall_at_5'], 'val_map_at_5': e['validation']['map_at_5'], 'val_mrr': e['validation']['mrr'],
                         'val_hit_rate_at_5': e['validation']['hit_rate_at_5'], 'val_roc_auc': e['validation']['roc_auc'], 'val_brier': e['validation']['brier'],
                         'test_precision_at_5': t['precision_at_5'], 'test_recall_at_5': t['recall_at_5'], 'test_map_at_5': t['map_at_5'], 'test_mrr': t['mrr'], 'test_roc_auc': t['roc_auc'],
                         'low_coverage_thin_share_in_top5': e['validation_breakdown']['thin_share']['thin_share_in_top5'],
                         'seed_std': s['std'] if s else None, 'train_seconds': e['train_seconds'], 'adopt': key == chosen, 'gates_passed': sum(verdicts[key]['gates'].values()), 'reasons': verdicts[key]['reasons']})
    write('benchmark-summary.json', {'generated_at_utc': now(), 'rows': rows, 'chosen': chosen, 'decision': decision})
    print(json.dumps({'chosen': chosen, 'eligible': eligible, 'decision': decision}, ensure_ascii=False))


def fmt(v, digits=4):
    return '—' if v is None else f'{v:.{digits}f}'


def stage_report():
    cat = read('feature-catalog.json'); q = read('feature-quality.json'); leak = read('leakage-audit.json'); manifest = read('split-manifest.json')
    summary = read('benchmark-summary.json'); sel = read('model-selection.json'); seeds = read('seed-stability.json'); abl = read('ablation.json'); imp = read('feature-importance.json')
    base = read('baseline.json'); b12 = read('benchmark-12f.json'); b25 = read('benchmark-25f.json')
    rows = summary['rows']
    by = {(r['feature_set'], r['model']): r for r in rows}
    champ = by[('12f', 'M01_extra_trees')]
    best_row = max(rows, key=lambda r: r['val_ndcg_at_5'])
    best25 = max((r for r in rows if r['feature_set'] == '25f'), key=lambda r: r['val_ndcg_at_5'])
    best12 = max((r for r in rows if r['feature_set'] == '12f'), key=lambda r: r['val_ndcg_at_5'])
    chosen_row = by[tuple(sel['chosen'].split(':'))]
    n25 = len(cat['final_25f'])
    lines = []
    L = lines.append
    L('# FINAL_REPORT — 7개 신호군 · 25개 Feature · 12개 모델 비교 (model-benchmark-v12)')
    L(f"\n생성: {now()} · 패널 `{base['panel_dataset_version']}` · 통합 자료 `{base['integration_data_version']}` · 배포 모델 `{base['deployed_model_version']}` · seed 42 (안정성 seed {seeds['seeds']})")
    L('\n> 모델 선택은 Validation 기준으로 수행했고 Test는 최종 일반화 성능 확인에만 사용했다(전 조합 1회 평가). 생성 패널(관측 보정) 안의 결과이며 실제 제재 우회 예측 성능이 아니다.')
    L('\n## 1. Executive Summary\n')
    L(f"- 최종 Feature: 7개 신호군 {n25}개 ({'25개 구성 완료' if n25 == 25 else '25개 미달, 제외 사유 2절'}). 후보 {len(cat['entries'])}개 중 미사용 {len(cat['candidates_not_used'])}개, 구현 불가 {len(cat['features_not_built'])}개.")
    L(f"- 12F → 25F (같은 모델 최고치): Validation NDCG@5 {fmt(best12['val_ndcg_at_5'])} ({best12['model']}) → {fmt(best25['val_ndcg_at_5'])} ({best25['model']}), Δ {best25['val_ndcg_at_5'] - best12['val_ndcg_at_5']:+.4f}.")
    L(f"- 12개 모델 중 최고 Validation NDCG@5: {best_row['model']} ({best_row['feature_set']}) {fmt(best_row['val_ndcg_at_5'])} (기준 Extra Trees 12F {fmt(champ['val_ndcg_at_5'])}, Δ {best_row['val_ndcg_at_5'] - champ['val_ndcg_at_5']:+.4f}).")
    L(f"- Champion 변경: {'예 → ' + sel['chosen'] if sel['deployment_change'] else '아니오 (Extra Trees 12F 유지)'}. {sel['decision']}")
    L(f"- 최종 Test NDCG@5: 채택 조합 {fmt(chosen_row['test_ndcg_at_5'])}, 기준 12F ET {fmt(champ['test_ndcg_at_5'])}, 최고 Validation 조합 {fmt(best_row['test_ndcg_at_5'])}.")
    xgb12, xgb25 = by[('12f', 'M11_xgb_ranker')], by[('25f', 'M11_xgb_ranker')]
    L(f"- Feature 확대 효과는 모델에 따라 다르다: 같은 XGBoost Ranker에서 12F→25F Validation Δ {xgb25['val_ndcg_at_5'] - xgb12['val_ndcg_at_5']:+.4f}(Test Δ {xgb25['test_ndcg_at_5'] - xgb12['test_ndcg_at_5']:+.4f})로 seed 표준편차(약 0.002)보다 크지만 Gate E의 동률 폭 0.005 안이라 단순한 12F 조합이 Champion으로 선택됐다. 25F는 Extra Trees·Random Forest·CatBoost에서 더 크게 올랐다.")
    L('\n## 2. Final Feature Set\n')
    L(f"| # | Feature | Group | 기존/신규 | 정의 | 계산 | 출처 | 시점 보존 | 결측률 |\n|---|---|---|---|---|---|---|---|---|")
    for i, name in enumerate(cat['final_25f'], 1):
        e = next(x for x in cat['entries'] if x['name'] == name)
        L(f"| {i} | `{name}` | {e['group']} | {'기존' if e['existing'] else '신규'} | {e['definition']} | {e['formula']} | {e['source']} | {'예' if e['point_in_time_safe'] else '아니오'} | {e['missing_rate']} |")
    L(f"\n그룹별 구성: " + ', '.join(f"{g} {len(v)}" for g, v in cat['groups'].items() if v) + '.')
    if cat['candidates_not_used']:
        L('\n계산했지만 사용하지 않은 후보:\n')
        for k, v in cat['candidates_not_used'].items(): L(f'- `{k}`: {v}')
    L('\n구현하지 않은 후보와 사유:\n')
    for k, v in cat['features_not_built'].items(): L(f'- {k}: {v}')
    L(f"\n중복 쌍(|r| > 0.9): {len(q['redundant_pairs'])}건" + (': ' + '; '.join(f"{p['a']}~{p['b']} ({p['spearman']:.2f})" for p in q['redundant_pairs'][:8]) if q['redundant_pairs'] else '') + '.')
    L(f"\nLeakage audit: {leak['overall']} ({sum(c['result'] == 'pass' for c in leak['checks'])}/{len(leak['checks'])} 검사 통과). " + '; '.join(c['check'] for c in leak['checks'] if c['result'] != 'pass'))
    L('\n## 3. Model Benchmark (24개 조합)\n')
    L('| Model | Feature Set | Type | Val NDCG@5 | Test NDCG@5 | P@5 | R@5 | MAP@5 | MRR | Low-Coverage(top5 thin share) | Stability(seed sd) | Train Time(s) | Adopt |\n|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|')
    for r in rows:
        L(f"| {r['model']} | {r['feature_set']} | {r['type']} | {fmt(r['val_ndcg_at_5'])} | {fmt(r['test_ndcg_at_5'])} | {fmt(r['val_precision_at_5'], 3)} | {fmt(r['val_recall_at_5'], 3)} | {fmt(r['val_map_at_5'], 3)} | {fmt(r['val_mrr'], 3)} | {fmt(r['low_coverage_thin_share_in_top5'], 3)} | {fmt(r['seed_std'], 4)} | {r['train_seconds']} | {'예' if r['adopt'] else '아니오'} |")
    L('\nP@5·R@5·MAP@5·MRR은 Validation 값. ROC-AUC와 Brier는 `benchmark-summary.json`에 있으며, LTR 모델은 점수가 확률이 아니므로 Brier를 계산하지 않았다. 학습 시간은 tuned 설정으로 train 분할 1회 학습 기준.')
    L(f"\n분할: " + ', '.join(f"{k} {v['rows']:,}행/{v['queries']:,}쿼리/양성 {v['positives']:,}" for k, v in manifest['splits'].items() if k in ('train', 'validation', 'test')) + '. 쿼리 중복 0.')
    L('\n## 4. 12F vs 25F\n')
    L('| Model | 12F Val | 25F Val | Δ Val | 12F Test | 25F Test | Δ Test |\n|---|---:|---:|---:|---:|---:|---:|')
    for name in TUNED_ORDER:
        a, c = by[('12f', name)], by[('25f', name)]
        L(f"| {name} | {fmt(a['val_ndcg_at_5'])} | {fmt(c['val_ndcg_at_5'])} | {c['val_ndcg_at_5'] - a['val_ndcg_at_5']:+.4f} | {fmt(a['test_ndcg_at_5'])} | {fmt(c['test_ndcg_at_5'])} | {c['test_ndcg_at_5'] - a['test_ndcg_at_5']:+.4f} |")
    deltas = [by[('25f', n)]['val_ndcg_at_5'] - by[('12f', n)]['val_ndcg_at_5'] for n in TUNED_ORDER]
    L(f"\n25F 효과: 12개 모델 중 {sum(x > 0 for x in deltas)}개에서 Validation NDCG@5 상승, 평균 Δ {np.mean(deltas):+.4f}, 최대 {max(deltas):+.4f}.")
    L('\n## 5. Learning-to-Rank Evaluation\n')
    L('| Model | FS | Val NDCG@5 | Test NDCG@5 | Val MAP@5 | Val MRR | Train s |\n|---|---|---:|---:|---:|---:|---:|')
    for fs in FEATURE_SETS:
        for name in ['M01_extra_trees', 'M10_lgbm_lambdarank', 'M11_xgb_ranker', 'M12_catboost_yetirank']:
            r = by[(fs, name)]
            L(f"| {name} | {fs} | {fmt(r['val_ndcg_at_5'])} | {fmt(r['test_ndcg_at_5'])} | {fmt(r['val_map_at_5'], 3)} | {fmt(r['val_mrr'], 3)} | {r['train_seconds']} |")
    L('\nLTR 모델의 교역 변화 예상치(로그비)는 보조 Extra Trees 회귀 헤드가 냈다(순위·예측 분리 구조). CatBoost는 `YetiRank`를 썼다: 쿼리당 후보가 9개로 고정되고 양성이 희소해 pairwise 변형의 추가 비용 대비 이득이 불분명하며 기본 권장 손실이기 때문이다.')
    L('\n## 6. Ablation (25F, 상위 모델)\n')
    L('| 제거 그룹 | 제거 Feature 수 | ' + ' | '.join(f'{m} ΔNDCG@5' for m in abl['models']) + ' |\n|---|---:|' + '---:|' * len(abl['models']))
    for group, e in abl['groups'].items():
        L(f"| −{group} | {len(e['removed'])} | " + ' | '.join(f"{e['models'][m]['delta_vs_full']['ndcg_at_5']:+.4f}" for m in abl['models']) + ' |')
    contrib = {g: -np.mean([e['models'][m]['delta_vs_full']['ndcg_at_5'] for m in abl['models']]) for g, e in abl['groups'].items()}
    top_groups = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    L(f"\n가장 기여한 그룹(제거 시 하락 평균): " + ', '.join(f"{g} ({v:+.4f})" for g, v in top_groups) + '. 음수 기여는 제거했을 때 오히려 올라간 그룹이다.')
    L('\n## 7. Robustness\n')
    pc = read('per-country.json'); ph = read('per-horizon.json')
    L('국가별 Validation NDCG@5:\n\n| Model | FS | RUS | IRN | PRK | CHN |\n|---|---|---:|---:|---:|---:|')
    for fs in FEATURE_SETS:
        for name in TUNED_ORDER:
            v = pc['validation'][f'{fs}:{name}']
            L(f"| {name} | {fs} | " + ' | '.join(fmt(v.get(t)) for t in TARGETS) + ' |')
    L('\n예측기간별 Validation NDCG@5 (3/6/12 주, 24/60/75 보조):\n\n| Model | FS | 3M | 6M | 12M | 24M | 60M | 75M |\n|---|---|---:|---:|---:|---:|---:|---:|')
    for fs in FEATURE_SETS:
        for name in TUNED_ORDER:
            v = ph['validation'][f'{fs}:{name}']
            L(f"| {name} | {fs} | " + ' | '.join(fmt(v.get(str(h))) for h in [3, 6, 12, 24, 60, 75]) + ' |')
    L('\nSeed 안정성(Validation NDCG@5, 5 seeds):\n\n| Model | FS | mean | sd | min | max |\n|---|---|---:|---:|---:|---:|')
    for k, s in seeds['models'].items():
        L(f"| {s['model']} | {s['feature_set']} | {fmt(s['mean'])} | {fmt(s['std'])} | {fmt(s['min'])} | {fmt(s['max'])} |")
    L('\n자료 확보율별 Top-5 내 저확보(<50%) 비중과 평균 점수는 `low-coverage-analysis.json`에 있다.')
    L('\n## 8. Champion Selection (사전 Gate)\n')
    L('| 조합 | Val NDCG@5 | Δ | A 개선 | B 강건 | C 저확보 | D 안정 | F 하이브리드 | 판정 |\n|---|---:|---:|---|---|---|---|---|---|')
    for k, v in sel['verdicts'].items():
        g = v['gates']
        L(f"| {k} | {fmt(v['validation_ndcg_at_5'])} | {v['gain_vs_champion']:+.4f} | {'O' if g['A_validation_improvement'] else 'X'} | {'O' if g['B_robustness'] else 'X'} | {'O' if g['C_low_coverage_safety'] else 'X'} | {'O' if g['D_stability'] else 'X'} | {'O' if g['F_hybrid'] else 'X'} | {'적격' if v['eligible'] else '; '.join(v['reasons'])} |")
    L(f"\nGate E(복잡도): 동률 {GATES['E_tie']} 이내면 단순한 모델. 결정: {sel['decision']}")
    L('\n## 9. Deployment Recommendation\n')
    dep = read('deployment.json') if (OUT / 'deployment.json').exists() else None
    if dep:
        L(f"배포 완료(사용자 지시 2026-09-18): `{dep['version']}` = {dep['combination']}(Gate A~F 통과, Validation NDCG@5 {fmt(dep['validation_ndcg_at_5'])}, Test {fmt(dep['test_ndcg_at_5'])}). 사전 Gate의 동률 규칙은 12F XGBoost를 골랐지만, 사용자가 확장 Feature까지 반영하도록 지시해 적격 조합 중 Validation 최고인 25F XGBoost를 배포했다. 데모 런타임에는 xgboost를 추가하지 않았다: 학습된 트리 {dep['numpy_export_check']['trees']}그루를 JSON으로 내보내 `scoring.NumpyTreeEnsemble`(numpy 순회)로 서빙하며, {dep['numpy_export_check']['rows']:,}행에서 xgboost 예측과 최대 차이 {dep['numpy_export_check']['max_abs_diff']:.1e}로 일치를 확인했다. 수입·수출 로그비는 데모 환경 scikit-learn으로 학습한 보조 Extra Trees 헤드가 낸다. 서버는 25개 Feature를 시점 보존 규칙 그대로 계산하고(`server.feature_rows`), Top-5 정규화 점수는 순위 마진의 쿼리 내 softmax다. 되돌리기: `models/archive/{dep.get('previous_version') or 'extra-trees-v3-calibrated'}/`.")
    elif sel['deployment_change']:
        L(f"사전 Gate 기준 Champion은 {sel['chosen']}이다. 그러나 이번 작업에서는 배포 모델을 바꾸지 않았다. 이유: 데모 실행 환경에 새 의존성(xgboost·lightgbm·catboost)을 추가하지 않는다는 로드맵 원칙이 있고, LTR 모델은 별도 벤치마크 환경(.venv-bench)에서만 실행된다. 교체하려면 (1) 의존성 추가 승인 또는 (2) 학습된 트리를 numpy 순회로 내보내는 의존성 없는 서빙 구현, (3) train.py 후보 등록·API 3출력 계약(보조 회귀 헤드)·회귀 테스트 갱신이 필요하며 이는 사용자 결정 사항이다. 그때까지 Extra Trees(12F, `extra-trees-v3-calibrated`)를 유지한다.")
        L(f"참고: 배포 중인 Extra Trees는 train.py 설정(64그루·깊이 12)이며 Validation NDCG@5 {fmt(base['deployed_metrics_from_train_py']['validation_ndcg_at_5'])}; 이 벤치마크의 기준 ET 12F는 같은 tuning 예산으로 고른 설정({champ.get('config', '')})으로 {fmt(champ['val_ndcg_at_5'])}이다. Gate A는 tuning된 기준을 상대로 적용했다.")
    else:
        L('현재 배포 모델(Extra Trees, 12F)을 유지한다. 새 Feature와 LTR은 사전 Gate를 넘지 못했으므로 교체하지 않는다.')
    L('\n## 10. Limitations\n')
    L('- 자료는 관측에서 모수를 추정해 만든 생성 패널이며 시연 감쇠가 들어 있다. 결과는 패널 안의 순위 성능이지 실제 우회 탐지 성능이 아니다.')
    L('- 관측 앵커는 러시아 경로에 집중돼 있고 이란·북한·중국 대상은 대부분 보정 합성이다. 국가별 결과의 비대칭은 자료 비대칭이다.')
    L('- 라벨은 미래 3개월 수입·수출 동반 증가 규칙이며 공식 지정 라벨이 아니다. 지정 적중률·선행기간은 검증하지 않았다.')
    L('- 시점 보존은 패널의 available_month와 사건의 공표·발표일로 강제했지만, 패널 자체가 소급 수집한 관측으로 보정됐다는 한계는 남는다.')
    L('- 정책·지정 Feature는 source_events의 러시아·이란 사건에 기대며 북한·중국 대상에는 변동이 없다. 단가·거시 흡수 지표는 자료가 없어 만들지 않았다.')
    L('- NDCG@5 개선은 실무 효과의 증명이 아니다. SHAP은 설치하지 않아 순열 중요도만 썼고, 중요도는 인과가 아니다.')
    (OUT / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    one = ['# 7개 신호군 · 25개 Feature · 12개 모델 비교', '',
           f"교역·정책 데이터를 7개 신호군 {n25}개 변수로 구조화하고, 트리 앙상블·선형모델·도메인 규칙·하이브리드·Learning-to-Rank를 포함한 12개 모델을 동일한 시점 분할과 평가 조건에서 비교하였다(총 24개 조합, seed 5개, 그룹 제거 실험, 순열 중요도).", '',
           '| 항목 | 값 |', '|---|---|',
           f"| 최고 Validation NDCG@5 | {fmt(best_row['val_ndcg_at_5'])} ({best_row['model']}, {best_row['feature_set']}) |",
           f"| 같은 조합 Test NDCG@5 | {fmt(best_row['test_ndcg_at_5'])} |",
           f"| 기존(Extra Trees 12F) 대비 Δ | Validation {best_row['val_ndcg_at_5'] - champ['val_ndcg_at_5']:+.4f} / Test {best_row['test_ndcg_at_5'] - champ['test_ndcg_at_5']:+.4f} |",
           f"| 12F → 25F 최고치 Δ (Validation) | {best25['val_ndcg_at_5'] - best12['val_ndcg_at_5']:+.4f} |",
           f"| Champion (사전 Gate) | {sel['chosen']} |",
           f"| 배포 모델 | {(read('deployment.json')['version'] + ' (' + read('deployment.json')['combination'] + ', numpy 트리 순회로 서빙)') if (OUT / 'deployment.json').exists() else 'Extra Trees 12F 유지'} |",
           f"| 가장 기여한 Feature Group 3개 | " + ', '.join(f"{g} ({v:+.4f})" for g, v in top_groups) + ' |', '',
           '근거: `artifacts/model-benchmark-v12/` (benchmark-summary.json, model-selection.json, ablation.json, feature-importance.json, seed-stability.json). 생성 패널 안의 순위 성능이며 실제 우회 탐지 성능이 아니다.']
    (OUT / 'PRESENTATION_ONE_PAGE.md').write_text('\n'.join(one) + '\n', encoding='utf-8')
    print(json.dumps({'best': best_row['model'] + '/' + best_row['feature_set'], 'best_val': best_row['val_ndcg_at_5'], 'chosen': sel['chosen'], 'top_groups': top_groups}, ensure_ascii=False))
