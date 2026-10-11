"""Benchmark v14: dataset with the v2 features and the shared helpers of the experiment scripts.

  build   every target x basis block (FeatureBuilderV2.block), labels, splits -> artifacts/model-benchmark-v14/cache/dataset.npz
          and split-manifest.json. Training bases are monthly; validation and test keep the v12/v13 grid (every fourth
          month from index 60), so their queries are the same rows the earlier benchmarks were scored on.
  replicas  independent draws of the panel up to the training cut-off (benchmark.replay) -> cache/replica-N.npz, for the
          data-volume experiment.
Rounds 2 and 3 read a grid-only build with the experimental families: build --basis-step 4 --name grid.npz --extended, and
replicas --extended.

Run with the benchmark environment (requirements-benchmark.txt + optuna): .venv-bench/Scripts/python -m benchmark.lab build   (in ml/)
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
from common import DB, TARGETS, mt

OUT = ROOT / 'artifacts/model-benchmark-v14'
CACHE = OUT / 'cache'
HORIZONS = (3, 6, 12, 24, 60, 75)
ARRAYS = ('X', 'Y', 'R', 'B', 'A', 'G', 'H', 'T', 'HS', 'C', 'grid')


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


def load_dataset(path=None):
    z = np.load(path or CACHE / 'dataset.npz', allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d['masks'] = {k: d.pop(f'mask_{k}') for k in ('train', 'validation', 'test')}
    d['inner'] = {k: d.pop(f'inner_{k}') for k in ('fit', 'holdout')}
    d['features'] = list(d['features'])
    return d


def _build_target(target, basis_step, extended):
    from benchmark.features_v2 import FeatureBuilderV2, build_dataset_v2
    d = build_dataset_v2(FeatureBuilderV2(extended=extended), basis_step=basis_step, horizons=HORIZONS, targets=[target])
    return {k: d[k] for k in ARRAYS} | {'features': d['features'], 'panel_end': d['panel_end'], 'masks': d['masks'], 'inner': d['inner']}


def stage_build(basis_step=1, name='dataset.npz', extended=False):
    from joblib import Parallel, delayed
    t0 = time.time()
    parts = Parallel(n_jobs=len(TARGETS))(delayed(_build_target)(t, basis_step, extended) for t in TARGETS)
    d = {k: np.concatenate([p[k] for p in parts]) for k in ARRAYS}
    masks = {k: np.concatenate([p['masks'][k] for p in parts]) for k in ('train', 'validation', 'test')}
    inner = {k: np.concatenate([p['inner'][k] for p in parts]) for k in ('fit', 'holdout')}
    features = parts[0]['features']; panel_end = int(parts[0]['panel_end'])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / name, **d, features=np.array(features), panel_end=panel_end,
                        **{f'mask_{k}': v for k, v in masks.items()}, **{f'inner_{k}': v for k, v in inner.items()})
    meta = panel_metadata()
    manifest = {'schema_version': 'btl-v14-benchmark-split-v1', 'built_at_utc': now(),
                'panel': {'dataset_version': meta.get('dataset_version'), 'built_at': meta.get('built_at'), 'parameters_sha256': meta.get('parameters_sha256'), 'panel_end': mt(panel_end)},
                'query_definition': 'target x hs6 x basis_month x horizon; documents are the candidate transit countries',
                'sampling': f'training bases every {basis_step} month(s) from index 60; validation, test and the inner holdout only on the v12/v13 grid (every 4 months from index 60); horizons 3/6/12/24/60/75',
                'label': 'future 3-month-average import ratio > 1.15 AND export ratio > 1.20 vs latest available 3-month average (unchanged v3 contract)',
                'split_rule': {'train': 'label maturity (basis+horizon+7) <= 2018-12', 'validation': 'grid basis >= 2019-01 and maturity <= 2021-12',
                               'test': 'grid basis >= 2022-01 and maturity <= panel end', 'inner_fit': 'train with maturity <= 2015-12',
                               'inner_holdout': 'train on the grid with basis >= 2016-01 (model, feature and hyperparameter search)'},
                'features': len(features), 'splits': {}, 'extraction_seconds': round(time.time() - t0, 1)}
    for split_name, mask in list(masks.items()) + [('inner_fit', inner['fit']), ('inner_holdout', inner['holdout'])]:
        g = d['G'][mask]; y = d['Y'][mask, 0]; b = d['B'][mask]
        queries, inverse = np.unique(g, return_inverse=True)
        with_positive = np.bincount(inverse, weights=y) > 0
        manifest['splits'][split_name] = {'rows': int(mask.sum()), 'queries': int(len(queries)), 'queries_with_positive': int(with_positive.sum()), 'positives': int(y.sum()),
                                    'positive_rate': round(float(y.mean()), 4), 'basis_range': [mt(int(b.min())), mt(int(b.max()))],
                                    'rows_by_target': {t: int((d['T'][mask] == t).sum()) for t in TARGETS},
                                    'rows_by_horizon': {str(h): int((d['H'][mask] == h).sum()) for h in HORIZONS},
                                    'query_ids_sha256': hashlib.sha256('\n'.join(sorted(queries)).encode()).hexdigest()}
    sets = {k: set(np.unique(d['G'][masks[k]])) for k in masks}
    manifest['query_overlap'] = {'train_validation': len(sets['train'] & sets['validation']), 'train_test': len(sets['train'] & sets['test']), 'validation_test': len(sets['validation'] & sets['test'])}
    A, B = d['A'], d['B']
    manifest['time_order'] = {'max_train_maturity': mt(int(A[masks['train']].max())), 'min_validation_basis': mt(int(B[masks['validation']].min())),
                              'max_validation_maturity': mt(int(A[masks['validation']].max())), 'min_test_basis': mt(int(B[masks['test']].min())),
                              'max_inner_fit_maturity': mt(int(A[inner['fit']].max())), 'min_inner_holdout_basis': mt(int(B[inner['holdout']].min()))}
    if name == 'dataset.npz':  # a scratch build (other name, e.g. grid-only for quick feature iterations) leaves the manifest alone
        write('split-manifest.json', manifest)
    print(json.dumps({k: v['rows'] for k, v in manifest['splits'].items()}), 'overlap', manifest['query_overlap'], 'seconds', manifest['extraction_seconds'])


def _build_replica(tag, target, basis_step, extended):
    from common import mi
    from benchmark.features_v2 import FeatureBuilderV2, build_dataset_v2
    from benchmark.replay import ReplicaPanel
    panel = ReplicaPanel(tag, mi(REPLICA_END))
    d = build_dataset_v2(FeatureBuilderV2(panel, extended=extended), basis_step=basis_step, horizons=HORIZONS, targets=[target])
    d['G'] = np.char.add(tag + '|', d['G'])  # replica queries never share an id with a panel query
    return {k: d[k] for k in ARRAYS} | {'features': d['features']}


def stage_replicas(count, basis_step=4, first=1, extended=False):
    """Independent draws of the panel up to the training cut-off (benchmark.replay), with the same features and labels.
    They are training rows only: every row matures by REPLICA_END, and nothing after it is generated."""
    from joblib import Parallel, delayed
    for k in range(first, first + count):
        t0 = time.time(); tag = f'replica-{k}'
        parts = Parallel(n_jobs=len(TARGETS))(delayed(_build_replica)(tag, t, basis_step, extended) for t in TARGETS)
        d = {key: np.concatenate([p[key] for p in parts]) for key in ARRAYS}
        np.savez_compressed(CACHE / f'{tag}.npz', **d, features=np.array(parts[0]['features']))
        print(json.dumps({'replica': tag, 'rows': int(len(d['Y'])), 'positive_rate': round(float(d['Y'][:, 0].mean()), 4), 'seconds': round(time.time() - t0, 1)}), flush=True)


REPLICA_END = '2018-12'  # the training cut-off: a replica holds no month after it


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'replicas'])
    parser.add_argument('--basis-step', type=int, default=None)
    parser.add_argument('--name', default='dataset.npz')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--extended', action='store_true', help='add the experimental feature families (features_v2_extra)')
    args = parser.parse_args()
    if args.stage == 'build':
        stage_build(args.basis_step or 1, args.name, args.extended)
    else:
        stage_replicas(args.count, args.basis_step or 4, args.first, args.extended)
