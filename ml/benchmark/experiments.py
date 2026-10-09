"""Benchmark v14 experiment helpers: rolling-origin folds inside the training period, model families, and the experiment log.

Evaluation discipline
  search      three rolling-origin folds inside the training period (fit: label maturity <= cut; score: grid bases after the cut,
              maturity within 36 months). Every model, feature and hyperparameter choice is made on their mean.
  confirm     train split -> validation split, looked at for a small number of frozen candidates.
  final       train split -> test split, once, for the adopted configuration (and the old baseline as reference).
Primary metric: NDCG@5 over queries with at least one positive (`ndcg_at_5_pos`, benchmark.metrics).

Every run appends one line to artifacts/model-benchmark-v14/experiments.jsonl.
"""
import json
import time

import numpy as np

from benchmark.lab import OUT, now  # sets the import path for `common`
from common import mi
from benchmark.metrics import ranking_metrics, row_metrics

PRIMARY = 'ndcg_at_5_pos'
FOLD_CUTS = ('2009-12', '2012-12', '2015-12')
LOG = OUT / 'experiments.jsonl'
THREADS = int(__import__('os').environ.get('BTL_BENCH_THREADS', 8))


def folds(d, train_rows=None):
    """[(name, fit mask, score mask)] for the rolling-origin folds; `train_rows` optionally thins the fit rows (e.g. the grid only)."""
    A, B, grid = d['A'], d['B'], d['grid']
    out = []
    for cut in FOLD_CUTS:
        c = mi(cut)
        fit = d['masks']['train'] & (A <= c)
        if train_rows is not None: fit = fit & train_rows
        score = d['masks']['train'] & grid & (B >= c + 1) & (A <= c + 36)
        out.append((f'fold_{cut[:4]}', fit, score))
    return out


def _groups(g):
    order = np.argsort(g, kind='stable'); gs = g[order]
    sizes = np.diff(np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(gs)])
    return order, sizes


def graded(d, mask):
    """Graded relevance for training only: 0 negative, 1 positive, 2 positive with both legs at least twice the threshold margin."""
    r = d['R'][mask]; y = d['Y'][mask, 0]
    strong = (r[:, 0] > 2 * np.log(1.15)) & (r[:, 1] > 2 * np.log(1.20))
    return (y + (y > 0) * strong).astype(int)


class Model:
    """Uniform wrapper: fit(X, d, mask) / score(X) -> ranking score. `kind` picks the learner family."""

    def __init__(self, kind, params=None, seed=42, target='binary'):
        self.kind = kind; self.params = dict(params or {}); self.seed = seed; self.target = target; self.model = None

    def fit(self, X, d, mask):
        Xm = X[mask]; y = d['Y'][mask, 0].astype(int)
        p = self.params; k = self.kind
        if k in ('xgb_rank', 'lgbm_rank', 'cat_rank'):
            order, sizes = _groups(d['G'][mask])
            rel = (graded(d, mask) if self.target == 'graded' else y)[order]; Xs = Xm[order]
            if k == 'xgb_rank':
                import xgboost as xgb
                base = dict(objective='rank:ndcg', lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=5, subsample=0.9, colsample_bytree=0.9,
                            tree_method='hist', n_jobs=THREADS, random_state=self.seed, max_depth=8, learning_rate=0.05, n_estimators=300)
                self.model = xgb.XGBRanker(**{**base, **p}).fit(Xs, rel, qid=np.repeat(np.arange(len(sizes)), sizes))
            elif k == 'lgbm_rank':
                import lightgbm as lgb
                base = dict(objective='lambdarank', lambdarank_truncation_level=9, min_child_samples=20, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                            n_jobs=THREADS, random_state=self.seed, verbose=-1, num_leaves=63, learning_rate=0.05, n_estimators=300)
                self.model = lgb.LGBMRanker(**{**base, **p}).fit(Xs, rel, group=sizes)
            else:
                from catboost import CatBoostRanker, Pool
                base = dict(loss_function='YetiRank', random_seed=self.seed, thread_count=THREADS, verbose=False, allow_writing_files=False, depth=6, learning_rate=0.1, iterations=500)
                self.model = CatBoostRanker(**{**base, **p}).fit(Pool(Xs, rel, group_id=np.repeat(np.arange(len(sizes)), sizes)))
        elif k == 'xgb_clf':
            import xgboost as xgb
            base = dict(objective='binary:logistic', subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=THREADS, random_state=self.seed,
                        max_depth=6, learning_rate=0.05, n_estimators=400)
            self.model = xgb.XGBClassifier(**{**base, **p}).fit(Xm, y)
        elif k == 'lgbm_clf':
            import lightgbm as lgb
            base = dict(objective='binary', min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=THREADS, random_state=self.seed, verbose=-1,
                        num_leaves=63, learning_rate=0.05, n_estimators=400)
            self.model = lgb.LGBMClassifier(**{**base, **p}).fit(Xm, y)
        elif k == 'cat_clf':
            from catboost import CatBoostClassifier
            base = dict(random_seed=self.seed, thread_count=THREADS, verbose=False, allow_writing_files=False, depth=6, learning_rate=0.1, iterations=500)
            self.model = CatBoostClassifier(**{**base, **p}).fit(Xm, y)
        elif k == 'xgb_reg':
            # Two regressors for the log ratios; the score is the joint probability of clearing both thresholds under independent normal errors.
            import xgboost as xgb
            base = dict(objective='reg:squarederror', subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=THREADS, random_state=self.seed,
                        max_depth=6, learning_rate=0.05, n_estimators=400)
            Yr = d['Y'][mask, 1:]
            self.model = [xgb.XGBRegressor(**{**base, **p}).fit(Xm, Yr[:, j]) for j in range(2)]
            self.sigma = [float(np.std(Yr[:, j] - self.model[j].predict(Xm))) * 1.25 for j in range(2)]  # in-sample residuals understate the error
        elif k in ('et', 'rf'):
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            base = dict(n_estimators=300, max_depth=None, min_samples_leaf=10, max_features=0.5, n_jobs=THREADS, random_state=self.seed)
            self.model = (ExtraTreesClassifier if k == 'et' else RandomForestClassifier)(**{**base, **p}).fit(np.nan_to_num(Xm), y)
        elif k == 'mlp':
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import QuantileTransformer
            base = dict(hidden_layer_sizes=(256, 128, 64), alpha=1e-4, batch_size=512, learning_rate_init=1e-3, max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=self.seed)
            self.model = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=0), MLPClassifier(**{**base, **p})).fit(np.nan_to_num(Xm), y)
        elif k == 'logit':
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import QuantileTransformer
            self.model = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=0), LogisticRegression(C=p.get('C', 1.0), max_iter=500)).fit(np.nan_to_num(Xm), y)
        else:
            raise ValueError(k)
        return self

    def score(self, X):
        k = self.kind
        if k in ('xgb_rank', 'lgbm_rank', 'cat_rank'):
            return np.asarray(self.model.predict(X), float)
        if k == 'xgb_clf':
            return np.asarray(self.model.predict(X, output_margin=True), float)
        if k == 'lgbm_clf':
            return np.asarray(self.model.predict(X, raw_score=True), float)
        if k == 'cat_clf':
            return np.asarray(self.model.predict(X, prediction_type='RawFormulaVal'), float)
        if k == 'xgb_reg':
            from scipy.stats import norm
            from benchmark.features_v2 import THRESHOLD
            z = [(self.model[j].predict(X) - THRESHOLD[j]) / self.sigma[j] for j in range(2)]
            return norm.logcdf(z[0]) + norm.logcdf(z[1])
        return self.model.predict_proba(np.nan_to_num(X))[:, 1]


def columns(d, names):
    index = {n: i for i, n in enumerate(d['features'])}
    return [index[n] for n in names]


def score_split(d, mask, score):
    out = ranking_metrics(d['Y'][mask, 0], score, d['G'][mask])
    out.update(row_metrics(d['Y'][mask, 0], score, False))
    return out


def run_folds(d, X, make, train_rows=None, extra=None):
    """Fit and score on the three inner folds; returns {'cv': mean primary, 'folds': {...}}."""
    result = {}; values = []
    for name, fit, score in folds(d, train_rows):
        t = time.perf_counter(); model = make().fit(X, d, fit); seconds = time.perf_counter() - t
        m = score_split(d, score, model.score(X[score]))
        result[name] = {k: m[k] for k in (PRIMARY, 'ndcg_at_5', 'precision_at_5', 'recall_at_5', 'map', 'mrr', 'roc_auc', 'pr_auc', 'queries_with_positive')} | {'fit_seconds': round(seconds, 1), 'fit_rows': int(fit.sum())}
        values.append(m[PRIMARY])
    return {'cv': float(np.mean(values)), 'cv_min': float(np.min(values)), 'folds': result}


def run_validation(d, X, make, train_rows=None, breakdown=False):
    train = d['masks']['train'] if train_rows is None else d['masks']['train'] & train_rows
    val = d['masks']['validation']
    t = time.perf_counter(); model = make().fit(X, d, train); seconds = time.perf_counter() - t
    s = model.score(X[val]); m = score_split(d, val, s)
    out = {**m, 'fit_seconds': round(seconds, 1), 'fit_rows': int(train.sum())}
    if breakdown:
        out['by_target'] = {t_: ranking_metrics(d['Y'][val][d['T'][val] == t_, 0], s[d['T'][val] == t_], d['G'][val][d['T'][val] == t_])[PRIMARY] for t_ in np.unique(d['T'][val])}
        out['by_horizon'] = {str(h): ranking_metrics(d['Y'][val][d['H'][val] == h, 0], s[d['H'][val] == h], d['G'][val][d['H'][val] == h])[PRIMARY] for h in np.unique(d['H'][val])}
    return out, model, s


def log(record):
    OUT.mkdir(parents=True, exist_ok=True)
    record = {'at_utc': now(), **record}
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=float) + '\n')
    return record
