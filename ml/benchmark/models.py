"""The twelve benchmark models with equal tuning budgets (<= 6 configurations each, chosen on the inner train holdout).

M01-M09 reuse scoring.py (Sk, TwoHead, RuleScorer, GateHybrid, BlendHybrid, StackHybrid). M10-M12 are
learning-to-rank models (LightGBM lambdarank, XGBoost rank:ndcg, CatBoost YetiRank) wrapped so they expose
the same fit(X, Y, inner) / predict(X) -> (n, 3) contract; their ranking score is the ranker output and the
two log-ratio columns come from an auxiliary Extra Trees regression head trained on the same features
(the ranking / forecasting separation of the plan, section 20).
"""
import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scoring import Sk, TwoHead, RuleScorer, GateHybrid, BlendHybrid, StackHybrid, Scorer

ET_CONFIGS = [dict(n_estimators=64, max_depth=12, min_samples_leaf=10), dict(n_estimators=128, max_depth=12, min_samples_leaf=10),
              dict(n_estimators=128, max_depth=16, min_samples_leaf=10), dict(n_estimators=128, max_depth=16, min_samples_leaf=5),
              dict(n_estimators=64, max_depth=8, min_samples_leaf=20), dict(n_estimators=200, max_depth=20, min_samples_leaf=5)]
RF_CONFIGS = ET_CONFIGS
HGB_CONFIGS = [dict(max_iter=300, learning_rate=0.06, max_depth=6), dict(max_iter=600, learning_rate=0.03, max_depth=6),
               dict(max_iter=300, learning_rate=0.06, max_depth=4), dict(max_iter=600, learning_rate=0.06, max_depth=4),
               dict(max_iter=300, learning_rate=0.1, max_depth=6), dict(max_iter=150, learning_rate=0.1, max_depth=8)]
LR_CONFIGS = [dict(C=0.01), dict(C=0.1), dict(C=1.0), dict(C=10.0)]
LGBM_CONFIGS = [dict(num_leaves=15, learning_rate=0.05, n_estimators=200), dict(num_leaves=31, learning_rate=0.05, n_estimators=200),
                dict(num_leaves=31, learning_rate=0.1, n_estimators=200), dict(num_leaves=63, learning_rate=0.05, n_estimators=300),
                dict(num_leaves=15, learning_rate=0.1, n_estimators=400), dict(num_leaves=31, learning_rate=0.03, n_estimators=400)]
XGB_CONFIGS = [dict(max_depth=4, learning_rate=0.05, n_estimators=200), dict(max_depth=6, learning_rate=0.05, n_estimators=200),
               dict(max_depth=6, learning_rate=0.1, n_estimators=200), dict(max_depth=4, learning_rate=0.1, n_estimators=400),
               dict(max_depth=8, learning_rate=0.05, n_estimators=300), dict(max_depth=6, learning_rate=0.03, n_estimators=400)]
CAT_CONFIGS = [dict(depth=4, learning_rate=0.05, iterations=300), dict(depth=6, learning_rate=0.05, iterations=300),
               dict(depth=6, learning_rate=0.1, iterations=300), dict(depth=4, learning_rate=0.1, iterations=600),
               dict(depth=8, learning_rate=0.05, iterations=300), dict(depth=6, learning_rate=0.03, iterations=600)]


def et_scorer(cfg, seed):
    return Sk(ExtraTreesRegressor(**cfg, max_features=.85, random_state=seed, n_jobs=4), 'extra_trees')


def rf_scorer(cfg, seed):
    return Sk(RandomForestRegressor(**cfg, max_features=.85, random_state=seed, n_jobs=4), 'random_forest')


def hgb_scorer(cfg, seed):
    common = dict(min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    return TwoHead(HistGradientBoostingClassifier(**cfg, **common), MultiOutputRegressor(HistGradientBoostingRegressor(**cfg, **common)), 'hgb')


def lr_scorer(cfg, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], max_iter=2000, random_state=seed))
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0 / cfg['C']))
    return TwoHead(clf, reg, 'logistic')


class AuxRegression:
    """Extra Trees head for the two log ratios; used by ranking-only models (deterministic seed)."""
    def __init__(self):
        self.model = ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=10, max_features=.85, random_state=42, n_jobs=4)

    def fit(self, X, Y):
        self.model.fit(X, Y[:, 1:])
        return self

    def predict(self, X):
        return self.model.predict(X)


def _sort_groups(X, y, groups):
    order = np.argsort(groups, kind='stable')
    g = groups[order]
    sizes = np.diff(np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)])
    return X[order], y[order], g, sizes


class LTRScorer(Scorer):
    """Learning-to-rank wrapper. fit needs query groups: pass them through fit(X, Y, inner, groups=...)."""
    probabilistic = False

    def __init__(self, kind, cfg, seed, name):
        self.kind = kind; self.cfg = cfg; self.seed = seed; self.name = name
        self.aux = AuxRegression(); self.ranker = None

    def _build(self):
        if self.kind == 'lgbm':
            import lightgbm as lgb
            return lgb.LGBMRanker(objective='lambdarank', min_child_samples=20, subsample=0.9, subsample_freq=1,
                                  colsample_bytree=0.9, random_state=self.seed, n_jobs=4, verbose=-1, **self.cfg)
        if self.kind == 'xgb':
            import xgboost as xgb
            return xgb.XGBRanker(objective='rank:ndcg', eval_metric='ndcg@5', lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=5,
                                 subsample=0.9, colsample_bytree=0.9, random_state=self.seed, n_jobs=4, tree_method='hist', **self.cfg)
        if self.kind == 'cat':
            from catboost import CatBoostRanker
            return CatBoostRanker(loss_function='YetiRank', eval_metric='NDCG:top=5', random_seed=self.seed, thread_count=4, verbose=False, allow_writing_files=False, **self.cfg)
        raise ValueError(self.kind)

    def fit(self, X, Y, inner=None, groups=None):
        if groups is None:
            raise ValueError('LTR models need query groups')
        X = np.asarray(X, float); y = np.asarray(Y)[:, 0].astype(int); self.n_features_ = X.shape[1]
        Xs, ys, gs, sizes = _sort_groups(X, y, np.asarray(groups))
        self.ranker = self._build()
        if self.kind == 'lgbm':
            self.ranker.fit(Xs, ys, group=sizes)
        elif self.kind == 'xgb':
            qid = np.repeat(np.arange(len(sizes)), sizes)
            self.ranker.fit(Xs, ys, qid=qid)
        else:
            from catboost import Pool
            gid = np.repeat(np.arange(len(sizes)), sizes)
            self.ranker.fit(Pool(Xs, ys, group_id=gid))
        self.aux.fit(X, np.asarray(Y, float))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        score = np.asarray(self.ranker.predict(X), float)
        return np.column_stack([score, self.aux.predict(X)])

    @property
    def feature_importances_(self):
        r = self.ranker
        try:
            if self.kind == 'cat':
                values = r.get_feature_importance(type='PredictionValuesChange')
            else:
                values = r.feature_importances_
            values = np.asarray(values, float).ravel()
            return values if values.size == self.n_features_ else None
        except Exception:
            return None


MODELS = {
    'M01_extra_trees': dict(label='Extra Trees Regressor', type='tree ensemble (multi-output)', configs=ET_CONFIGS, factory=et_scorer, stochastic=True, probabilistic=True, components=[]),
    'M02_hgb': dict(label='Histogram Gradient Boosting (two-head)', type='boosting', configs=HGB_CONFIGS, factory=hgb_scorer, stochastic=True, probabilistic=True, components=[]),
    'M03_random_forest': dict(label='Random Forest Regressor', type='tree ensemble (multi-output)', configs=RF_CONFIGS, factory=rf_scorer, stochastic=True, probabilistic=True, components=[]),
    'M04_logistic': dict(label='Logistic Regression (+Ridge ratios)', type='linear', configs=LR_CONFIGS, factory=lr_scorer, stochastic=False, probabilistic=True, components=[]),
    'M05_rule': dict(label='Rule-based Score', type='rules', configs=[{}], factory=lambda cfg, seed: RuleScorer(), stochastic=False, probabilistic=True, components=[]),
    'M06_rule_gate': dict(label='Rule Gate (ET score capped by coverage/mirror rule)', type='hybrid', configs=[{}], factory=None, stochastic=True, probabilistic=True, components=['M01_extra_trees']),
    'M07_weighted_blend': dict(label='Weighted Blend (rule, ET, HGB)', type='hybrid', configs=[{}], factory=None, stochastic=True, probabilistic=True, components=['M05_rule', 'M01_extra_trees', 'M02_hgb']),
    'M08_stacking': dict(label='Stacking (logistic meta over rule, ET, HGB)', type='hybrid', configs=[{}], factory=None, stochastic=True, probabilistic=True, components=['M05_rule', 'M01_extra_trees', 'M02_hgb']),
    'M09_two_head': dict(label='Two-head (HGB classifier + ET regressor)', type='two-head', configs=[{}], factory=None, stochastic=True, probabilistic=True, components=['M02_hgb', 'M01_extra_trees']),
    'M10_lgbm_lambdarank': dict(label='LightGBM LambdaRank', type='learning-to-rank', configs=LGBM_CONFIGS, factory=lambda cfg, seed: LTRScorer('lgbm', cfg, seed, 'lgbm_lambdarank'), stochastic=True, probabilistic=False, components=[]),
    'M11_xgb_ranker': dict(label='XGBoost Ranker (rank:ndcg)', type='learning-to-rank', configs=XGB_CONFIGS, factory=lambda cfg, seed: LTRScorer('xgb', cfg, seed, 'xgb_ranker'), stochastic=True, probabilistic=False, components=[]),
    'M12_catboost_yetirank': dict(label='CatBoost YetiRank', type='learning-to-rank', configs=CAT_CONFIGS, factory=lambda cfg, seed: LTRScorer('cat', cfg, seed, 'catboost_yetirank'), stochastic=True, probabilistic=False, components=[]),
}
TUNED_ORDER = ['M01_extra_trees', 'M02_hgb', 'M03_random_forest', 'M04_logistic', 'M05_rule', 'M10_lgbm_lambdarank', 'M11_xgb_ranker', 'M12_catboost_yetirank',
               'M06_rule_gate', 'M07_weighted_blend', 'M08_stacking', 'M09_two_head']
SIMPLICITY = ['M04_logistic', 'M05_rule', 'M01_extra_trees', 'M03_random_forest', 'M02_hgb', 'M10_lgbm_lambdarank', 'M11_xgb_ranker', 'M12_catboost_yetirank',
              'M09_two_head', 'M06_rule_gate', 'M07_weighted_blend', 'M08_stacking']


def make_model(name, best, seed):
    """Instantiate a model with its tuned configuration; hybrids draw their components' tuned configs from `best`."""
    spec = MODELS[name]
    if spec['factory'] is not None:
        return spec['factory'](best.get(name, spec['configs'][0]), seed)
    et = et_scorer(best.get('M01_extra_trees', ET_CONFIGS[0]), seed)
    hg = hgb_scorer(best.get('M02_hgb', HGB_CONFIGS[0]), seed)
    if name == 'M06_rule_gate':
        return GateHybrid(et)
    if name == 'M07_weighted_blend':
        return BlendHybrid([RuleScorer(), et, hg])
    if name == 'M08_stacking':
        return StackHybrid([RuleScorer(), et, hg])
    if name == 'M09_two_head':
        cfg = best.get('M02_hgb', HGB_CONFIGS[0])
        return TwoHead(HistGradientBoostingClassifier(**cfg, min_samples_leaf=40, l2_regularization=1.0, random_state=seed),
                       ExtraTreesRegressor(**best.get('M01_extra_trees', ET_CONFIGS[0]), max_features=.85, random_state=seed, n_jobs=4), 'two_head')
    raise ValueError(name)


def fit_model(model, X, Y, inner, groups):
    """Uniform fit: LTR models receive query groups; others the inner split for hybrids."""
    t = time.perf_counter()
    if isinstance(model, LTRScorer):
        model.fit(X, Y, None, groups=groups)
    else:
        model.fit(X, Y, inner)
    return time.perf_counter() - t
