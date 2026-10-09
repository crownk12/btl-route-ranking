"""Candidate scorers for the generated-panel model selection (track A).

Every scorer exposes fit(X, Y, inner=None) and predict(X) -> (n, 3): column 0 is the
coupled-growth score used for ranking, columns 1-2 are the import/export log ratios
used for the forecast curves. `inner` is an out-of-sample slice of the training
period (fit part, holdout part) used only by hybrids to set blend weights or a
stacking meta-model; the validation and test splits never touch it.

Rule thresholds below were fixed before any candidate was evaluated and are not
tuned on validation or test.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ndcg_score

# Feature indexes from common.FEATURES
IMPORT_GROWTH, EXPORT_GROWTH, IMPORT_YOY, EXPORT_YOY, COUPLED, IMPORT_VOL, EXPORT_VOL, COVERAGE, MIRROR_GAP, EVENT_COUNT, EVENT_AGE, HORIZON = range(12)

RULE = dict(coupled_full=0.5, yoy_full=0.6, coverage_full=0.6, mirror_free=0.3, event_half_life=36.0,
            weights=dict(coupled=0.45, yoy=0.35, event=0.20), ratio_persistence=0.5, score_floor=0.05)


def _arr(X):
    return np.asarray(X, dtype=float)


def coverage_cap(X):
    """Upper bound for a score when coverage is thin or mirror reports disagree (1.0 = no cap)."""
    X = _arr(X)
    cov = np.clip(X[:, COVERAGE] / RULE['coverage_full'], 0, 1)
    mirror = 1.0 / (1.0 + np.maximum(0.0, X[:, MIRROR_GAP] - RULE['mirror_free']))
    return RULE['score_floor'] + (1 - RULE['score_floor']) * cov * mirror


def rule_score(X):
    X = _arr(X)
    w = RULE['weights']
    coupled = np.clip(X[:, COUPLED] / RULE['coupled_full'], 0, 1)
    yoy = np.clip(np.minimum(X[:, IMPORT_YOY], X[:, EXPORT_YOY]) / RULE['yoy_full'], 0, 1)
    has_event = X[:, EVENT_COUNT] > 0
    event = np.where(has_event, np.exp(-X[:, EVENT_AGE] / RULE['event_half_life']), 0.0)
    raw = w['coupled'] * coupled + w['yoy'] * yoy + w['event'] * event
    return raw * coverage_cap(X)


def rule_ratios(X):
    """Persistence forecast: recent 3-month growth carried forward with a fixed damping."""
    X = _arr(X)
    growth = np.clip(X[:, [IMPORT_GROWTH, EXPORT_GROWTH]], -0.9, 5.0)
    return np.clip(RULE['ratio_persistence'] * np.log1p(growth), -2, 2)


class Scorer:
    name = 'scorer'
    components = ()

    def set_params(self, **params):
        for child in self.components:
            if hasattr(child, 'set_params'):
                try:
                    child.set_params(**params)
                except ValueError:
                    pass
        return self

    def fit(self, X, Y, inner=None):
        return self

    def predict(self, X):
        raise NotImplementedError


class Sk(Scorer):
    """A scikit-learn multi-output regressor that already predicts (n, 3)."""
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.components = (model,)

    def fit(self, X, Y, inner=None):
        self.model.fit(_arr(X), _arr(Y))
        return self

    def predict(self, X):
        return self.model.predict(_arr(X))

    @property
    def feature_importances_(self):
        return getattr(self.model, 'feature_importances_', None)


class TwoHead(Scorer):
    """Separate ranking head (classifier probability) and magnitude head (regressor for the two log ratios)."""
    def __init__(self, classifier, regressor, name):
        self.classifier = classifier
        self.regressor = regressor
        self.name = name
        self.components = (classifier, regressor)

    def fit(self, X, Y, inner=None):
        X = _arr(X); Y = _arr(Y)
        self.classifier.fit(X, Y[:, 0].astype(int))
        self.regressor.fit(X, Y[:, 1:])
        return self

    def predict(self, X):
        X = _arr(X)
        score = self.classifier.predict_proba(X)[:, 1]
        return np.column_stack([score, self.regressor.predict(X)])

    @property
    def feature_importances_(self):
        return getattr(self.classifier, 'feature_importances_', None)


class RuleScorer(Scorer):
    name = 'rule'

    def predict(self, X):
        return np.column_stack([rule_score(X), rule_ratios(X)])


class GateHybrid(Scorer):
    """Model score capped by the rule's coverage/mirror bound; log ratios from the model."""
    def __init__(self, base, name='gate_hybrid'):
        self.base = base
        self.name = name
        self.components = (base,)

    def fit(self, X, Y, inner=None):
        self.base.fit(X, Y, inner)
        return self

    def predict(self, X):
        p = np.array(self.base.predict(X), dtype=float)
        p[:, 0] = np.minimum(p[:, 0], coverage_cap(X))
        return p


def _ndcg5(y, s, groups):
    values = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 5:
            continue
        values.append(ndcg_score([y[m]], [s[m]], k=5))
    return float(np.mean(values)) if values else 0.0


def _simplex_grid(k, step=0.1):
    steps = int(round(1 / step))
    if k == 1:
        yield (1.0,)
        return
    for i in range(steps + 1):
        for rest in _simplex_grid(k - 1, step):
            total = i * step + sum(rest)
            if abs(total) < 1e-9 or total <= 1 + 1e-9:
                yield (round(i * step, 10),) + rest
    return


class BlendHybrid(Scorer):
    """Weighted mean of component scores and log ratios; weights chosen on the inner train holdout by NDCG@5."""
    def __init__(self, bases, name='blend_hybrid', step=0.1):
        self.bases = list(bases)
        self.name = name
        self.step = step
        self.weights = None
        self.components = tuple(bases)

    def fit(self, X, Y, inner=None):
        if inner is None:
            raise ValueError('BlendHybrid needs the inner train split to set weights')
        Xf, Yf, Xh, Yh, Gh = inner
        preds = [np.array(b.fit(Xf, Yf, None).predict(Xh))[:, 0] for b in self.bases]
        best = None
        for w in _simplex_grid(len(self.bases), self.step):
            if abs(sum(w) - 1) > 1e-6:
                continue
            s = sum(wi * p for wi, p in zip(w, preds))
            value = _ndcg5(Yh[:, 0], s, Gh)
            if best is None or value > best[0] + 1e-12:
                best = (value, w)
        self.weights = np.array(best[1])
        self.inner_ndcg = best[0]
        for b in self.bases:
            b.fit(X, Y, inner)
        return self

    def predict(self, X):
        preds = [np.array(b.predict(X), dtype=float) for b in self.bases]
        return sum(w * p for w, p in zip(self.weights, preds))


class StackHybrid(Scorer):
    """Logistic meta-model over component scores (fitted on the inner train holdout); log ratios averaged."""
    def __init__(self, bases, name='stack_hybrid'):
        self.bases = list(bases)
        self.name = name
        self.meta = LogisticRegression(C=1.0, max_iter=1000)
        self.components = tuple(bases)

    def fit(self, X, Y, inner=None):
        if inner is None:
            raise ValueError('StackHybrid needs the inner train split to fit the meta-model')
        Xf, Yf, Xh, Yh, Gh = inner
        stack = np.column_stack([np.array(b.fit(Xf, Yf, None).predict(Xh))[:, 0] for b in self.bases])
        self.meta.fit(stack, Yh[:, 0].astype(int))
        for b in self.bases:
            b.fit(X, Y, inner)
        return self

    def predict(self, X):
        preds = [np.array(b.predict(X), dtype=float) for b in self.bases]
        stack = np.column_stack([p[:, 0] for p in preds])
        score = self.meta.predict_proba(stack)[:, 1]
        ratios = np.mean([p[:, 1:] for p in preds], axis=0)
        return np.column_stack([score, ratios])


def logistic_two_head(seed=42):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return TwoHead(clf, reg, 'logistic')


class NumpyTreeEnsemble:
    """Dependency-free evaluator for an XGBoost tree ensemble exported with save_model(*.json).

    Prediction is the raw margin: base_score plus the sum of leaf weights, following the same
    split rule as XGBoost (x < threshold goes left; missing values follow default_left).
    """
    def __init__(self, trees, base_score, num_feature):
        self.trees = trees  # list of dicts with numpy arrays: left, right, index, threshold, default_left, weight
        self.base_score = float(base_score)
        self.num_feature = int(num_feature)

    @classmethod
    def from_xgboost_json(cls, model, base_margin=None):
        """base_margin: the raw-margin intercept when the caller measured it (a classifier stores its base score as a
        probability, a ranker as a margin); default: the stored base score as it is, which is right for the rankers."""
        learner = model['learner']
        base = learner['learner_model_param']['base_score']
        base = float(str(base).strip('[]').split(',')[0]) if base_margin is None else float(base_margin)
        trees = []
        for t in learner['gradient_booster']['model']['trees']:
            trees.append(dict(left=np.asarray(t['left_children'], int), right=np.asarray(t['right_children'], int),
                              index=np.asarray(t['split_indices'], int), threshold=np.asarray(t['split_conditions'], np.float32),
                              default_left=np.asarray(t['default_left'], bool),
                              # leaf values live in split_conditions for leaf nodes (base_weights are the unscaled node weights)
                              weight=np.asarray(t['split_conditions'], float)))
        return cls(trees, base, learner['learner_model_param']['num_feature'])

    @classmethod
    def combine(cls, parts):
        """One ensemble whose margin is sum(weight * margin of each part): leaf values scaled, trees concatenated.
        parts: [(NumpyTreeEnsemble, weight)]."""
        trees = []
        for ensemble, weight in parts:
            # only the leaf entries of `weight` are read at prediction time, so scaling the whole array is safe
            trees += [dict(t, weight=t['weight'] * float(weight)) for t in ensemble.trees]
        return cls(trees, sum(float(w) * e.base_score for e, w in parts), parts[0][0].num_feature)

    def predict_margin(self, X):
        X = np.asarray(X, np.float32)  # XGBoost compares in float32; matching it keeps boundary rows on the same side
        out = np.full(len(X), self.base_score)
        rows = np.arange(len(X))
        for t in self.trees:
            node = np.zeros(len(X), int)
            active = t['left'][node] != -1
            while active.any():
                idx = t['index'][node]
                value = X[rows, idx]
                missing = ~np.isfinite(value)
                go_left = np.where(missing, t['default_left'][node], value < t['threshold'][node])
                nxt = np.where(go_left, t['left'][node], t['right'][node])
                node = np.where(active, nxt, node)
                active = t['left'][node] != -1
            out += t['weight'][node]
        return out


class RankerWithAux(Scorer):
    """Deployed ranking model: numpy tree ensemble for the ranking margin plus a scikit-learn head for the two log ratios."""
    score_kind = 'margin'

    def __init__(self, ensemble, aux, name='xgb_ranker'):
        self.ensemble = ensemble
        self.aux = aux
        self.name = name
        self.components = (aux,)

    def predict(self, X):
        X = np.asarray(X, float)
        return np.column_stack([self.ensemble.predict_margin(X), self.aux.predict(X)])
