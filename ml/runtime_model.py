"""Portable, NumPy-only inference for the deployed ranking and forecast trees."""
import json
from pathlib import Path

import numpy as np


FORMAT = 'before-the-list-tree-model'
SCHEMA_VERSION = 1


def _array(tree, name, dtype, length=None):
    try:
        raw = np.asarray(tree[name])
        kind = np.dtype(dtype).kind
        if raw.dtype.kind not in ('iu' if kind in 'iu' else 'b' if kind == 'b' else 'iuf'):
            raise ValueError('Unexpected array element type')
        result = raw.astype(dtype)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid model tree array: {name}') from exc
    if result.ndim != 1 or length is not None and len(result) != length:
        raise ValueError(f'Invalid model tree array shape: {name}')
    return result


def _tree(data, features, outputs=None):
    if not isinstance(data, dict):
        raise ValueError('Invalid model tree')
    left = _array(data, 'left', np.int64)
    n = len(left)
    if not n:
        raise ValueError('Empty model tree')
    right = _array(data, 'right', np.int64, n)
    feature = _array(data, 'feature', np.int64, n)
    threshold = _array(data, 'threshold', np.float64, n)
    missing = _array(data, 'missing_left', np.bool_, n)
    leaf = left == -1
    branch = ~leaf
    indices = np.arange(n)
    # Children are allocated after their parent in both exported tree formats.
    # This also rejects cycles before an evaluator can loop indefinitely.
    if (np.any(right[leaf] != -1) or np.any(left[branch] <= indices[branch])
            or np.any(right[branch] <= indices[branch]) or np.any(left[branch] >= n)
            or np.any(right[branch] >= n) or np.any(feature[branch] < 0)
            or np.any(feature[branch] >= features) or not np.isfinite(threshold).all()):
        raise ValueError('Invalid model tree topology or split')
    try:
        value = np.asarray(data['value'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid model tree value') from exc
    if value.shape != ((n, outputs) if outputs else (n,)) or not np.isfinite(value).all():
        raise ValueError('Invalid model tree value shape or number')
    return dict(left=left, right=right, feature=feature, threshold=threshold,
                missing_left=missing, value=value)


def _leaves(tree, values, inclusive):
    nodes = np.zeros(len(values), dtype=np.int64)
    active = np.flatnonzero(tree['left'][nodes] != -1)
    while len(active):
        current = nodes[active]
        observed = values[active, tree['feature'][current]]
        threshold = tree['threshold'][current]
        # sklearn promotes float32 inputs to double for its double thresholds;
        # the exported ranker stores its thresholds as float32.
        goes_left = observed <= threshold if inclusive else observed < threshold
        goes_left = np.where(np.isnan(observed), tree['missing_left'][current], goes_left)
        nodes[active] = np.where(goes_left, tree['left'][current], tree['right'][current])
        active = active[tree['left'][nodes[active]] != -1]
    return nodes


class RuntimeModel:
    score_kind = 'margin'

    def __init__(self, ranker, auxiliary, features):
        if (not isinstance(ranker, dict) or not isinstance(auxiliary, dict)
                or ranker.get('kind') != 'xgboost_margin' or auxiliary.get('kind') != 'extra_trees_mean'
                or auxiliary.get('outputs') != 2):
            raise ValueError('Unsupported portable model heads')
        self.n_features_in_ = features
        try:
            self.base_score = float(ranker['base_score'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid ranking base score') from exc
        if not np.isfinite(self.base_score):
            raise ValueError('Invalid ranking base score')
        self.ranking = self._trees(ranker, features)
        self.auxiliary = self._trees(auxiliary, features, outputs=2)
        # Original NumpyTreeEnsemble compares two float32 arrays.
        for tree in self.ranking:
            tree['threshold'] = tree['threshold'].astype(np.float32)
            if not np.isfinite(tree['threshold']).all():
                raise ValueError('Ranking threshold exceeds float32 range')

    @staticmethod
    def _trees(head, features, outputs=None):
        raw = head.get('trees')
        if not isinstance(raw, list) or not raw:
            raise ValueError('Model head has no trees')
        return [_tree(tree, features, outputs) for tree in raw]

    def set_params(self, **params):
        # Serving requests n_jobs=1; these NumPy evaluators are already serial.
        if any(key != 'n_jobs' for key in params):
            raise ValueError('Unsupported portable inference parameter')
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != self.n_features_in_ or not len(values)
                or np.isinf(values).any()):
            raise ValueError('Expected a nonempty feature matrix without infinite values')
        score = np.full(len(values), self.base_score, dtype=np.float64)
        auxiliary = np.zeros((len(values), 2), dtype=np.float64)
        for tree in self.ranking:
            score += tree['value'][_leaves(tree, values, inclusive=False)]
        for tree in self.auxiliary:
            auxiliary += tree['value'][_leaves(tree, values, inclusive=True)]
        auxiliary /= len(self.auxiliary)
        return np.column_stack((score, auxiliary))


def load_json_pack(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(document, dict) or document.get('format') != FORMAT
            or document.get('schema_version') != SCHEMA_VERSION):
        raise ValueError('Unsupported portable model format or schema version')
    metadata = document.get('metadata')
    if not isinstance(metadata, dict) or 'model' in metadata:
        raise ValueError('Invalid portable model metadata')
    features = metadata.get('features')
    if (not isinstance(features, list) or not features
            or any(not isinstance(name, str) or not name for name in features)
            or len(features) != len(set(features))
            or metadata.get('score_kind') != 'margin'
            or not isinstance(metadata.get('version'), str) or not metadata['version']
            or not isinstance(metadata.get('trained_through'), str) or not metadata['trained_through']):
        raise ValueError('Invalid portable model features, score kind or version')
    model = RuntimeModel(document.get('ranking'), document.get('auxiliary'), len(features))
    return {**metadata, 'model': model}


def load_runtime_pack(root):
    """Prefer the portable asset, retaining the untouched development pickle fallback."""
    root = Path(root)
    portable = root / 'models/runtime-model.json'
    if portable.exists():
        return load_json_pack(portable)
    import joblib
    return joblib.load(root / 'models/model.joblib')
