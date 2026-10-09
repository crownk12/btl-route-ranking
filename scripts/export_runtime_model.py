"""Export the existing model to portable JSON without modifying the source pickle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
from runtime_model import FORMAT, SCHEMA_VERSION, load_json_pack  # noqa: E402


def export_model(output, source=None):
    import joblib
    import numpy as np
    source = Path(source or ROOT / 'models/model.joblib').resolve()
    output = Path(output).resolve()
    if output == source or output.suffix.lower() != '.json':
        raise ValueError('Export must target a separate .json file')
    pack = joblib.load(source)
    model = pack['model']
    if (type(model).__name__ != 'RankerWithAux' or type(model.ensemble).__name__ != 'NumpyTreeEnsemble'
            or type(model.aux).__name__ != 'ExtraTreesRegressor' or model.aux.n_outputs_ != 2
            or model.aux.n_features_in_ != len(pack['features'])):
        raise ValueError('Only the deployed ranker and two-output ExtraTrees model are supported')
    rank_trees = [dict(left=t['left'].tolist(), right=t['right'].tolist(), feature=t['index'].tolist(),
                       threshold=t['threshold'].tolist(), missing_left=t['default_left'].tolist(),
                       value=t['weight'].tolist()) for t in model.ensemble.trees]
    auxiliary = []
    for estimator in model.aux.estimators_:
        tree = estimator.tree_
        auxiliary.append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                              feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                              missing_left=tree.missing_go_to_left.astype(bool).tolist(),
                              value=tree.value[:, :, 0].tolist()))
    document = dict(format=FORMAT, schema_version=SCHEMA_VERSION,
                    metadata={key: value for key, value in pack.items() if key != 'model'},
                    ranking=dict(kind='xgboost_margin', base_score=model.ensemble.base_score, trees=rank_trees),
                    auxiliary=dict(kind='extra_trees_mean', outputs=2, trees=auxiliary))
    payload = json.dumps(document, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.building')
    try:
        temporary.write_text(payload, encoding='utf-8', newline='\n')
        exported = load_json_pack(temporary)['model']
        model.set_params(n_jobs=1)
        probe = np.zeros((3, len(pack['features'])))
        error = float(np.max(np.abs(model.predict(probe) - exported.predict(probe))))
        if error > 1e-10:
            raise ValueError(f'Export prediction mismatch: {error}')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(output), bytes=output.stat().st_size, version=pack['version'],
                ranking_trees=len(rank_trees), auxiliary_trees=len(auxiliary), zero_probe_max_abs_error=error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'models/model.joblib')
    args = parser.parse_args()
    print(json.dumps(export_model(args.output, args.source), ensure_ascii=False))
