"""Portable predictions must match the unchanged deployed model and split semantics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import btl_paths  # noqa: E402,F401
import joblib
import numpy as np
from runtime_model import FORMAT, SCHEMA_VERSION, load_json_pack, load_runtime_pack

spec = importlib.util.spec_from_file_location('export_runtime_model', ROOT / 'scripts/export_runtime_model.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def tiny_document():
    tree = dict(left=[1, -1, -1], right=[2, -1, -1], feature=[0, -2, -2],
                threshold=[1.0, -2.0, -2.0], missing_left=[True, False, False], value=[0.0, 2.0, 5.0])
    aux = {**tree, 'missing_left': [False, False, False], 'value': [[0.0, 0.0], [3.0, 4.0], [6.0, 7.0]]}
    return dict(format=FORMAT, schema_version=SCHEMA_VERSION,
                metadata=dict(version='test', features=['x'], trained_through='2026-09', score_kind='margin'),
                ranking=dict(kind='xgboost_margin', base_score=0.5, trees=[tree]),
                auxiliary=dict(kind='extra_trees_mean', outputs=2, trees=[aux]))


class PortableSchemaTests(unittest.TestCase):
    def load(self, document):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            return load_json_pack(path)

    def test_float32_boundary_comparisons_and_missing_routes(self):
        model = self.load(tiny_document())['model']
        values = [[np.nextafter(np.float32(1), np.float32(0))], [1],
                  [np.nextafter(np.float32(1), np.float32(2))], [np.nan],
                  [1 + 1e-10]]  # rounds to one, exactly as sklearn's input validation does
        np.testing.assert_array_equal(model.predict(values),
                                      [[2.5, 3, 4], [5.5, 3, 4], [5.5, 6, 7], [2.5, 6, 7], [5.5, 3, 4]])

    def test_rejects_invalid_schema_tree_and_metadata(self):
        variants = []
        value = tiny_document(); value['schema_version'] = 99; variants.append(value)
        value = tiny_document(); value['metadata']['features'] = [{}]; variants.append(value)
        value = tiny_document(); value['metadata']['score_kind'] = 'probability'; variants.append(value)
        value = tiny_document(); value['ranking']['trees'][0]['left'][0] = 0; variants.append(value)
        value = tiny_document(); value['ranking']['trees'][0]['left'][0] = 1.5; variants.append(value)
        value = tiny_document(); value['ranking']['trees'][0]['feature'][0] = 7; variants.append(value)
        value = tiny_document(); value['auxiliary']['trees'][0]['value'] = [[0, 1]]; variants.append(value)
        for document in variants:
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.load(document)

    def test_invalid_preferred_json_does_not_silently_load_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'models').mkdir()
            (root / 'models/runtime-model.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_runtime_pack(root)


class DeployedModelEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / 'runtime-model.json'
        source = ROOT / 'models/model.joblib'
        cls.source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        exporter.export_model(cls.output)
        cls.original_pack = joblib.load(source)
        cls.original = cls.original_pack['model'].set_params(n_jobs=1)
        cls.portable_pack = load_json_pack(cls.output)
        cls.portable = cls.portable_pack['model']
        cls.errors = []

    @classmethod
    def tearDownClass(cls):
        print('PORTABLE_MODEL_EQUIVALENCE ' + json.dumps(cls.errors, sort_keys=True), flush=True)
        cls.temporary.cleanup()

    def compare(self, name, values, groups=None):
        expected = self.original.predict(values)
        actual = self.portable.predict(values)
        errors = np.max(np.abs(expected - actual), axis=0)
        self.errors.append(dict(case=name, rows=len(values), max_abs_error=errors.tolist()))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        if groups is not None:
            for begin, end in groups:
                np.testing.assert_array_equal(np.argsort(-actual[begin:end, 0], kind='stable')[:5],
                                              np.argsort(-expected[begin:end, 0], kind='stable')[:5])

    def test_metadata_source_and_original_fallback_are_preserved(self):
        self.assertEqual({k: v for k, v in self.original_pack.items() if k != 'model'},
                         {k: v for k, v in self.portable_pack.items() if k != 'model'})
        self.assertEqual(hashlib.sha256((ROOT / 'models/model.joblib').read_bytes()).hexdigest(), self.source_hash)
        self.assertEqual(type(load_runtime_pack(ROOT)['model']), type(self.original))
        with self.assertRaises(ValueError):
            exporter.export_model(ROOT / 'models/model.joblib')

    def test_random_zero_and_nan_matrices(self):
        rng = np.random.default_rng(7264)
        width = len(self.original_pack['features'])
        values = rng.normal(0, 3, (3000, width))
        values[:, 10] = rng.uniform(0, 120, len(values))
        values[:, 11] = rng.uniform(0, 75, len(values))
        self.compare('random', values)
        self.compare('zeros', np.zeros((64, width)))
        values = rng.normal(0, 2, (256, width))
        values[rng.random(values.shape) < 0.2] = np.nan
        values[0] = np.nan
        self.compare('nan_routes', values)

    def test_actual_tree_split_boundaries(self):
        width = len(self.original_pack['features'])
        rows = []
        for tree in self.portable.ranking + self.portable.auxiliary:
            if tree['left'][0] == -1:
                continue
            threshold = np.float32(tree['threshold'][0]); feature = tree['feature'][0]
            for value in (np.nextafter(threshold, np.float32(-np.inf)), threshold,
                          np.nextafter(threshold, np.float32(np.inf))):
                row = np.zeros(width); row[feature] = value; rows.append(row)
        self.compare('tree_root_float32_boundaries', np.array(rows))

    def test_all_targets_items_and_top_five_on_actual_features(self):
        from common import Panel, TARGETS, ITEMS_ALL, candidates_for
        from benchmark.features import FeatureBuilder, ALL_FEATURES
        panel = Panel()
        features = self.original_pack['features']; values = []; groups = []
        v2 = not set(features) <= set(ALL_FEATURES)  # feature set v2 is built per target x basis block
        if v2:
            from benchmark.features_v2 import FeatureBuilderV2
            blocks = FeatureBuilderV2(panel); columns = [blocks.names.index(name) for name in features]
        else:
            builder = FeatureBuilder(panel)
        for basis in (60, min(384, panel.end), panel.end):
            for target in TARGETS:
                if v2:
                    block = blocks.block(target, basis, [0, 12, 75], ITEMS_ALL)['X']
                    for k in range(3):
                        for i in range(block.shape[0]):
                            groups.append((len(values), len(values) + block.shape[1]))
                            values.extend(block[i, :, k, columns].T.tolist())
                    continue
                keys = [(target, hs, candidate) for hs in ITEMS_ALL for candidate in candidates_for(target)]
                rows = [builder.row(key, basis, 0)[0] for key in keys]
                rows = builder.add_cross_features(rows, keys, [basis] * len(keys), [0] * len(keys))
                for horizon in (0, 12, 75):
                    start = len(values)
                    for row in rows:
                        row = {**row, 'horizon': horizon}
                        values.append([row[name] for name in features])
                    count = len(candidates_for(target))
                    groups.extend((i, i + count) for i in range(start, len(values), count))
        self.compare('four_targets_seventeen_items_three_bases_three_horizons', np.asarray(values), groups)

    def test_portable_load_and_predict_without_sklearn_joblib_or_scipy(self):
        script = '''
import importlib.abc, json, pathlib, sys
class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'joblib', 'scipy'}:
            raise AssertionError('Unexpected runtime dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingDependencies())
sys.path.insert(0, sys.argv[1])
from runtime_model import load_json_pack
pack = load_json_pack(pathlib.Path(sys.argv[2]))
print(json.dumps(pack['model'].predict([[0] * len(pack['features'])]).tolist()))
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(ROOT / 'ml'), str(self.output)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(np.asarray(json.loads(result.stdout)).shape, (1, 3))


if __name__ == '__main__':
    unittest.main()
