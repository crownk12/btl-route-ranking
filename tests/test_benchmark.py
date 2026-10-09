"""Benchmark v12 boundary checks: point-in-time features, cross-candidate deviations, metrics, and split integrity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]));import btl_paths  # noqa: E402,F401
import json
import unittest

import numpy as np

from common import Panel, FEATURES, mi
from benchmark.features import FeatureBuilder, ALL_FEATURES, NEW_CANDIDATES, GROUPS, group_of, _change_point, _coupling
from benchmark.metrics import ranking_metrics, evaluate

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = FeatureBuilder()

    def test_every_candidate_has_a_group_and_a_definition(self):
        from benchmark.run import feature_definitions
        defs = feature_definitions()
        for name in ALL_FEATURES:
            self.assertIsNotNone(group_of(name), name)
            self.assertIn(name, defs, name)
        self.assertEqual(len(ALL_FEATURES), len(set(ALL_FEATURES)))
        self.assertEqual(ALL_FEATURES[:12], FEATURES)

    def test_row_is_point_in_time(self):
        """Rows published after the basis must not change any feature: rebuild the panel with those rows removed."""
        key = ('RUS', '848210', 'IND'); basis = 430
        before, _ = self.builder.row(key, basis, 6)
        p = self.builder.panel
        original = p.rows[key]
        try:
            p.rows[key] = [r for r in original if r['available_month'] <= basis]
            after, _ = self.builder.row(key, basis, 6)
        finally:
            p.rows[key] = original
        for name in ALL_FEATURES:
            if name in ('mirror_gap_excess', 'neighbor_deviation', 'similar_item_deviation'):
                continue
            self.assertAlmostEqual(before[name], after[name], places=9, msg=name)

    def test_policy_and_designation_features_respect_announce_dates(self):
        key = ('RUS', '848210', 'KAZ')
        early, _ = self.builder.row(key, mi('2021-12'), 3)
        late, _ = self.builder.row(key, mi('2024-12'), 3)
        self.assertEqual(early['control_actions_to_date'], 0.0)
        self.assertEqual(early['months_since_control'], 120.0)
        self.assertGreaterEqual(late['control_actions_to_date'], 1.0)
        self.assertLess(late['months_since_control'], 120.0)
        self.assertEqual(early['transit_designations_to_date'], 0.0)
        self.assertGreaterEqual(late['transit_designations_to_date'], early['transit_designations_to_date'])
        pre_eaeu, _ = self.builder.row(key, mi('2014-06'), 3)
        self.assertEqual(pre_eaeu['economic_bloc_link'], 0.0)
        self.assertEqual(late['economic_bloc_link'], 1.0)
        self.assertEqual(late['geographic_adjacency'], 1.0)
        self.assertEqual(late['hs_chapter'], 84.0)

    def test_cross_features_are_deviations_from_peers(self):
        rows = [{'mirror_gap_level': v, 'coupled_growth': v} for v in [1.0, 2.0, 3.0, 10.0]]
        keys = [('RUS', '848210', c) for c in ['A', 'B', 'C', 'D']]
        out = FeatureBuilder.add_cross_features(rows, keys, [400] * 4, [3] * 4)
        self.assertAlmostEqual(out[3]['neighbor_deviation'], 10.0 - 2.0)
        self.assertAlmostEqual(out[0]['mirror_gap_excess'], 1.0 - 3.0)
        self.assertEqual(out[0]['similar_item_deviation'], 0.0)  # only one item per candidate here

    def test_change_point_and_coupling_helpers(self):
        flat = np.zeros(24)
        self.assertEqual(_change_point(flat)[0], 0.0)
        shifted = np.r_[np.zeros(12), np.ones(12) * 2.0] + np.random.default_rng(0).normal(0, 0.05, 24)
        detected, magnitude, slope_change, since = _change_point(shifted)
        self.assertEqual(detected, 1.0); self.assertGreater(magnitude, 1.5); self.assertLessEqual(since, 13)
        a = np.cumsum(np.random.default_rng(1).normal(0, 1, 13)); b = np.r_[0, a[:-1]]
        self.assertGreater(_coupling(a, b), 0.8)


class MetricTests(unittest.TestCase):
    def test_ranking_metrics_on_a_perfect_and_a_reversed_ranking(self):
        y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0], float); g = np.array(['q'] * 9)
        perfect = np.array([9, 1, 2, 3, 4, 8, 5, 6, 7], float)
        m = ranking_metrics(y, perfect, g)
        self.assertAlmostEqual(m['ndcg_at_5'], 1.0); self.assertAlmostEqual(m['recall_at_5'], 1.0); self.assertAlmostEqual(m['mrr'], 1.0); self.assertEqual(m['hit_rate_at_5'], 1.0)
        reversed_ = -perfect
        m2 = ranking_metrics(y, reversed_, g)
        self.assertEqual(m2['recall_at_5'], 0.0); self.assertEqual(m2['hit_rate_at_5'], 0.0)
        out = evaluate(np.column_stack([y, np.zeros(9), np.zeros(9)]), np.column_stack([perfect, np.full(9, np.nan), np.full(9, np.nan)]), g, probabilistic=False)
        self.assertIsNone(out['brier']); self.assertIsNone(out['log_ratio_mae'])


class SplitManifestTests(unittest.TestCase):
    def test_manifest_has_no_query_overlap_and_time_order(self):
        for folder in ('model-benchmark-v12', 'model-benchmark-v14'):
            path = ROOT / 'artifacts' / folder / 'split-manifest.json'
            if not path.exists():
                self.skipTest('benchmark not built')
            m = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(all(v == 0 for v in m['query_overlap'].values()))
            self.assertLess(m['splits']['train']['basis_range'][1], m['splits']['validation']['basis_range'][0])
            self.assertLess(m['splits']['validation']['basis_range'][1], m['splits']['test']['basis_range'][0])

    def test_v14_scores_the_same_queries_as_v13(self):
        """Benchmark v14 changed the features and the training rows, not the validation and test queries."""
        old, new = (ROOT / 'artifacts' / f / 'split-manifest.json' for f in ('model-benchmark-v13-expanded', 'model-benchmark-v14'))
        if not (old.exists() and new.exists()):
            self.skipTest('benchmark not built')
        old, new = (json.loads(p.read_text(encoding='utf-8'))['splits'] for p in (old, new))
        for split in ('validation', 'test'):
            self.assertEqual(old[split]['query_ids_sha256'], new[split]['query_ids_sha256'])
            self.assertEqual(old[split]['rows'], new[split]['rows'])



class FeatureEquivalenceTests(unittest.TestCase):
    def test_builder_reproduces_the_cached_benchmark_matrix(self):
        """The v1 builder must reproduce the cached benchmark matrix (the 25-feature packs kept in models/archive were trained on it)."""
        from adapters.contracts import panel_set  # the deployed model was trained on the dataset of the active item set
        cache = ROOT / ('artifacts/model-benchmark-v13-expanded' if panel_set() == 'expanded' else 'artifacts/model-benchmark-v12') / 'cache/dataset.npz'
        if not cache.exists():
            self.skipTest('benchmark cache not built')
        z = np.load(cache, allow_pickle=True)
        names = list(z['features']); rng = np.random.default_rng(0)
        builder = FeatureBuilder()
        picks = rng.choice(len(z['G']), size=40, replace=False)
        # cross-candidate features need the peers, so rebuild the whole query (and the other items) for each pick
        for i in picks:
            target, hs, basis, h = z['G'][i].split(':'); basis = int(basis); h = int(h)
            from common import ITEMS, candidates_for
            keys = [(target, item, c) for item in ITEMS for c in candidates_for(target)]
            rows = [builder.row(k, basis, h)[0] for k in keys]
            rows = FeatureBuilder.add_cross_features(rows, keys, [basis] * len(keys), [h] * len(keys))
            row = rows[keys.index((target, hs, str(z['C'][i])))]
            expected = z['X'][i]
            for j, name in enumerate(names):
                if np.isfinite(expected[j]):
                    self.assertAlmostEqual(row[name], float(expected[j]), places=6, msg=f'{name} @ {z["G"][i]}')


class FeatureV2Tests(unittest.TestCase):
    """Feature set v2 (benchmark v14): the block the training matrix and the server share."""
    @classmethod
    def setUpClass(cls):
        from benchmark.features_v2 import FeatureBuilderV2
        cls.builder = FeatureBuilderV2()

    def test_names_groups_and_v1_columns(self):
        from benchmark.features_v2 import ALL_FEATURES_V2, GROUPS_V2, group_of_v2, layout
        self.assertEqual(self.builder.names, ALL_FEATURES_V2)
        self.assertEqual(ALL_FEATURES_V2[:len(ALL_FEATURES)], ALL_FEATURES)
        self.assertEqual(len(ALL_FEATURES_V2), len(set(ALL_FEATURES_V2)))
        for name in ALL_FEATURES_V2:
            self.assertIsNotNone(group_of_v2(name), name)
        grouped = [n for names in GROUPS_V2.values() for n in names]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertTrue(set(ALL_FEATURES_V2) < set(layout(True)['names']))

    def test_block_is_point_in_time(self):
        """Rows published after the basis month must not change any value of the block, peers included."""
        target, basis = 'RUS', 430
        before = self.builder.block(target, basis, [3, 12, 75])['X']
        p = self.builder.panel; original = p.rows
        try:
            p.rows = {k: [r for r in rows if r['available_month'] <= basis] for k, rows in original.items()}
            after = self.builder.block(target, basis, [3, 12, 75])['X']
        finally:
            p.rows = original
        np.testing.assert_array_equal(before, after)

    def test_block_keeps_the_v1_values(self):
        """The v1 columns of a block are the values FeatureBuilder.row and add_cross_features give."""
        from common import ITEMS, candidates_for
        target, basis, h = 'IRN', 300, 6
        block = self.builder.block(target, basis, [h])
        v1 = FeatureBuilder(self.builder.panel)
        keys = [(target, hs, c) for hs in ITEMS for c in candidates_for(target)]
        rows = FeatureBuilder.add_cross_features([v1.row(k, basis, h)[0] for k in keys], keys, [basis] * len(keys), [h] * len(keys))
        for key, row in zip(keys, rows):
            got = block['X'][block['items'].index(key[1]), block['candidates'].index(key[2]), 0]
            for j, name in enumerate(ALL_FEATURES):
                self.assertAlmostEqual(got[j], row[name], places=9, msg=f'{name} @ {key}')

    def test_seasonal_position_follows_a_known_cycle(self):
        """A pure 12-month cycle: the seasonal change to the forecast window is recovered and the deseasonalised slope is flat."""
        from benchmark.features_v2 import leg_features, STATIC_LEG, DYNAMIC_LEG
        months = np.arange(100, 160); basis = 160
        cycle = lambda m: 0.3 * np.sin(np.asarray(m) * np.pi / 6 + 1.0)
        raw = np.exp(10 + cycle(months))
        static, dynamic = leg_features(raw, months, basis, [3, 6])
        s = dict(zip(STATIC_LEG, static)); d3 = dict(zip(DYNAMIC_LEG, dynamic[0]))
        self.assertAlmostEqual(s['amp'], 0.3, places=2)
        self.assertAlmostEqual(s['slope6'], 0.0, places=2)
        future = basis + 3 + np.arange(3)
        self.assertAlmostEqual(d3['seas_delta'], cycle(future).mean() - cycle(months[-3:]).mean(), places=2)
        self.assertAlmostEqual(d3['snaive_mean'], np.log(np.exp(10 + cycle(future)).mean() / raw[-3:].mean()), places=6)

    def test_others_median_deviation(self):
        from benchmark.features_v2 import _others_median_deviation
        out = _others_median_deviation(np.array([[1.0, 2.0, 3.0, 10.0]]), 1)
        np.testing.assert_allclose(out[0], [1 - 3, 2 - 3, 3 - 2, 10 - 2])
        self.assertEqual(_others_median_deviation(np.array([[5.0]]), 1)[0, 0], 0.0)

    def test_cached_matrix_matches_the_block(self):
        """The serving-time block must give the same values the deployed model was trained on."""
        cache = ROOT / 'artifacts/model-benchmark-v14/cache/dataset.npz'
        if not cache.exists():
            self.skipTest('benchmark v14 cache not built')
        z = np.load(cache, allow_pickle=True)
        columns = [self.builder.names.index(n) for n in list(z['features'])]
        G, C, X = z['G'], z['C'], z['X']
        rng = np.random.default_rng(0)
        for i in rng.choice(len(G), size=12, replace=False):
            target, hs, basis, h = G[i].split(':')
            block = self.builder.block(target, int(basis), [int(h)])
            got = block['X'][block['items'].index(hs), block['candidates'].index(str(C[i])), 0, columns]
            np.testing.assert_allclose(got.astype(np.float32), X[i], rtol=0, atol=1e-6, err_msg=str(G[i]))


class CommonMovementTests(unittest.TestCase):
    """Export leg without the target-wide common movement (FeatureBuilderV2(common=True), export-surge track v2)."""
    @classmethod
    def setUpClass(cls):
        from benchmark.features_v2 import FeatureBuilderV2
        cls.plain = FeatureBuilderV2()
        cls.builder = FeatureBuilderV2(cls.plain.panel, common=True)

    def test_columns_are_appended_and_the_others_do_not_move(self):
        from benchmark.features_v2 import ALL_FEATURES_V2, COMMON_NAMES, group_of_v2
        self.assertEqual(self.builder.names, ALL_FEATURES_V2 + COMMON_NAMES)
        self.assertEqual(len(COMMON_NAMES), 35)
        for name in COMMON_NAMES:
            self.assertEqual(group_of_v2(name), 'common_movement_removed')
        with_common = self.builder.block('PRK', 352, [1, 3])['X']
        np.testing.assert_array_equal(with_common[..., :len(ALL_FEATURES_V2)], self.plain.block('PRK', 352, [1, 3])['X'])
        self.assertTrue(np.abs(with_common[..., len(ALL_FEATURES_V2):]).sum() > 0)

    def test_block_is_point_in_time_and_independent_of_the_items_asked_for(self):
        from benchmark.features_v2 import ALL_FEATURES_V2
        target, basis = 'RUS', 430
        block = self.builder.block(target, basis, [1, 3]); before = block['X']
        p = self.builder.panel; original = p.rows
        try:
            p.rows = {k: [r for r in rows if r['available_month'] <= basis] for k, rows in original.items()}
            after = self.builder.block(target, basis, [1, 3])['X']
        finally:
            p.rows = original
        np.testing.assert_array_equal(before, after)
        # the common movement comes from the whole active item set even when one item is requested
        one = self.builder.block(target, basis, [1, 3], items=block['items'][4:5])['X']
        np.testing.assert_array_equal(one[0, ..., len(ALL_FEATURES_V2):], before[4, ..., len(ALL_FEATURES_V2):])

    def test_a_shared_ramp_is_removed_with_each_route_s_own_loading(self):
        from benchmark.features_v2 import common_movement, common_features, COMMON_NAMES, HISTORY
        basis = 200; months = np.arange(basis - HISTORY + 1, basis + 1)
        ramp = np.clip((months - (basis - 30)) / 10.0, 0, 1)
        rng = np.random.default_rng(0); loads = np.linspace(0.3, 0.9, 12); routes = []
        for load in loads:
            out = np.exp(8 + load * ramp + rng.normal(0, 0.01, HISTORY))
            routes.append((np.column_stack([np.ones(HISTORY), out]), months, [np.zeros(2), np.zeros(2)]))
        common, centred = common_movement(routes, basis)
        self.assertEqual(common.shape, (HISTORY,)); self.assertEqual(centred.shape, (len(routes), HISTORY))
        scale = np.median(loads)  # the common movement carries the median route's ramp
        for r in (0, 5, 11):
            a, _, coefs = routes[r]
            f = dict(zip(COMMON_NAMES, common_features(a[:, 1], months, basis, [1], coefs[1], common, centred[r])[0]))
            self.assertAlmostEqual(f['fac_load_out'] * scale, loads[r], delta=0.05)
            self.assertAlmostEqual(f['adj_dev48_out'], 0.0, delta=0.03)   # the raw series stands 0.1-0.5 above its 48-month mean
            self.assertAlmostEqual(f['adj_height_out'], 0.0, delta=0.03)
        self.assertEqual(np.abs(common_features(None, None, basis, [1, 3], None, common, centred[0])).sum(), 0.0)

    def test_cached_export_matrix_matches_the_block(self):
        """The serving-time block must give the same values the export-surge model was trained on."""
        cache = ROOT / 'artifacts/model-benchmark-v14/cache/dataset-export-v2.npz'
        if not cache.exists():
            self.skipTest('export-surge v2 cache not built')
        z = np.load(cache, allow_pickle=True)
        G, C, X = z['G'], z['C'], z['X']
        columns = [self.builder.names.index(n) for n in list(z['features'])]
        rng = np.random.default_rng(1)
        for i in rng.choice(len(G), size=10, replace=False):
            target, hs, basis, h = G[i].split(':')
            block = self.builder.block(target, int(basis), [int(h)])
            got = block['X'][block['items'].index(hs), block['candidates'].index(str(C[i])), 0, columns]
            np.testing.assert_allclose(got.astype(np.float32), X[i], rtol=0, atol=1e-6, err_msg=str(G[i]))


class MetricV14Tests(unittest.TestCase):
    def test_queries_without_a_positive_only_enter_the_all_query_mean(self):
        y = np.array([1, 0, 0, 0, 0, 0] + [0] * 6, float); g = np.array(['a'] * 6 + ['b'] * 6)
        score = np.array([9, 1, 2, 3, 4, 5] + [1, 2, 3, 4, 5, 6], float)
        m = ranking_metrics(y, score, g)
        self.assertAlmostEqual(m['ndcg_at_5_pos'], 1.0); self.assertAlmostEqual(m['ndcg_at_5'], 0.5)
        self.assertEqual((m['queries'], m['queries_with_positive']), (2, 1))

    def test_ndcg_matches_sklearn_with_ties(self):
        from sklearn.metrics import ndcg_score
        from benchmark.metrics import _ndcg
        rng = np.random.default_rng(3)
        for _ in range(50):
            y = (rng.random(9) < 0.4).astype(float); s = np.round(rng.normal(size=9), 1)  # rounding makes ties
            if y.sum():
                self.assertAlmostEqual(_ndcg(y, s, 5), ndcg_score([y], [s], k=5), places=12)


class RealBenchmarkTests(unittest.TestCase):
    def test_query_scores_use_each_metric_on_the_queries_it_fits(self):
        from benchmark.real_bench import query_scores
        y = np.array([1, 0] + [1, 0, 0, 0, 0] + [0, 0, 0], float)
        g = np.array(['two'] * 2 + ['five'] * 5 + ['none'] * 3)
        score = np.array([2, 1] + [5, 4, 3, 2, 1] + [3, 2, 1], float)
        result, pairs = query_scores(y, score, g)
        self.assertEqual((result['pair_queries'], result['ndcg_at_3_queries'], result['ndcg_at_5_pos_queries']), (2, 1, 1))
        self.assertAlmostEqual(result['pairs_ordered_correctly'], 1.0); self.assertAlmostEqual(result['ndcg_at_3'], 1.0)
        worst, _ = query_scores(y, -score, g)
        self.assertAlmostEqual(worst['pairs_ordered_correctly'], 0.0)

    def test_collected_import_rows_keep_the_contract(self):
        """Candidate-reported imports: known reporters and partners, positive values, one row per reporter, partner, item and month."""
        from common import CANDIDATES, ITEMS_ALL
        path = ROOT / 'data/observed-imports.json'
        if not path.exists():
            self.skipTest('imports not collected')
        rows = json.loads(path.read_text(encoding='utf-8'))['rows']
        self.assertGreater(len(rows), 1000)
        self.assertTrue(all(r['reporter'] in CANDIDATES and r['partner'] in ('WLD', 'CHN') and r['reporter'] != r['partner'] for r in rows))
        self.assertTrue(all(r['hs'] in ITEMS_ALL and r['usd'] >= 0 and r['available_month'] >= r['month'] for r in rows))
        keys = [(r['reporter'], r['partner'], r['hs'], r['month'], r['edition']) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))


    def test_wider_pool_rows_keep_the_contract(self):
        """Evaluation-only reporters: none of them is a candidate of the application, and both legs carry known partners."""
        from common import CANDIDATES, TARGETS, ITEMS_ALL
        path = ROOT / 'data/observed-wider.json'
        if not path.exists():
            self.skipTest('wider pool not collected')
        d = json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(set(d['reporters']) & set(CANDIDATES + TARGETS))
        for leg, partners in (('exports', set(TARGETS)), ('imports', {'WLD', 'CHN'})):
            rows = d[leg]['rows']
            self.assertTrue(all(r['reporter'] in d['reporters'] and r['partner'] in partners and r['hs'] in ITEMS_ALL and r['usd'] >= 0 for r in rows), leg)


class ReplayTests(unittest.TestCase):
    def test_replay_reproduces_the_panel_and_a_replica_does_not(self):
        """The ceiling diagnostic rests on the replay being the generator: sampled months must equal the database cell for cell."""
        from benchmark.replay import replay, verify
        rows, latent = replay('', latents=True)
        self.assertGreater(verify(rows, every=29), 0)

        def unexplained(key):  # log imports minus every stored component: the target-wide shock, shared by the target's routes
            z = latent[key]
            return np.log(z['final'][0]) - (z['path'][0] + z['seasonal'][0] + z['noise'][0] + sum(p['legs'][0] for p in z['pulses']) + sum(e['legs'][0] for e in z['events']))
        key = ('RUS', '848210', 'KAZ')
        np.testing.assert_allclose(unexplained(key), unexplained(('RUS', '854231', 'TUR')), atol=1e-9)
        end = 150
        replica, _ = replay('replica-test', end)
        self.assertNotEqual(replica[key][200]['inbound'], rows[key][200]['inbound'])
        self.assertEqual(max(r['month'] for r in replica[key]), end)


if __name__ == '__main__':
    unittest.main()
