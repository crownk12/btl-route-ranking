"""Ranking and classification metrics per query for the benchmark.

Primary metric since benchmark v14: NDCG@5 over the queries that hold at least one positive (`ndcg_at_5_pos`).
A query without a positive has no correct order, and sklearn scores it 0, so the all-query mean (`ndcg_at_5`,
the v12/v13 primary) is capped by the share of queries with a positive (0.62 to 0.74 on this panel) whatever
the model does. Both are reported; `ndcg_at_5` keeps its old meaning so earlier records stay comparable.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score


def per_query_indices(groups):
    order = np.argsort(groups, kind='stable')
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    return [order[a:b] for a, b in zip(np.r_[0, boundaries], np.r_[boundaries, len(order)])]


def _ndcg(yy, ss, k):
    """NDCG@k with linear gains for one query, tied scores sharing their gain (same value as sklearn.metrics.ndcg_score);
    0 when the query has no relevant row (sklearn's convention)."""
    discount = 1.0 / np.log2(np.arange(2, len(yy) + 2)); discount[k:] = 0.0
    ideal = float(np.sum(np.sort(yy)[::-1] * discount))
    if ideal <= 0:
        return 0.0
    _, inverse, counts = np.unique(-ss, return_inverse=True, return_counts=True)
    gain = np.zeros(len(counts)); np.add.at(gain, inverse, yy); gain /= counts
    ends = np.cumsum(counts) - 1; cumulative = np.cumsum(discount)
    share = np.r_[cumulative[ends[0]], np.diff(cumulative[ends])]
    return float(np.sum(gain * share)) / ideal


def ranking_metrics(y, score, groups, k=5, min_size=5):
    """Mean over queries with >= min_size candidates. Every `*_pos` value, Recall, MAP, MRR and HitRate use the queries with >= 1 positive."""
    ndcg = []; ndcg_pos = []; ndcg10_pos = []; precision = []; recall = []; recall10 = []; hit = []; ap = []; ap_full = []; rr = []
    for idx in per_query_indices(groups):
        if len(idx) < min_size:
            continue
        yy = y[idx]; ss = score[idx]
        order = np.argsort(-ss, kind='stable')
        top = yy[order[:k]]
        value = _ndcg(yy, ss, k)
        ndcg.append(value)
        precision.append(top.mean())
        positives = yy.sum()
        if positives > 0:
            ndcg_pos.append(value); ndcg10_pos.append(_ndcg(yy, ss, 10))
            recall.append(top.sum() / positives); recall10.append(yy[order[:10]].sum() / positives)
            hit.append(float(top.sum() > 0))
            ranked = yy[order]
            hits = np.cumsum(ranked > 0); ranks = np.arange(1, len(ranked) + 1)
            ap.append(float(np.sum((hits / ranks)[:k] * (ranked[:k] > 0)) / min(positives, k)))
            ap_full.append(float(np.sum((hits / ranks) * (ranked > 0)) / positives))
            first = np.flatnonzero(ranked > 0)
            rr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
    mean = lambda values: float(np.mean(values)) if values else None
    return {'queries': len(ndcg), 'ndcg_at_5': mean(ndcg), 'precision_at_5': mean(precision),
            'queries_with_positive': len(recall), 'ndcg_at_5_pos': mean(ndcg_pos), 'ndcg_at_10_pos': mean(ndcg10_pos),
            'recall_at_5': mean(recall), 'recall_at_10': mean(recall10), 'hit_rate_at_5': mean(hit),
            'map_at_5': mean(ap), 'map': mean(ap_full), 'mrr': mean(rr)}


def row_metrics(y, score, probabilistic):
    out = {'rows': int(len(y)), 'positive_rate': float(y.mean())}
    both = len(np.unique(y)) == 2
    out['roc_auc'] = float(roc_auc_score(y, score)) if both else None
    out['pr_auc'] = float(average_precision_score(y, score)) if both else None
    out['brier'] = float(brier_score_loss(y, np.clip(score, 0, 1))) if probabilistic else None
    return out


def evaluate(y, pred, groups, probabilistic=True, k=5):
    """pred: (n, 3) with column 0 the ranking score; columns 1-2 log ratios (may be NaN for ranking-only models)."""
    score = np.asarray(pred)[:, 0]
    out = ranking_metrics(y[:, 0], score, groups, k=k)
    out.update(row_metrics(y[:, 0], score, probabilistic))
    ratios = np.asarray(pred)[:, 1:]
    out['log_ratio_mae'] = float(np.mean(np.abs(y[:, 1:] - ratios))) if np.isfinite(ratios).all() else None
    return out
