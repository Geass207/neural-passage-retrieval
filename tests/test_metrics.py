"""Unit tests for the ranking metrics."""

import math

import pytest

from src.evaluation.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k

QRELS = {"q1": {"d1": 1}, "q2": {"d5": 1, "d6": 1}}
RUN = {
    "q1": [("d3", 3.0), ("d1", 2.0), ("d2", 1.0)],
    "q2": [("d5", 9.0), ("d9", 8.0), ("d8", 7.0)],
}


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(QRELS, RUN, k=10) == pytest.approx((1 / 2 + 1) / 2)


def test_mrr_is_zero_beyond_cutoff():
    assert mrr_at_k({"q1": {"d1": 1}}, RUN, k=1) == 0.0


def test_recall_counts_fraction_of_relevant_passages():
    assert recall_at_k(QRELS, RUN, k=1) == pytest.approx((0 + 1 / 2) / 2)
    assert recall_at_k(QRELS, RUN, k=3) == pytest.approx((1 + 1 / 2) / 2)


def test_ndcg_matches_hand_computation():
    q1 = (1 / math.log2(3)) / 1.0
    q2 = 1.0 / (1.0 + 1 / math.log2(3))
    assert ndcg_at_k(QRELS, RUN, k=10) == pytest.approx((q1 + q2) / 2)


def test_missing_query_scores_zero():
    assert mrr_at_k({"q3": {"d1": 1}}, RUN) == 0.0
    assert recall_at_k({"q3": {"d1": 1}}, RUN, k=10) == 0.0


def test_evaluate_run_reports_all_metrics():
    metrics = evaluate_run(QRELS, RUN, ks=[10, 100])
    assert set(metrics) == {"MRR@10", "nDCG@10", "Recall@10", "Recall@100"}
