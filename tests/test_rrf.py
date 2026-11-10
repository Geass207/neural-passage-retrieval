"""Unit tests for Reciprocal Rank Fusion."""

import pytest

from src.retrieval.hybrid.rrf_retriever import RRFRetriever


def test_rrf_rewards_agreement_between_systems():
    a = {"q": [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]}
    b = {"q": [("d2", 0.9), ("d4", 0.8), ("d1", 0.7)]}
    fused = RRFRetriever().fuse(a, b, k=60)["q"]
    assert [pid for pid, _ in fused[:2]] == ["d2", "d1"]
    assert dict(fused)["d2"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_keeps_queries_seen_by_only_one_system_and_truncates():
    a = {"q1": [("d1", 1.0), ("d2", 0.5)]}
    b = {"q2": [("d3", 1.0)]}
    fused = RRFRetriever().fuse(a, b, top_k=1)
    assert set(fused) == {"q1", "q2"}
    assert fused["q1"] == [("d1", pytest.approx(1 / 61))]
