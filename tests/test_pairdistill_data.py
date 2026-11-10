"""Unit tests for the construction of teacher preference pairs."""

from src.retrieval.pairdistill.data import build_pairwise_examples


def _ranking(n=100):
    return [(f"d{i}", float(n - i)) for i in range(n)]


def test_pairs_respect_teacher_order_and_margins():
    examples = build_pairwise_examples(
        {"q": _ranking()}, num_pairs_per_query=8, positive_pool=10,
        negative_pool_start=20, min_rank_gap=5, min_score_gap=0.01,
    )
    assert len(examples) == 8
    for ex in examples:
        pos_rank, neg_rank = int(ex.pos_pid[1:]) + 1, int(ex.neg_pid[1:]) + 1
        assert pos_rank <= 10 and neg_rank >= 20
        assert neg_rank - pos_rank >= 5
        assert ex.teacher_gap >= 0.01


def test_pairs_are_deterministic_given_seed():
    a = build_pairwise_examples({"q": _ranking()}, seed=7)
    b = build_pairwise_examples({"q": _ranking()}, seed=7)
    assert a == b


def test_ties_below_score_gap_are_dropped():
    flat = [(f"d{i}", 1.0) for i in range(50)]
    assert build_pairwise_examples({"q": flat}) == []


def test_short_rankings_are_skipped():
    assert build_pairwise_examples({"q": [("d0", 1.0)]}) == []
