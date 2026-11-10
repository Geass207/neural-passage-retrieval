"""BM25-teacher pairwise distillation components."""

from src.retrieval.pairdistill.data import PairwiseExample, build_pairwise_examples

__all__ = [
    "PairwiseExample",
    "build_pairwise_examples",
]
