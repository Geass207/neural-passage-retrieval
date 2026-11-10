"""Training data utilities for BM25-teacher pairwise distillation."""

from dataclasses import dataclass
import random
from typing import List

from src.data.types import Results


@dataclass(frozen=True)
class PairwiseExample:
    """One teacher preference: for a query, pos_pid should outrank neg_pid."""

    qid: str
    pos_pid: str
    neg_pid: str
    teacher_pos_score: float
    teacher_neg_score: float

    @property
    def teacher_gap(self) -> float:
        return self.teacher_pos_score - self.teacher_neg_score


def build_pairwise_examples(
    teacher_results: Results,
    num_pairs_per_query: int = 4,
    positive_pool: int = 10,
    negative_pool_start: int = 20,
    min_rank_gap: int = 5,
    min_score_gap: float = 0.01,
    seed: int = 42,
) -> List[PairwiseExample]:
    """
    Convert teacher rankings into pairwise training examples.

    BM25 is used as a teacher ranking. Documents near the top become preferred
    documents, and lower-ranked documents become comparison documents. The
    student then learns the teacher's pairwise ordering signal.
    """
    rng = random.Random(seed)
    examples: List[PairwiseExample] = []

    for qid, ranking in teacher_results.items():
        ranked = [
            (rank, pid, float(score))
            for rank, (pid, score) in enumerate(ranking, start=1)
        ]
        if len(ranked) < 2:
            continue

        positives = ranked[: max(1, min(positive_pool, len(ranked) - 1))]
        negative_start_idx = min(max(negative_pool_start - 1, 1), len(ranked) - 1)
        negatives = ranked[negative_start_idx:]
        if not negatives:
            negatives = ranked[len(ranked) // 2 :]

        for _ in range(num_pairs_per_query):
            pos_rank, pos_pid, pos_score = rng.choice(positives)
            eligible_negatives = [
                item
                for item in negatives
                if item[0] > pos_rank and item[0] - pos_rank >= min_rank_gap
            ]

            if not eligible_negatives:
                eligible_negatives = [
                    item
                    for item in ranked[pos_rank:]
                    if item[0] - pos_rank >= min_rank_gap
                ]
            if not eligible_negatives:
                continue

            _neg_rank, neg_pid, neg_score = rng.choice(eligible_negatives)
            if pos_pid == neg_pid:
                continue
            if (pos_score - neg_score) < min_score_gap:
                continue

            examples.append(
                PairwiseExample(
                    qid=qid,
                    pos_pid=pos_pid,
                    neg_pid=neg_pid,
                    teacher_pos_score=pos_score,
                    teacher_neg_score=neg_score,
                )
            )

    return examples
