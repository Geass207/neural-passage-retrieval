from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["Passage", "Query", "Qrels", "Results"]


@dataclass
class Passage:
    pid: str
    text: str
    title: str = ""


@dataclass
class Query:
    qid: str
    text: str


# {qid: {pid: relevance_score}}
Qrels = Dict[str, Dict[str, int]]

# {qid: [(pid, score), ...]} sorted by score descending
Results = Dict[str, List[Tuple[str, float]]]
