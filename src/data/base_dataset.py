from abc import ABC, abstractmethod
from typing import Dict, Tuple

from src.data.types import Passage, Query, Qrels


class BaseDataset(ABC):
    """Abstract base class for all datasets. Implement this to add a new dataset."""

    @abstractmethod
    def load_corpus(self) -> Dict[str, Passage]:
        """Load all passages. Returns {pid: Passage}."""
        ...

    @abstractmethod
    def load_queries(self, split: str = "dev") -> Dict[str, Query]:
        """Load queries for a given split. Returns {qid: Query}."""
        ...

    @abstractmethod
    def load_qrels(self, split: str = "dev") -> Qrels:
        """Load relevance judgements for a given split. Returns {qid: {pid: score}}."""
        ...

    def get_split(self, split: str = "dev") -> Tuple[Dict[str, Passage], Dict[str, Query], Qrels]:
        """Convenience method: load corpus, queries, and qrels in one call."""
        return self.load_corpus(), self.load_queries(split), self.load_qrels(split)
