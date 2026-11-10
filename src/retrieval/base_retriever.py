"""Interface abstraite commune à tous les systèmes de recherche."""
from abc import ABC, abstractmethod
from typing import Dict

from src.data.types import Passage, Query, Results


class BaseRetriever(ABC):
    """
    Classe de base pour tous les retrievers (BM25, bi-encoder, reranker...).
    Implémenter index() et retrieve() suffit pour brancher un nouveau système.
    """

    @abstractmethod
    def index(self, corpus: Dict[str, Passage]) -> None:
        """Construit l'index à partir du corpus fourni."""
        ...

    @abstractmethod
    def retrieve(self, queries: Dict[str, Query], top_k: int = 100) -> Results:
        """Retourne les top_k passages pour chaque requête, triés par score décroissant."""
        ...

    def save_index(self, path: str) -> None:
        """Sauvegarde l'index sur disque (optionnel)."""
        pass

    def load_index(self, path: str) -> None:
        """Charge l'index depuis le disque (optionnel)."""
        pass
