"""Retriever BM25 : implémente BaseRetriever via BM25Index."""
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.data.types import Passage, Query, Results
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.bm25.bm25_index import BM25Index


class BM25Retriever(BaseRetriever):
    """
    Système de recherche basé sur BM25 via bm25s.
    Utilise la recherche en batch — toutes les requêtes traitées en une passe.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        self._bm25       = BM25Index()
        self._index_path = index_path

    def index(self, corpus: Dict[str, Passage]) -> None:
        """Construit l'index BM25. Charge depuis le disque si déjà existant."""
        if self._index_path and Path(self._index_path).exists():
            print(f"Index existant trouvé — chargement depuis {self._index_path}")
            self._bm25.load(self._index_path)
            return

        self._bm25.build(corpus)

        if self._index_path:
            self._bm25.save(self._index_path)

    def retrieve(self, queries: Dict[str, Query], top_k: int = 100) -> Results:
        """
        Retourne les top_k passages les plus pertinents pour chaque requête.
        Traitement en batch : toutes les requêtes envoyées à bm25s en une passe.
        """
        if not self._bm25.is_built:
            raise RuntimeError("L'index n'est pas prêt. Appelez index() d'abord.")

        print(f"Recherche BM25 sur {len(queries):,} requêtes (batch)...")
        qids        = list(queries.keys())
        query_texts = [queries[qid].text for qid in qids]

        batch = self._bm25.retrieve_batch(query_texts, top_k)
        return {qid: pairs for qid, pairs in zip(qids, batch)}
