"""Dense student retriever used after pairwise distillation."""

from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.types import Passage, Query, Results
from src.retrieval.base_retriever import BaseRetriever


def format_passage(passage: Passage) -> str:
    """Return the text seen by the dense encoder for one passage."""
    if passage.title:
        return f"{passage.title} {passage.text}"
    return passage.text


class DenseStudentRetriever(BaseRetriever):
    """
    SentenceTransformer dense retriever.

    It encodes all passages, builds an inner-product index over normalized
    embeddings, then returns the top-k passages for each query.
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 128,
        query_batch_size: int = 32,
        device: Optional[str] = None,
        use_faiss: bool = True,
        keep_embeddings: bool = False,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size
        self.use_faiss = use_faiss
        self.keep_embeddings = keep_embeddings
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self._pids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

    def index(self, corpus: Dict[str, Passage]) -> None:
        self._pids = list(corpus.keys())
        texts = [format_passage(corpus[pid]) for pid in self._pids]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        ).astype("float32", copy=False)

        self._faiss_index = None
        if self.use_faiss:
            try:
                import faiss

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                self._faiss_index = index
                self._embeddings = embeddings if self.keep_embeddings else None
                return
            except ImportError:
                print("FAISS is not installed; falling back to NumPy search.")

        self._embeddings = embeddings

    def retrieve(self, queries: Dict[str, Query], top_k: int = 100) -> Results:
        if not self._pids:
            raise RuntimeError("The dense index is not ready. Call index() first.")
        if self._faiss_index is None and self._embeddings is None:
            raise RuntimeError("No dense search backend is available.")

        qids = list(queries.keys())
        texts = [queries[qid].text for qid in qids]
        top_k = min(top_k, len(self._pids))
        results: Results = {}

        for start in tqdm(range(0, len(qids), self.query_batch_size), desc="Dense search"):
            end = start + self.query_batch_size
            batch_qids = qids[start:end]
            batch_texts = texts[start:end]
            query_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.query_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            ).astype("float32", copy=False)

            if self._faiss_index is not None:
                scores, indices = self._faiss_index.search(query_embeddings, top_k)
            else:
                scores_matrix = query_embeddings @ self._embeddings.T
                indices = np.argpartition(-scores_matrix, kth=top_k - 1, axis=1)[:, :top_k]
                row = np.arange(scores_matrix.shape[0])[:, None]
                scores = scores_matrix[row, indices]
                order = np.argsort(-scores, axis=1)
                indices = np.take_along_axis(indices, order, axis=1)
                scores = np.take_along_axis(scores, order, axis=1)

            for row_idx, qid in enumerate(batch_qids):
                results[qid] = [
                    (self._pids[int(pid_idx)], float(score))
                    for pid_idx, score in zip(indices[row_idx], scores[row_idx])
                    if int(pid_idx) >= 0
                ]

        return results
