"""
Cross-Encoder Reranker : deuxième étage de reranking pour la recherche d'information.
Position dans le pipeline : BM25 (top-100) → ce reranker → top-k reclassé

À la différence du bi-encoder (DenseRetriever) qui encode séparément,
le cross-encoder encode la paire (requête, passage) ensemble pour plus de précision.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.data.types import Passage, Query, Results
from src.retrieval.base_retriever import BaseRetriever


class CrossEncoderReranker(BaseRetriever):
    """
    Reranker basé sur un Cross-Encoder pour reclasser les résultats BM25.

    Utilisation :
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
        reranker.index(corpus)  # Optionnel: cache les embeddings
        results = reranker.rerank(bm25_results, corpus, queries, top_k=10)

    Pipeline :
        1. BM25 retourne top-100 pour chaque requête
        2. Cross-Encoder rescores chaque paire (query, passage)
        3. Retourne top-k reclassé
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model_name: Nom du modèle Cross-Encoder sur HuggingFace
            model_path: Chemin vers un modèle fine-tuné local
            device: Device pour l'inférence (cuda/cpu)
            max_length: Longueur max des paires (query + passage)
        """
        self.model_name = model_path or model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"Chargement du Cross-Encoder : {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=max_length,
            device=self.device,
        )

        if self.device == "cuda":
            self.model.model.half()

        self.corpus_pids: Optional[List[str]] = None
        self.corpus_texts: Optional[List[str]] = None
        self._is_indexed = False

    def index(self, corpus: Dict[str, Passage]) -> None:
        """
        Pré-extrait les textes du corpus pour un reranking plus rapide.
        Optionnel mais recommandé pour éviter les lookup répétés.
        """
        self.corpus_pids = list(corpus.keys())
        self.corpus_texts = [corpus[pid].text for pid in self.corpus_pids]
        self._is_indexed = True
        print(f"Indexé {len(corpus):,} passages pour reranking")

    def retrieve(self, queries: Dict[str, Query], top_k: int = 100) -> Results:
        raise NotImplementedError(
            "CrossEncoderReranker ne fait pas de retrieval direct. "
            "Utilisez rerank() avec les résultats BM25."
        )

    def rerank(
        self,
        first_stage_results: Results,
        corpus: Dict[str, Passage],
        queries: Dict[str, Query],
        top_k: int = 10,
    ) -> Results:
        """
        Reclasse les résultats du premier étage (BM25) avec le Cross-Encoder.

        Args:
            first_stage_results: Résultats BM25 au format {qid: [(pid, score), ...]}
            corpus: Corpus complet pour récupérer les textes des passages
            queries: Requêtes au format {qid: Query}
            top_k: Nombre de résultats à retourner

        Returns:
            Résultats reclassés au format {qid: [(pid, score), ...]}
        """
        results: Results = {}

        for qid, query in tqdm(queries.items(), desc="Reranking"):
            if qid not in first_stage_results:
                results[qid] = []
                continue

            bm25_candidates = first_stage_results[qid]

            if not bm25_candidates:
                results[qid] = []
                continue

            query_text = query.text

            passage_texts = []
            passage_ids = []

            for pid, _ in bm25_candidates:
                if pid in corpus:
                    passage_ids.append(pid)
                    passage_texts.append(
                        [query_text, corpus[pid].text]
                    )

            if not passage_texts:
                results[qid] = []
                continue

            scores = self.model.predict(passage_texts)

            reranked = sorted(
                zip(passage_ids, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            results[qid] = [(pid, float(score)) for pid, score in reranked]

        return results

    def rerank_batch(
        self,
        first_stage_results: Results,
        corpus: Dict[str, Passage],
        queries: Dict[str, Query],
        batch_size: int = 32,
        top_k: int = 10,
    ) -> Results:
        """
        Version optimisée par batch pour traiter plusieurs requêtes ensemble.
        Plus rapide mais nécessite plus de mémoire.
        """
        results: Results = {}
        qids = list(queries.keys())

        all_pairs: List[Tuple[str, str, str]] = []

        query_mapping = []
        for qid in qids:
            if qid not in first_stage_results:
                results[qid] = []
                continue

            query_text = queries[qid].text
            for pid, _ in first_stage_results[qid]:
                if pid in corpus:
                    query_mapping.append((qid, pid))
                    all_pairs.append((query_text, corpus[pid].text))

        if not all_pairs:
            return results

        all_scores = self.model.predict(all_pairs, batch_size=batch_size)

        score_map = {}
        for (qid, pid), score in zip(query_mapping, all_scores):
            if qid not in score_map:
                score_map[qid] = []
            score_map[qid].append((pid, score))

        for qid in qids:
            if qid in score_map:
                reranked = sorted(
                    score_map[qid],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                results[qid] = [(pid, float(score)) for pid, score in reranked]
            else:
                results[qid] = []

        return results