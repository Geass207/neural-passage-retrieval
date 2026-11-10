"""Retriever dense : implémente BaseRetriever avec des bi-encoders (sentence-transformers)."""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.types import Passage, Query, Results
from src.retrieval.base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Système de recherche basé sur des bi-encoders (embeddings denses).
    Utilise sentence-transformers pour encoder corpus et requêtes,
    puis effectue une recherche par similarité cosinus.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        model_path: Optional[str] = None,
        embeddings_cache: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Nom du modèle HuggingFace (ex: "BAAI/bge-large-en-v1.5")
            model_path: Chemin vers un modèle fine-tuné local (optionnel)
            embeddings_cache: Chemin pour cacher les embeddings du corpus
            device: Device pour l'inférence (cuda/cpu), auto-détecté si None
        """
        self.model_name = model_path or model_name
        self.embeddings_cache = embeddings_cache
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with FP16 for faster inference
        print(f"Chargement du modèle : {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Enable fp16 if CUDA is available
        if self.device == "cuda":
            self.model.half()

        # État interne
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_pids: Optional[list] = None
        self._is_indexed = False

    def index(self, corpus: Dict[str, Passage]) -> None:
        """
        Encode tous les passages du corpus et les indexe en mémoire.
        Peut charger depuis cache si disponible.
        """
        # Vérifier le cache
        if self.embeddings_cache and Path(self.embeddings_cache).exists():
            print(f"Cache d'embeddings trouvé — chargement depuis {self.embeddings_cache}")
            data = np.load(self.embeddings_cache, allow_pickle=True)
            self.corpus_embeddings = data["embeddings"]
            self.corpus_pids = data["pids"].tolist()
            self._is_indexed = True
            return

        # Encode le corpus
        print(f"Encodage de {len(corpus):,} passages...")
        passages_texts = [p.text for p in corpus.values()]
        self.corpus_pids = list(corpus.keys())

        self.corpus_embeddings = self.model.encode(
            passages_texts,
            batch_size=512,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Normalise les embeddings pour la similarité cosinus
        self.corpus_embeddings = self.corpus_embeddings / (
            np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Sauvegarde le cache
        if self.embeddings_cache:
            Path(self.embeddings_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.embeddings_cache,
                embeddings=self.corpus_embeddings,
                pids=np.array(self.corpus_pids),
            )
            print(f"Cache d'embeddings sauvegardé : {self.embeddings_cache}")

        self._is_indexed = True

    def retrieve(self, queries: Dict[str, Query], top_k: int = 100) -> Results:
        """
        Encode les requêtes et effectue une recherche par similarité cosinus.
        Retourne les top_k passages les plus pertinents pour chaque requête.
        Format : {qid: [(pid, score), ...]} trié par score décroissant.
        """
        if not self._is_indexed:
            raise RuntimeError("L'index n'est pas prêt. Appelez index() d'abord.")

        # Encode les requêtes
        print(f"Encodage de {len(queries):,} requêtes...")
        queries_texts = [q.text for q in queries.values()]
        query_embeddings = self.model.encode(
            queries_texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Normalise les requêtes
        query_embeddings = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Calcul requête par requête — évite la matrice (6980 × 8,8M) qui ferait OOM
        results: Results = {}
        qids = list(queries.keys())
        for i, qid in tqdm(enumerate(qids), total=len(qids), desc="Classement dense"):
            scores_i = query_embeddings[i] @ self.corpus_embeddings.T  # (n_corpus,)
            top_indices = np.argsort(-scores_i)[:top_k]
            top_pids = [self.corpus_pids[idx] for idx in top_indices]
            results[qid] = list(zip(top_pids, scores_i[top_indices].astype(float)))

        return results

    def save_index(self, path: str) -> None:
        """Sauvegarde les embeddings du corpus sur disque."""
        if not self._is_indexed:
            raise RuntimeError("Aucun index à sauvegarder.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.corpus_embeddings,
            pids=np.array(self.corpus_pids),
        )
        print(f"Index sauvegardé : {path}")

    def load_index(self, path: str) -> None:
        """Charge les embeddings du corpus depuis le disque."""
        data = np.load(path, allow_pickle=True)
        self.corpus_embeddings = data["embeddings"]
        self.corpus_pids = data["pids"].tolist()
        self._is_indexed = True
        print(f"Index chargé : {path}")











