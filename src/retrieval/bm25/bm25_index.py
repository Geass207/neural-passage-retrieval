"""Construction, sauvegarde et chargement de l'index BM25 via bm25s."""
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.data.types import Passage

# Même regex que bm25s par défaut
_TOKENIZE_RE = re.compile(r"(?u)\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return _TOKENIZE_RE.findall(text.lower())


class BM25Index:
    """
    Encapsule un index bm25s.

    Le vocabulaire construit à l'indexation est sauvegardé et réutilisé pour
    tokeniser les requêtes — garantit la cohérence corpus/requêtes.
    """

    def __init__(self) -> None:
        self._index = None
        self._pids:  List[str]      = []
        self._vocab: Dict[str, int] = {}   # terme → id dans l'index bm25s

    def build(self, corpus: Dict[str, Passage]) -> None:
        """Construit l'index BM25 à partir du corpus."""
        import bm25s

        print(f"Construction de l'index BM25 sur {len(corpus):,} passages...")
        self._pids = list(corpus.keys())

        texts = [
            f"{p.title} {p.text}" if p.title else p.text
            for p in tqdm(corpus.values(), desc="Préparation textes")
        ]

        corpus_tokens   = bm25s.tokenize(texts, show_progress=True)
        self._vocab     = corpus_tokens.vocab   # vocabulaire à conserver
        self._index     = bm25s.BM25()
        self._index.index(corpus_tokens)
        print("Index BM25 construit.")

    def retrieve_batch(self, query_texts: List[str], top_k: int) -> List[List[tuple]]:
        """
        Recherche en batch.
        Les requêtes sont tokenisées avec le vocabulaire du corpus pour garantir
        la cohérence des token IDs.
        """
        if self._index is None:
            raise RuntimeError("L'index n'est pas encore construit.")

        import bm25s

        # Tokenise les requêtes avec le même vocabulaire que le corpus
        query_ids = [
            [self._vocab[t] for t in _tokenize(text) if t in self._vocab]
            for text in query_texts
        ]
        tokenized_queries = bm25s.tokenization.Tokenized(
            ids=query_ids, vocab=self._vocab
        )

        results, scores = self._index.retrieve(
            tokenized_queries, k=min(top_k, len(self._pids))
        )

        return [
            [(self._pids[int(idx)], float(sc)) for idx, sc in zip(idxs, scs)]
            for idxs, scs in zip(results, scores)
        ]

    def save(self, path: str) -> None:
        """Sérialise l'index sur disque."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "pids": self._pids, "vocab": self._vocab}, f)
        print(f"Index sauvegardé dans {path}")

    def load(self, path: str) -> None:
        """Charge l'index depuis le disque."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._pids  = data["pids"]
        self._vocab = data.get("vocab", {})
        print(f"Index chargé depuis {path} ({len(self._pids):,} passages)")

    @property
    def is_built(self) -> bool:
        return self._index is not None
