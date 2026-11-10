"""Chargeur MS MARCO : lit les fichiers au format BEIR produits par msmarco_downloader."""
from pathlib import Path
from typing import Dict

from src.data.base_dataset import BaseDataset
from src.data.io_utils import load_corpus_jsonl, load_queries_jsonl, load_qrels_tsv
from src.data.types import Passage, Query, Qrels


class MSMarcoLoader(BaseDataset):
    """
    Charge MS MARCO depuis les fichiers au format BEIR dans data_dir/msmarco/.
    Si les fichiers sont absents, les télécharge automatiquement via msmarco_downloader.
    """

    def __init__(self, data_dir: str = "data/") -> None:
        self.root = Path(data_dir) / "msmarco"

    def _ensure_downloaded(self, split: str = "dev") -> None:
        corpus = self.root / "corpus.jsonl"
        queries = self.root / "queries.jsonl"
        qrels = self.root / "qrels" / f"{split}.tsv"
        if not (corpus.exists() and queries.exists() and qrels.exists()):
            print("MS MARCO files not found — downloading...")
            from src.data.loaders.msmarco_downloader import download_msmarco
            download_msmarco(str(self.root.parent))

        if split == "train" and not self._queries_cover_qrels(split):
            print("MS MARCO train queries missing — rebuilding queries.jsonl...")
            from src.data.loaders.msmarco_downloader import download_msmarco
            download_msmarco(str(self.root.parent), rebuild_queries=True)

    def _queries_cover_qrels(self, split: str) -> bool:
        """Vérifie rapidement qu'au moins une requête du split existe dans queries.jsonl."""
        queries = self.root / "queries.jsonl"
        qrels = self.root / "qrels" / f"{split}.tsv"
        if not (queries.exists() and qrels.exists()):
            return False

        first_qid = None
        with open(qrels, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts:
                    first_qid = parts[0]
                    break
        if first_qid is None:
            return False

        with open(queries, "r", encoding="utf-8") as f:
            needle = f'"_id": "{first_qid}"'
            return any(needle in line for line in f)

    def load_corpus(self) -> Dict[str, Passage]:
        """Charge le corpus MS MARCO complet."""
        self._ensure_downloaded()
        return load_corpus_jsonl(str(self.root / "corpus.jsonl"))

    def load_queries(self, split: str = "dev") -> Dict[str, Query]:
        """Charge toutes les requêtes MS MARCO (le split est géré par les qrels)."""
        self._ensure_downloaded(split)
        # Toutes les requêtes sont dans queries.jsonl ; les qrels filtrent par split
        return load_queries_jsonl(str(self.root / "queries.jsonl"))

    def load_qrels(self, split: str = "dev") -> Qrels:
        """Charge les jugements de pertinence pour le split demandé."""
        self._ensure_downloaded(split)
        path = self.root / "qrels" / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Qrels introuvables : {path}")
        return load_qrels_tsv(str(path))
