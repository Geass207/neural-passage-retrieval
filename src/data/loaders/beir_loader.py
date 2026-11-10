"""Chargeur générique pour tout dataset au format BEIR."""
from pathlib import Path
from typing import Dict

from src.data.base_dataset import BaseDataset
from src.data.io_utils import load_corpus_jsonl, load_queries_jsonl, load_qrels_tsv
from src.data.types import Passage, Query, Qrels


class BEIRLoader(BaseDataset):
    """
    Chargeur générique pour n'importe quel dataset au format BEIR.
    Attend la structure : data_dir/<name>/{corpus.jsonl, queries.jsonl, qrels/<split>.tsv}
    """

    def __init__(self, name: str, data_dir: str = "data/") -> None:
        self.root = Path(data_dir) / name
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' introuvable dans {self.root}. "
                "Téléchargez les données manuellement ou utilisez un loader dédié."
            )

    def load_corpus(self) -> Dict[str, Passage]:
        """Charge le corpus depuis corpus.jsonl."""
        return load_corpus_jsonl(str(self.root / "corpus.jsonl"))

    def load_queries(self, split: str = "dev") -> Dict[str, Query]:
        """Charge les requêtes depuis queries.jsonl."""
        return load_queries_jsonl(str(self.root / "queries.jsonl"))

    def load_qrels(self, split: str = "dev") -> Qrels:
        """Charge les jugements de pertinence depuis qrels/<split>.tsv."""
        path = self.root / "qrels" / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Qrels introuvables : {path}")
        return load_qrels_tsv(str(path))
