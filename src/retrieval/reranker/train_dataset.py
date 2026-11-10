"""
Dataset d'entraînement MS MARCO pour le reranker Mistral pointwise.

Chaque exemple est un triplet (requête, passage+, passage-) formaté comme prompt.
Label 1 = pertinent ("yes"), label 0 = non pertinent ("no").

Source : ir_datasets "msmarco-passage/train/triples-small" — déjà installé,
pas de problème de schéma HuggingFace.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "Query: {query}\n"
    "Passage: {passage}\n"
    "Is this passage relevant to the query? Answer yes or no:"
)


class MSMarcoRerankerDataset(Dataset):
    """
    Dataset pointwise via ir_datasets : un (prompt, label) par exemple.

    Chaque triplet (query, pos, neg) donne 2 exemples : 1 positif + 1 négatif.
    max_samples triplets → 2 × max_samples exemples au total.

    max_samples     : nombre de triplets à charger depuis MS MARCO train
    max_passage_len : limite en caractères du passage
    seed            : graine de reproductibilité
    """

    def __init__(
        self,
        max_samples: Optional[int] = None,
        max_passage_len: int = 800,
        seed: int = 42,
    ) -> None:
        self.max_passage_len = max_passage_len
        rng = random.Random(seed)

        examples: List[Dict] = []
        for query_text, pos_text, neg_text in self._iter_triplets(max_samples):
            examples.append({"query": query_text, "passage": pos_text[:max_passage_len], "label": 1})
            examples.append({"query": query_text, "passage": neg_text[:max_passage_len], "label": 0})

        rng.shuffle(examples)
        self.examples = examples

        n_pos = sum(1 for e in examples if e["label"] == 1)
        print(f"Dataset : {len(examples)} exemples ({n_pos} pos, {len(examples)-n_pos} neg)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex    = self.examples[idx]
        prompt = PROMPT_TEMPLATE.format(query=ex["query"], passage=ex["passage"])
        label  = "yes" if ex["label"] == 1 else "no"
        return {"prompt": prompt, "label": label, "label_id": ex["label"]}

    @staticmethod
    def _iter_triplets(max_samples: Optional[int]):
        """
        Itère sur les triplets MS MARCO train via ir_datasets.
        Télécharge automatiquement si nécessaire (~2 Go dans ~/.ir_datasets/).
        """
        import ir_datasets

        ds = ir_datasets.load("msmarco-passage/train/triples-small")

        # Charge toutes les requêtes en mémoire (~50 Mo, instantané)
        queries: Dict[str, str] = {q.query_id: q.text for q in ds.queries_iter()}

        # Accès aléatoire aux passages par pid (mmap, pas de chargement complet)
        docs_store = ds.docs_store()

        n = 0
        for pair in ds.docpairs_iter():
            query = queries.get(pair.query_id, "")
            if not query:
                continue

            pos_doc = docs_store.get(pair.doc_id_a)
            neg_doc = docs_store.get(pair.doc_id_b)
            if pos_doc is None or neg_doc is None:
                continue

            yield query, pos_doc.text, neg_doc.text
            n += 1
            if max_samples and n >= max_samples:
                break
