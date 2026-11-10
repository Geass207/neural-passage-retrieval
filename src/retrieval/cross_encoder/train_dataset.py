"""
Dataset d'entraînement pour le cross-encoder BGE sur MS MARCO.

Chaque triplet MS MARCO produit deux paires :
    - (requête, passage positif) avec label 1.0
    - (requête, passage négatif) avec label 0.0

Le chargement repose sur ir_datasets afin d'éviter les hypothèses fragiles
sur des splits "train" au format BEIR.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from sentence_transformers import InputExample


def load_cross_encoder_examples(
    max_triplets: int,
    seed: int = 42,
) -> List[InputExample]:
    """Charge un sous-ensemble fixe de triplets MS MARCO et retourne des paires labellisées."""
    import ir_datasets

    dataset = ir_datasets.load("msmarco-passage/train/triples-small")
    queries: Dict[str, str] = {query.query_id: query.text for query in dataset.queries_iter()}
    docs_store = dataset.docs_store()

    examples: List[InputExample] = []
    rng = random.Random(seed)

    n_triplets = 0
    for pair in dataset.docpairs_iter():
        query_text = queries.get(pair.query_id)
        if not query_text:
            continue

        pos_doc = docs_store.get(pair.doc_id_a)
        neg_doc = docs_store.get(pair.doc_id_b)
        if pos_doc is None or neg_doc is None:
            continue

        examples.append(InputExample(texts=[query_text, pos_doc.text], label=1.0))
        examples.append(InputExample(texts=[query_text, neg_doc.text], label=0.0))
        n_triplets += 1

        if n_triplets >= max_triplets:
            break

    rng.shuffle(examples)
    return examples
