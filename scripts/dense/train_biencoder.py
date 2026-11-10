"""
Fine-tuning léger d'un bi-encoder dense sur MS MARCO avec sentence-transformers.

Objectif :
    Adapter BGE à faible coût en utilisant des paires (requête, passage+)
    et la perte MultipleNegativesRankingLoss.

Pourquoi ce choix :
    - plus simple et plus robuste qu'un pipeline complet avec FAISS + hard negatives
    - exploitable sur GPU MIG 12 Go
    - cohérent avec une contrainte de temps serrée avant la remise

Utilisation :
    python scripts/dense/train_biencoder.py \
        --model BAAI/bge-large-en-v1.5 \
        --max_train_samples 50000 \
        --epochs 1 \
        --batch_size 8 \
        --output_dir results/dense/training/bge_biencoder_ft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# Désactive explicitement les intégrations de reporting externes.
os.environ.setdefault("WANDB_DISABLED", "true")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Fine-tuning dense bi-encoder sur MS MARCO")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max_train_samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/dense/training/bge_biencoder_ft")
    parser.add_argument(
        "--save_training_examples",
        action="store_true",
        help="Sauvegarde un petit aperçu des paires d'entraînement pour audit rapide",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Fixe les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_examples(max_train_samples: int) -> Tuple[List[InputExample], List[Dict[str, str]]]:
    """
    Charge un sous-ensemble fixe de paires positives MS MARCO.

    On utilise uniquement (query, positive passage).
    Les négatifs sont fournis implicitement par les autres éléments du batch
    via MultipleNegativesRankingLoss.
    """
    import ir_datasets

    dataset = ir_datasets.load("msmarco-passage/train/triples-small")
    queries: Dict[str, str] = {query.query_id: query.text for query in dataset.queries_iter()}
    docs_store = dataset.docs_store()

    examples: List[InputExample] = []
    preview_rows: List[Dict[str, str]] = []

    for pair in dataset.docpairs_iter():
        query_text = queries.get(pair.query_id)
        if not query_text:
            continue

        pos_doc = docs_store.get(pair.doc_id_a)
        if pos_doc is None or not pos_doc.text:
            continue

        examples.append(InputExample(texts=[query_text, pos_doc.text]))

        if len(preview_rows) < 20:
            preview_rows.append(
                {
                    "query_id": pair.query_id,
                    "positive_pid": pair.doc_id_a,
                    "query": query_text,
                    "positive_text": pos_doc.text[:300],
                }
            )

        if len(examples) >= max_train_samples:
            break

    return examples, preview_rows


def main() -> None:
    """Point d'entrée principal."""
    args = parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("Fine-tuning dense bi-encoder")
    print("=" * 60)
    print(f"Modèle             : {args.model}")
    print(f"Max train samples  : {args.max_train_samples:,}")
    print(f"Époques            : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.lr}")
    print(f"Max sequence len   : {args.max_seq_length}")
    print(f"Device             : {config['device']}")

    print("\n[1/3] Chargement des exemples MS MARCO...")
    train_examples, preview_rows = load_training_examples(args.max_train_samples)
    print(f"Exemples chargés : {len(train_examples):,}")

    if args.save_training_examples:
        (output_dir / "training_examples_preview.json").write_text(
            json.dumps(preview_rows, indent=2, ensure_ascii=False)
        )

    print("\n[2/3] Initialisation du modèle...")
    model = SentenceTransformer(args.model, device=config["device"])
    model.max_seq_length = args.max_seq_length

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * args.epochs * args.warmup_ratio)
    print(f"Warmup steps : {warmup_steps}")

    print("\n[3/3] Entraînement...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(output_dir / "best_checkpoint"),
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )

    summary = {
        "n_examples": len(train_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "output_path": str(output_dir / "best_checkpoint"),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\nEntraînement terminé.")
    print(f"Checkpoint : {output_dir / 'best_checkpoint'}")


if __name__ == "__main__":
    main()
