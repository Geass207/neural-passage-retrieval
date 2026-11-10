"""
Fine-tuning léger du cross-encoder BGE-reranker-v2-m3 sur MS MARCO.

Le but n'est pas de faire un entraînement massif, mais d'obtenir un reranker
cohérent avec le protocole commun du projet.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.cross_encoder.train_dataset import load_cross_encoder_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Cross-Encoder Reranker")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3", help="Model à fine-tuner")
    parser.add_argument("--epochs", type=int, default=1, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Longueur max")
    parser.add_argument("--max_train_triplets", type=int, default=5000, help="Nombre max de triplets MS MARCO")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio warmup")
    parser.add_argument("--output_dir", default="results/bge_reranker_ft", help="Répertoire de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fine-tuning Cross-Encoder Reranker")
    print("=" * 60)
    print(f"Modèle: {args.model}")
    print(f"Époques: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    print("\n[1/3] Chargement des données d'entraînement...")
    train_examples = load_cross_encoder_examples(
        max_triplets=args.max_train_triplets,
        seed=args.seed,
    )
    print(f"Examples chargés : {len(train_examples):,}")

    print("\n[2/3] Initialisation du modèle...")
    model = CrossEncoder(
        args.model,
        max_length=args.max_seq_length,
        device=device,
    )

    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    print("\n[3/3] Fine-tuning...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    # Sauvegarde explicite — output_path dans fit() n'est pas fiable en ST v3
    checkpoint_dir = output_dir / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir))

    summary = {
        "model": args.model,
        "max_train_triplets": args.max_train_triplets,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "device": device,
        "output_path": str(output_dir / "best_checkpoint"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"✓ Modèle sauvegardé: {output_dir / 'best_checkpoint'}")
    print("\n=== Fine-tuning terminé ===")


if __name__ == "__main__":
    main()
