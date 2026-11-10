"""
Script d'évaluation Dense Retrieval sur MS MARCO (ou tout dataset BEIR).
Utilise BGE-large-en-v1.5 ou un modèle fine-tuné comme bi-encoder.

Usage :
    # Dataset complet (plus long)
    python scripts/run_dense_eval.py

    # 10% subset pour développement rapide
    python scripts/run_dense_eval.py --subset-pct 0.1

    # Modèle custom
    python scripts/run_dense_eval.py --model-path ./output/finetuned_model --rebuild-embeddings

    # 5% subset avec modèle custom
    python scripts/run_dense_eval.py --model-path ./output/finetuned_model --subset-pct 0.05 --rebuild-embeddings
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.evaluation.results_io import save_run
from src.retrieval.dense.dense_retriever import DenseRetriever


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation Dense Retrieval sur un dataset BEIR")
    parser.add_argument("--config", default="config.yaml", help="Chemin vers config.yaml")
    parser.add_argument("--split", default=None, help="Split à évaluer (ex: dev)")
    parser.add_argument(
        "--model",
        default="BAAI/bge-large-en-v1.5",
        help="Nom du modèle HuggingFace (par défaut: BGE-large-en-v1.5)",
    )
    parser.add_argument("--model-path", default=None, help="Chemin vers un modèle fine-tuné local")
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force la reconstruction des embeddings du corpus",
    )
    parser.add_argument(
        "--subset-pct",
        type=float,
        default=None,
        help="Utiliser un subset du dataset (ex: 0.1 = 10%% pour développement rapide). None = dataset complet",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du script d'évaluation."""
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    split = args.split or cfg["dataset"]["split"]
    data_dir = cfg["dataset"]["data_dir"]
    top_k = cfg["retrieval"]["top_k"]

    model_suffix = args.model_path.split("/")[-1] if args.model_path else args.model.split("/")[-1]
    embeddings_cache = f"data/{dataset_name}/dense_embeddings_{model_suffix}.npz"

    if args.rebuild_embeddings and Path(embeddings_cache).exists():
        Path(embeddings_cache).unlink()
        print("Cache d'embeddings supprimé — reconstruction forcée.")

    model_name = args.model_path or args.model

    # Subset info
    corpus_info = f" (subset: {args.subset_pct*100:.0f}%)" if args.subset_pct else ""
    print(f"=== Dense Retrieval Evaluation | Dataset: {dataset_name} | Split: {split}{corpus_info} ===")
    print(f"Model: {model_name}")

    # CUDA status
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} (CUDA enabled)")
    else:
        print("[CPU] Running on CPU (CUDA not available - this will be slow)")

    # Load dataset (full or subset)
    if args.subset_pct:
        print(f"[SUBSET] Using MS MARCO subset ({args.subset_pct*100:.0f}%) for faster development")
        dataset = get_dataset("msmarco_subset", data_dir=data_dir, subset_pct=args.subset_pct)
    else:
        dataset = get_dataset(dataset_name, data_dir=data_dir)
    corpus, queries, qrels = dataset.get_split(split)

    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    print(f"Corpus : {len(corpus):,} passages | Requêtes : {len(queries):,} | Qrels : {len(qrels):,}")

    retriever = DenseRetriever(
        model_name=args.model,
        model_path=args.model_path,
        embeddings_cache=embeddings_cache,
    )
    retriever.index(corpus)

    results = retriever.retrieve(queries, top_k=top_k)

    metrics = evaluate_run(qrels, results, ks=[10, 50, 100])
    print("\n=== Métriques ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    run_dir = f"results/dense_{dataset_name}_{split}"
    run_config = {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "top_k": top_k,
        "embeddings_cache": embeddings_cache,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "subset_pct": args.subset_pct,
    }
    save_run(
        run_dir=run_dir,
        metrics=metrics,
        results=results,
        qrels=qrels,
        queries=queries,
        corpus=corpus,
        config=run_config,
    )
    print(f"\nRun complet. Résultats dans : {run_dir}/")


if __name__ == "__main__":
    main()



