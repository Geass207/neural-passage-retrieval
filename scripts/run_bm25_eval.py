"""
Script d'évaluation BM25 sur MS MARCO (ou tout dataset BEIR).
Usage :
    python scripts/run_bm25_eval.py
    python scripts/run_bm25_eval.py --config config.yaml --split dev
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

# Ajoute la racine du projet au PYTHONPATH pour pouvoir importer src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.evaluation.results_io import save_run
from src.retrieval.bm25.bm25_retriever import BM25Retriever


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation BM25 sur un dataset BEIR")
    parser.add_argument("--config", default="config.yaml", help="Chemin vers config.yaml")
    parser.add_argument("--split", default=None, help="Split à évaluer (ex: dev)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force la reconstruction de l'index")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal du script d'évaluation."""
    args = parse_args()

    # --- Chargement de la configuration ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    split = args.split or cfg["dataset"]["split"]
    data_dir = cfg["dataset"]["data_dir"]
    top_k = cfg["retrieval"]["top_k"]
    index_path = cfg["bm25"]["index_path"]

    # Si reconstruction forcée, supprime l'index existant
    if args.rebuild_index and Path(index_path).exists():
        Path(index_path).unlink()
        print("Index existant supprimé — reconstruction forcée.")

    print(f"=== Évaluation BM25 | Dataset: {dataset_name} | Split: {split} ===")

    # --- Chargement des données ---
    dataset = get_dataset(dataset_name, data_dir=data_dir)
    corpus, queries, qrels = dataset.get_split(split)

    # Filtre les requêtes pour ne garder que celles présentes dans les qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    print(f"Corpus : {len(corpus):,} passages | Requêtes : {len(queries):,} | Qrels : {len(qrels):,}")

    # --- Indexation ---
    retriever = BM25Retriever(index_path=index_path)
    retriever.index(corpus)

    # --- Recherche ---
    results = retriever.retrieve(queries, top_k=top_k)

    # --- Évaluation ---
    metrics = evaluate_run(qrels, results, ks=[10, 50, 100])
    print("\n=== Métriques ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # --- Sauvegarde des artefacts ---
    run_dir = f"results/baselines/bm25_{dataset_name}_{split}"
    run_config = {
        "dataset": dataset_name,
        "split": split,
        "top_k": top_k,
        "index_path": index_path,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
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

    # Sauvegarde des résultats bruts pour que eval.py puisse les charger
    # sans avoir à refaire la recherche BM25 sur 8,8M passages
    run_json_path = Path(run_dir) / "run.json"
    run_json_path.write_text(json.dumps(results, ensure_ascii=False))
    print(f"\nRun complet. Résultats dans : {run_dir}/")


if __name__ == "__main__":
    main()
