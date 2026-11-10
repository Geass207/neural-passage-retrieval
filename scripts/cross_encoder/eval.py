"""
Évaluation du cross-encoder BGE sur MS MARCO dev.

Protocole aligné sur le reranker Mistral :
    BM25 top-k-first -> reranker -> top-k-final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.evaluation.results_io import save_run
from src.retrieval.cross_encoder.cross_encoder_reranker import CrossEncoderReranker


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation Cross-Encoder sur MS MARCO dev")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--top_k_first", type=int, default=20)
    parser.add_argument("--top_k_rerank", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--run_name", default="bge_reranker_msmarco_dev")
    return parser.parse_args()


def main() -> None:
    """Point d'entrée principal."""
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"]["split"]
    data_dir = cfg["dataset"]["data_dir"]

    print(f"=== Évaluation Cross-Encoder | Dataset : {dataset_name} | Split : {split} ===")
    dataset = get_dataset(dataset_name, data_dir=data_dir)
    queries = dataset.load_queries(split)
    qrels = dataset.load_qrels(split)
    queries = {qid: query for qid, query in queries.items() if qid in qrels}

    if args.max_queries:
        import random

        random.seed(42)
        subset_qids = random.sample(list(queries.keys()), min(args.max_queries, len(queries)))
        queries = {qid: queries[qid] for qid in subset_qids}
        qrels = {qid: qrels[qid] for qid in subset_qids}

    bm25_results_path = PROJECT_ROOT / f"results/baselines/bm25_{dataset_name}_{split}/run.json"
    if not bm25_results_path.exists():
        raise FileNotFoundError(
            f"Résultats BM25 introuvables : {bm25_results_path}\n"
            "Lance d'abord la baseline BM25."
        )

    raw = json.loads(bm25_results_path.read_text())
    bm25_results = {
        qid: [tuple(pair) for pair in raw[qid][:args.top_k_first]]
        for qid in queries
        if qid in raw
    }

    metrics_bm25 = evaluate_run(qrels, bm25_results, ks=[10, 50, 100])
    print("Métriques BM25 candidate set :")
    for name, value in metrics_bm25.items():
        print(f"  {name}: {value:.4f}")

    needed_pids = {pid for pairs in bm25_results.values() for pid, _ in pairs}
    print(f"\nChargement de {len(needed_pids):,} passages candidats...")
    corpus_full = dataset.load_corpus()
    corpus = {pid: corpus_full[pid] for pid in needed_pids if pid in corpus_full}
    del corpus_full

    reranker = CrossEncoderReranker(
        model_name=args.model,
        model_path=args.model_path,
        max_length=args.max_length,
    )

    t0 = time.time()
    reranked_results = reranker.rerank_batch(
        bm25_results,
        corpus,
        queries,
        batch_size=args.batch_size,
        top_k=args.top_k_rerank,
    )
    latency_total = time.time() - t0
    latency_ms_per_query = latency_total / max(len(queries), 1) * 1000.0

    metrics = evaluate_run(qrels, reranked_results, ks=[10])
    print("\nMétriques Cross-Encoder :")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  Latence moyenne : {latency_ms_per_query:.1f} ms/requête")

    run_dir = PROJECT_ROOT / f"results/{args.run_name}"
    run_config = {
        "dataset": dataset_name,
        "split": split,
        "model": args.model,
        "model_path": args.model_path,
        "top_k_first": args.top_k_first,
        "top_k_rerank": args.top_k_rerank,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "latency_ms_per_query": latency_ms_per_query,
        "metrics_bm25": metrics_bm25,
    }
    save_run(
        run_dir=str(run_dir),
        metrics=metrics,
        results=reranked_results,
        qrels=qrels,
        queries=queries,
        corpus=corpus,
        config=run_config,
    )
    print(f"\nRun complet. Résultats dans : {run_dir}/")


if __name__ == "__main__":
    main()
