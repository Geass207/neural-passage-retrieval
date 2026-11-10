"""
Évaluation du reranker Mistral-7B LoRA sur MS MARCO dev.

Workflow :
  1. BM25 récupère les top-100 candidats (premier étage)
  2. Mistral reranke les 100 candidats (second étage)
  3. Métriques calculées aux deux étages pour comparer l'apport du reranker

Utilisation :
    # Avec le modèle fine-tuné
    python scripts/reranker/eval.py --adapter_path results/mistral/training/mistral_lora_r8/best_checkpoint

    # Zero-shot (modèle de base sans fine-tuning, pour comparaison)
    python scripts/reranker/eval.py --zero_shot

    # Smoke test local sans GPU
    python scripts/eval_mistral_reranker.py --smoke_test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.evaluation.results_io import save_run
from src.retrieval.bm25.bm25_retriever import BM25Retriever
from src.retrieval.reranker.mistral_lora_reranker import MistralLoRAReranker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--adapter_path",  default=None,
                   help="Chemin vers le checkpoint LoRA fine-tuné")
    p.add_argument("--zero_shot",     action="store_true",
                   help="Évalue le modèle de base sans adaptateur (baseline zero-shot)")
    p.add_argument("--top_k_first",   type=int, default=100,
                   help="Nombre de candidats récupérés par le premier étage (BM25)")
    p.add_argument("--top_k_rerank",  type=int, default=10,
                   help="Nombre de passages conservés après reranking")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--max_length",    type=int, default=512)
    p.add_argument("--lora_rank",     type=int, default=8,
                   help="Rang LoRA utilisé — sert uniquement pour nommer le dossier de résultats")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Évalue sur 50 requêtes seulement avec opt-125m")
    p.add_argument("--max_queries",   type=int, default=None,
                   help="Limite le nombre de requêtes évaluées (ex: 500 pour aller vite)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    split        = cfg["dataset"]["split"]
    data_dir     = cfg["dataset"]["data_dir"]
    index_path   = cfg["bm25"]["index_path"]

    # --- Chargement des requêtes et qrels (léger) ---
    print(f"=== Évaluation Mistral-7B LoRA | Dataset : {dataset_name} | Split : {split} ===")
    dataset  = get_dataset(dataset_name, data_dir=data_dir)
    _, queries, qrels = dataset.get_split(split)  # corpus chargé plus bas si nécessaire

    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    if args.smoke_test:
        subset_qids = list(queries.keys())[:50]
        queries = {qid: queries[qid] for qid in subset_qids}
        qrels   = {qid: qrels[qid]   for qid in subset_qids if qid in qrels}
    elif args.max_queries:
        import random as _random
        _random.seed(42)
        subset_qids = _random.sample(list(queries.keys()), min(args.max_queries, len(queries)))
        queries = {qid: queries[qid] for qid in subset_qids}
        qrels   = {qid: qrels[qid]   for qid in subset_qids if qid in qrels}

    print(f"Requêtes : {len(queries):,}")

    # --- Premier étage : résultats BM25 pré-calculés ---
    # On évite de charger les 8,8M passages en mémoire — trop de RAM.
    # Le job submit_bm25_slurm.sh a sauvegardé run.json avec les top-100 par requête.
    import json as _json
    bm25_run_dir      = f"results/baselines/bm25_{dataset_name}_{split}"
    bm25_results_path = Path(bm25_run_dir) / "run.json"

    if not bm25_results_path.exists():
        raise FileNotFoundError(
            f"Résultats BM25 introuvables : {bm25_results_path}\n"
            "Lance d'abord : sbatch scripts/reranker/submit_bm25_slurm.sh"
        )

    print(f"\n[Étage 1] Chargement résultats BM25 depuis {bm25_results_path}...")
    raw         = _json.loads(bm25_results_path.read_text())
    # Filtre aux top_k_first candidats — réduit le nombre de passages à scorer par Mistral
    bm25_results = {qid: [tuple(p) for p in pairs[:args.top_k_first]] for qid, pairs in raw.items()}

    metrics_bm25 = evaluate_run(qrels, bm25_results, ks=[10, 50, 100])
    print("Métriques BM25 :")
    for name, value in metrics_bm25.items():
        print(f"  {name}: {value:.4f}")

    # --- Chargement du sous-corpus nécessaire pour Mistral ---
    # On charge uniquement les passages présents dans le top-100 BM25 (~700k max)
    needed_pids = {pid for pairs in bm25_results.values() for pid, _ in pairs}
    print(f"\nChargement de {len(needed_pids):,} passages (top-100 BM25 uniquement)...")
    corpus_full = dataset.load_corpus()
    corpus = {pid: corpus_full[pid] for pid in needed_pids if pid in corpus_full}
    del corpus_full  # libère la mémoire immédiatement

    # --- Second étage : Mistral reranker ---
    print("\n[Étage 2] Mistral reranker...")
    base_model = "facebook/opt-125m" if args.smoke_test else "mistralai/Mistral-7B-Instruct-v0.2"
    adapter    = None if args.zero_shot else args.adapter_path

    reranker = MistralLoRAReranker(
        base_model=base_model,
        adapter_path=adapter,
        max_length=args.max_length,
    )
    reranker.load()

    t0 = time.time()
    reranked_results = reranker.rerank(
        bm25_results, corpus, queries,
        top_k=args.top_k_rerank,
        batch_size=args.batch_size,
    )
    latency_total = time.time() - t0
    latency_per_query = latency_total / max(len(queries), 1) * 1000  # en ms

    metrics_reranked = evaluate_run(qrels, reranked_results, ks=[10])
    print("\nMétriques Mistral reranker :")
    for name, value in metrics_reranked.items():
        print(f"  {name}: {value:.4f}")
    print(f"  Latence moyenne : {latency_per_query:.1f} ms/requête")

    # Delta par rapport au BM25 seul
    delta_mrr  = metrics_reranked["MRR@10"]  - metrics_bm25["MRR@10"]
    delta_ndcg = metrics_reranked["nDCG@10"] - metrics_bm25["nDCG@10"]
    print(f"\n  ΔnDCG@10 vs BM25 : {delta_ndcg:+.4f}")
    print(f"  ΔMRR@10  vs BM25 : {delta_mrr:+.4f}")

    # --- Sauvegarde des artefacts ---
    suffix = "zero_shot" if args.zero_shot else f"r{args.lora_rank}"
    run_dir = f"results/mistral/evaluation/mistral_{suffix}_{dataset_name}_{split}"

    run_config = {
        "dataset": dataset_name,
        "split": split,
        "base_model": base_model,
        "adapter_path": str(adapter),
        "zero_shot": args.zero_shot,
        "lora_rank": args.lora_rank,
        "top_k_first": args.top_k_first,
        "top_k_rerank": args.top_k_rerank,
        "batch_size": args.batch_size,
        "latency_ms_per_query": latency_per_query,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "metrics_bm25": metrics_bm25,
    }
    save_run(
        run_dir=run_dir,
        metrics=metrics_reranked,
        results=reranked_results,
        qrels=qrels,
        queries=queries,
        corpus=corpus,
        config=run_config,
    )
    print(f"\nRun complet. Résultats dans : {run_dir}/")


if __name__ == "__main__":
    main()
