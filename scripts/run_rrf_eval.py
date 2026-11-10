import json
import argparse
import os
from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.evaluation.results_io import save_run
from src.retrieval.hybrid.rrf_retriever import RRFRetriever

def load_json_run(file_path):
    """Charge un fichier run.json."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Évaluation de la fusion RRF (BM25 + Dense/Mistral)")
    parser.add_argument("--bm25-run", required=True, help="Chemin vers le fichier run.json de BM25")
    parser.add_argument("--dense-run", required=True, help="Chemin vers le fichier run.json du modèle dense (10%)")
    
    # NOUVEAU : Arguments optionnels pour tester d'autres modèles
    parser.add_argument("--mistral-run", required=False, help="Chemin vers le fichier run.json du modèle Mistral r8")
    parser.add_argument("--bge-large-run", required=False, help="Chemin vers le fichier run.json du modèle BGE-large")
    
    args = parser.parse_args()

    # 1. Chargement du dataset pour l'évaluation
    print("Chargement du dataset MS MARCO (dev)...")
    dataset = get_dataset("msmarco", data_dir="data/")
    corpus, queries, qrels = dataset.get_split("dev")

    # 2. Chargement des runs existants de base
    print("Chargement des résultats de base...")
    bm25_results = load_json_run(args.bm25_run)
    dense_results = load_json_run(args.dense_run)

    # Initialisation de notre Retriever RRF
    rrf_retriever = RRFRetriever()

    # Liste des ablations de base (BM25 + Dense 10%)
    ablations = [
        {"name": "bm25_only", "results": bm25_results, "config": {"method": "bm25_baseline"}},
        {"name": "dense_only", "results": dense_results, "config": {"method": "dense_subset10", "note": "ATTENTION: Dense est sur 10% du corpus"}},
        {"name": "rrf_bm25_dense_k30", "results": rrf_retriever.fuse(bm25_results, dense_results, k=30), "config": {"method": "hybrid_rrf", "k": 30, "note": "BM25 + Dense 10%"}},
        {"name": "rrf_bm25_dense_k60", "results": rrf_retriever.fuse(bm25_results, dense_results, k=60), "config": {"method": "hybrid_rrf", "k": 60, "note": "BM25 + Dense 10%"}},
        {"name": "rrf_bm25_dense_k100", "results": rrf_retriever.fuse(bm25_results, dense_results, k=100), "config": {"method": "hybrid_rrf", "k": 100, "note": "BM25 + Dense 10%"}}
    ]

    # 3. Ajout dynamique des tests Mistral si le fichier est fourni en argument
    if args.mistral_run and os.path.exists(args.mistral_run):
        print(f"Chargement des résultats Mistral depuis {args.mistral_run}...")
        mistral_results = load_json_run(args.mistral_run)
        ablations.extend([
            {"name": "mistral_only", "results": mistral_results, "config": {"method": "mistral_r8_baseline", "note": "Top 20 BM25"}},
            {"name": "rrf_bm25_mistral_k60", "results": rrf_retriever.fuse(bm25_results, mistral_results, k=60), "config": {"method": "hybrid_rrf", "k": 60, "note": "BM25 + Mistral r8 (Top 20)"}}
        ])

    # 4. Ajout dynamique des tests BGE-large si le fichier est fourni en argument
    if args.bge_large_run and os.path.exists(args.bge_large_run):
        print(f"Chargement des résultats BGE-large depuis {args.bge_large_run}...")
        bge_large_results = load_json_run(args.bge_large_run)
        ablations.extend([
            {"name": "bge_large_only", "results": bge_large_results, "config": {"method": "bge_large_baseline", "note": "30% corpus, 2094 queries"}},
            {"name": "rrf_bm25_bge_large_k60", "results": rrf_retriever.fuse(bm25_results, bge_large_results, k=60), "config": {"method": "hybrid_rrf", "k": 60, "note": "BM25 + BGE-large (30% corpus)"}}
        ])

    # 5. Évaluation et sauvegarde pour chaque ablation
    ks_to_eval = [10, 50, 100]
    
    for ablation in ablations:
        run_name = ablation["name"]
        print(f"\nÉvaluation de l'ablation : {run_name}")
        
        # Calcul des métriques (MRR, nDCG, Recall, etc.)
        metrics = evaluate_run(qrels, ablation["results"], ks=ks_to_eval)
        print(f"Métriques pour {run_name}: {metrics}")
        
        # Sauvegarde
        run_dir = os.path.join("results", "hybrid", "evaluation", run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        save_run(
            run_dir=run_dir,
            metrics=metrics,
            results=ablation["results"],
            qrels=qrels,
            queries=queries,
            corpus=corpus,
            config=ablation["config"]
        )
        print(f"Sauvegardé dans : {run_dir}")

if __name__ == "__main__":
    main()