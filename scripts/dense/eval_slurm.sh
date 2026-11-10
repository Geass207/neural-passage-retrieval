#!/bin/bash
# Évaluation BGE-large off-the-shelf sur le subset 10% de MS MARCO
# Prérequis : transférer les embeddings
#
# Usage : bash scripts/dense/eval_slurm.sh

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers --user --quiet

cd "$(dirname "$0")/../.."

python3 - <<'EOF'
import sys, json, numpy as np, random
from pathlib import Path
sys.path.insert(0, '.')

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.retrieval.dense.dense_retriever import DenseRetriever

# --- Données ---
dataset = get_dataset('msmarco', data_dir='data/')
corpus, queries, qrels = dataset.get_split('dev')
queries = {qid: q for qid, q in queries.items() if qid in qrels}

random.seed(42)
subset_pids = random.sample(list(corpus.keys()), len(corpus) // 10)
corpus = {pid: corpus[pid] for pid in subset_pids}
print(f'Corpus subset : {len(corpus):,} passages | Requêtes : {len(queries):,}')

# --- Chargement des embeddings (déjà calculés) ---
retriever = DenseRetriever(
    model_name='BAAI/bge-large-en-v1.5',
    embeddings_cache='data/msmarco/dense_embeddings_bge-large-en-v1.5.npz'
)
retriever.index(corpus)

# --- Encodage des requêtes ---
qids = list(queries.keys())
query_embs = retriever.model.encode(
    [queries[qid].text for qid in qids],
    batch_size=64, convert_to_numpy=True, show_progress_bar=True
)
query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

# --- Retrieval en batch (évite OOM) ---
print('Classement dense...')
results = {}
BATCH = 200
for i in range(0, len(qids), BATCH):
    batch_qids = qids[i:i+BATCH]
    scores = query_embs[i:i+BATCH] @ retriever.corpus_embeddings.T
    for j, qid in enumerate(batch_qids):
        top_idx = np.argpartition(-scores[j], 100)[:100]
        top_idx = top_idx[np.argsort(-scores[j][top_idx])]
        results[qid] = [(retriever.corpus_pids[idx], float(scores[j][idx])) for idx in top_idx]

# --- Métriques ---
metrics = evaluate_run(qrels, results, ks=[10, 50, 100])
print('\n=== Métriques BGE off-the-shelf (subset 10%) ===')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')

# --- Sauvegarde ---
Path('results/dense_msmarco_dev').mkdir(parents=True, exist_ok=True)
json.dump(results, open('results/dense_msmarco_dev/run.json', 'w'))
json.dump(metrics, open('results/dense_msmarco_dev/metrics.json', 'w'), indent=2)
print('\nArtefacts sauvegardés dans results/dense_msmarco_dev/')
EOF
