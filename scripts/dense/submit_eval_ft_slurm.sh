#!/bin/bash
# Job SLURM — évaluation du bi-encoder fine-tuné BGE-base sur subset 10% MS MARCO
# Soumettre avec : sbatch scripts/dense/submit_eval_ft_slurm.sh

#SBATCH --job-name=bge-ft-eval
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bge-ft-eval-%j.out
#SBATCH --error=logs/bge-ft-eval-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=============================="
echo "Job BGE fine-tuned eval | Début : $(date)"
echo "=============================="

python3 - <<'EOF'
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')

from src.data.dataset_factory import get_dataset
from src.evaluation.metrics import evaluate_run
from src.retrieval.dense.dense_retriever import DenseRetriever

dataset = get_dataset('msmarco', data_dir='data/')
corpus, queries, qrels = dataset.get_split('dev')
queries = {qid: q for qid, q in queries.items() if qid in qrels}

random.seed(42)
subset_pids = random.sample(list(corpus.keys()), len(corpus) // 10)
corpus = {pid: corpus[pid] for pid in subset_pids}
print(f'Corpus subset : {len(corpus):,} passages | Requêtes : {len(queries):,}')

retriever = DenseRetriever(
    model_name='BAAI/bge-base-en-v1.5',
    model_path='results/dense/training/bge_base_biencoder_ft/best_checkpoint',
    embeddings_cache='/tmp/bge_base_ft_subset10_embeddings.npz'
)
retriever.index(corpus)

qids = list(queries.keys())
query_embs = retriever.model.encode(
    [queries[qid].text for qid in qids],
    batch_size=64,
    convert_to_numpy=True,
    show_progress_bar=True,
)
query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

print('Classement dense fine-tuné en batch...')
results = {}
BATCH = 200
for i in range(0, len(qids), BATCH):
    batch_qids = qids[i:i + BATCH]
    scores = query_embs[i:i + BATCH] @ retriever.corpus_embeddings.T
    for j, qid in enumerate(batch_qids):
        top_idx = np.argpartition(-scores[j], 100)[:100]
        top_idx = top_idx[np.argsort(-scores[j][top_idx])]
        results[qid] = [
            (retriever.corpus_pids[idx], float(scores[j][idx]))
            for idx in top_idx
        ]

metrics = evaluate_run(qrels, results, ks=[10, 50, 100])
print('\n=== Métriques BGE-base fine-tuné (subset 10%) ===')
for key, value in metrics.items():
    print(f'  {key}: {value:.4f}')

run_dir = Path('results/dense/evaluation/dense_bge_base_ft_subset10_msmarco_dev')
run_dir.mkdir(parents=True, exist_ok=True)

config = {
    'dataset': 'msmarco',
    'split': 'dev',
    'model': 'BAAI/bge-base-en-v1.5',
    'model_path': 'results/dense/training/bge_base_biencoder_ft/best_checkpoint',
    'subset_pct': 0.1,
    'seed': 42,
    'n_queries': len(queries),
    'n_corpus': len(corpus),
}

json.dump(results, open(run_dir / 'run.json', 'w'))
json.dump(metrics, open(run_dir / 'metrics.json', 'w'), indent=2)
json.dump(config, open(run_dir / 'config.json', 'w'), indent=2)
print(f'\nArtefacts sauvegardés dans {run_dir}/')
EOF

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
