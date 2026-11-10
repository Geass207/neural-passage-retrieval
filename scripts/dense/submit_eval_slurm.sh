#!/bin/bash
# Job SLURM — évaluation BGE-large off-the-shelf (subset 10% MS MARCO)
# Prérequis : embeddings déjà transférés
# Soumettre avec : sbatch scripts/dense/submit_eval_slurm.sh

#SBATCH --job-name=bge-eval
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dense-eval-%j.out
#SBATCH --error=logs/dense-eval-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=============================="
echo "Job BGE eval | Début : $(date)"
echo "=============================="

# Copie le .npz vers le stockage local du nœud (évite les blocages CephFS)
echo "Copie des embeddings vers /tmp..."
cp data/msmarco/dense_embeddings_bge-large-en-v1.5.npz /tmp/bge_embeddings.npz
echo "Copie terminée."

python3 - <<'EOF'
import sys, json, numpy as np, random
from pathlib import Path
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
    model_name='BAAI/bge-large-en-v1.5',
    embeddings_cache='/tmp/bge_embeddings.npz'  # stockage local, pas réseau
)
retriever.index(corpus)

qids = list(queries.keys())
query_embs = retriever.model.encode(
    [queries[qid].text for qid in qids],
    batch_size=64, convert_to_numpy=True, show_progress_bar=True
)
query_embs = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

print('Classement dense en batch...')
results = {}
BATCH = 200
for i in range(0, len(qids), BATCH):
    batch_qids = qids[i:i+BATCH]
    scores = query_embs[i:i+BATCH] @ retriever.corpus_embeddings.T
    for j, qid in enumerate(batch_qids):
        top_idx = np.argpartition(-scores[j], 100)[:100]
        top_idx = top_idx[np.argsort(-scores[j][top_idx])]
        results[qid] = [(retriever.corpus_pids[idx], float(scores[j][idx])) for idx in top_idx]

metrics = evaluate_run(qrels, results, ks=[10, 50, 100])
print('\n=== Métriques BGE off-the-shelf (subset 10%) ===')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')

Path('results/dense_msmarco_dev').mkdir(parents=True, exist_ok=True)
json.dump(results, open('results/dense_msmarco_dev/run.json', 'w'))
json.dump(metrics, open('results/dense_msmarco_dev/metrics.json', 'w'), indent=2)
print('\nArtefacts sauvegardés dans results/dense_msmarco_dev/')
EOF

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
