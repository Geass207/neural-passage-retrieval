#!/bin/bash
# Job SLURM — évaluation du cross-encoder BGE sur MS MARCO dev
# Zero-shot + fine-tuné, protocole aligné sur le reste du projet
# Soumettre avec : sbatch scripts/cross_encoder/submit_eval_slurm.sh
# Surveiller     : sq
# Annuler        : scancel <JobID>

#SBATCH --job-name=bge-reranker-eval
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bge-reranker-eval-%j.out
#SBATCH --error=logs/bge-reranker-eval-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers ir_datasets bm25s python-dotenv --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

mkdir -p $HOME/tmp $HOME/.ir_datasets
export TMPDIR=$HOME/tmp
export IR_DATASETS_HOME=$HOME/.ir_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache HuggingFace vers $SLURM_TMPDIR (espace local du nœud, créé automatiquement par SLURM)
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
mkdir -p $HF_HOME

echo "=============================="
echo "Job BGE reranker eval | Début : $(date)"
echo "Node                  : $SLURM_NODELIST"
echo "=============================="

# --- Zero-shot : 6980 requêtes, BM25 top-20 ---
echo "[eval] zero-shot — 6980 requêtes..."
python scripts/cross_encoder/eval.py \
    --model        BAAI/bge-reranker-base \
    --top_k_first  20 \
    --top_k_rerank 10 \
    --batch_size   32 \
    --max_length   256 \
    --run_name     cross_encoder/evaluation/bge_reranker_zs_msmarco_dev

# --- Fine-tuné : 6980 requêtes, BM25 top-20 ---
echo "[eval] fine-tuné — 6980 requêtes..."
python scripts/cross_encoder/eval.py \
    --model        BAAI/bge-reranker-base \
    --model_path   results/cross_encoder/training/bge_reranker_ft/best_checkpoint \
    --top_k_first  20 \
    --top_k_rerank 10 \
    --batch_size   32 \
    --max_length   256 \
    --run_name     cross_encoder/evaluation/bge_reranker_ft_msmarco_dev

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
