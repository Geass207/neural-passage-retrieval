#!/bin/bash
# Job SLURM — évaluation Mistral reranker (résultats principaux)
# Run principal (rank=8) + zero-shot sur 6980 requêtes, top_k_first=20
# Soumettre avec : sbatch scripts/reranker/submit_eval_only_slurm.sh

#SBATCH --job-name=mistral-eval
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install ir_datasets bm25s python-dotenv --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=============================="
echo "Job eval Mistral | Début : $(date)"
echo "=============================="

# --- Résultats principaux : 6980 requêtes, top_k_first=20 ---

echo "[eval] rank=8 — résultat principal (6980 requêtes)..."
python scripts/reranker/eval.py \
    --adapter_path results/mistral/training/mistral_lora_r8/best_checkpoint \
    --lora_rank    8 \
    --top_k_first  20 \
    --batch_size   8 \
    --max_length   256

echo "[eval] zero-shot — baseline LLM (6980 requêtes)..."
python scripts/reranker/eval.py \
    --zero_shot \
    --top_k_first 20 \
    --batch_size  8 \
    --max_length  256

# --- Ablations : 500 requêtes, rank 4 et 16 ---

echo "[ablation] rank=4 (500 requêtes)..."
python scripts/reranker/eval.py \
    --adapter_path results/mistral/training/mistral_lora_r4/best_checkpoint \
    --lora_rank    4 \
    --top_k_first  20 \
    --batch_size   8 \
    --max_length   256 \
    --max_queries  500

echo "[ablation] rank=16 (500 requêtes)..."
python scripts/reranker/eval.py \
    --adapter_path results/mistral/training/mistral_lora_r16/best_checkpoint \
    --lora_rank    16 \
    --top_k_first  20 \
    --batch_size   8 \
    --max_length   256 \
    --max_queries  500

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
