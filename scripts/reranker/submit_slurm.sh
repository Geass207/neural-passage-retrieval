#!/bin/bash
# Script SLURM pour un cluster SLURM
# Soumettre avec : sbatch scripts/reranker/submit_slurm.sh
# Surveiller     : sq
# Annuler        : scancel <JobID>

#SBATCH --job-name=mistral-reranker
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reranker-%j.out
#SBATCH --error=logs/reranker-%j.err

# module load <your-site-modules>
module load gcc arrow

# Packages supplémentaires non inclus dans le module du cours
pip install ir_datasets bm25s python-dotenv --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# ir_datasets télécharge ~8Go — /tmp est trop petit, on redirige vers $HOME
mkdir -p $HOME/tmp $HOME/.ir_datasets
export TMPDIR=$HOME/tmp
export IR_DATASETS_HOME=$HOME/.ir_datasets

echo "=============================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Début    : $(date)"
echo "=============================="

# --- Entraînement — ablations sur le rang LoRA ---
# batch_size=1 + grad_accum=32 → batch effectif = 32
# max_length=256 pour réduire la VRAM

for rank in 8 4 16; do
    echo "[train] LoRA rank=$rank..."
    python scripts/reranker/train.py \
        --lora_rank         $rank \
        --epochs            1 \
        --batch_size        1 \
        --grad_accum        32 \
        --max_length        256 \
        --max_train_samples 10000 \
        --seed              42
done

# --- Évaluation ---
# top_k_first=50 (au lieu de 100) + batch_size=8 → ~4× plus rapide
# Estimation : ~5s/requête × 6980 = ~10h par run, 4 runs = ~17h < 24h

for rank in 4 8 16; do
    echo "[eval] rank=$rank..."
    python scripts/reranker/eval.py \
        --adapter_path results/mistral/training/mistral_lora_r${rank}/best_checkpoint \
        --lora_rank    $rank \
        --top_k_first  50 \
        --batch_size   8 \
        --max_length   256
done

echo "[eval] zero-shot..."
python scripts/reranker/eval.py \
    --zero_shot \
    --top_k_first 50 \
    --batch_size  8 \
    --max_length  256

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
