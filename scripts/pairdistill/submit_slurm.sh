#!/bin/bash
# Job SLURM — PairDistill BM25-teacher sur MS MARCO (protocole corrigé)
# Entraînement sur le split train, évaluation sur le split dev complet
# Soumettre avec : sbatch scripts/pairdistill/submit_slurm.sh
# Surveiller     : sq
# Annuler        : scancel <JobID>

#SBATCH --job-name=pairdistill-train
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pairdistill-train-%j.out
#SBATCH --error=logs/pairdistill-train-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers ir_datasets faiss-cpu python-dotenv --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

mkdir -p $HOME/tmp $HOME/.ir_datasets
export TMPDIR=$HOME/tmp
export IR_DATASETS_HOME=$HOME/.ir_datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache HuggingFace vers $SLURM_TMPDIR (espace local du nœud)
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
mkdir -p $HF_HOME

echo "=============================="
echo "Job PairDistill | Début : $(date)"
echo "Node            : $SLURM_NODELIST"
echo "=============================="

# Protocole corrigé :
#   --train-split train  → entraînement sur les queries MS MARCO train (pas dev)
#   --eval-split dev     → évaluation sur les 6980 queries dev standard
#   --max-train-queries  → 10 000 queries train (sur ~500K dispo)
#   pas de --max-corpus  → corpus complet pour une évaluation comparable aux autres
#   --teacher-index-path → réutilise l'index BM25 déjà construit (évite de tout rebuilder)
python scripts/run_pairdistill_bm25.py \
    --train-split        train \
    --eval-split         dev \
    --max-train-queries  10000 \
    --teacher-index-path data/msmarco/bm25_index.pkl \
    --epochs             2 \
    --batch-size         16 \
    --encode-batch-size  256 \
    --teacher-top-k      100 \
    --top-k              100 \
    --seed               42 \
    --output-dir         results/pairdistill/training/bge_small_ft \
    --run-dir            results/pairdistill/evaluation/pairdistill_bge_small_msmarco_dev

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
