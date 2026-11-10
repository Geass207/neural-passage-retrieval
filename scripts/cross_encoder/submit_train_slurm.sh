#!/bin/bash
# Job SLURM — fine-tuning du cross-encoder BGE-reranker-v2-m3 sur MS MARCO
# Soumettre avec : sbatch scripts/cross_encoder/submit_train_slurm.sh
# Surveiller     : sq
# Annuler        : scancel <JobID>

#SBATCH --job-name=bge-reranker-train
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bge-reranker-train-%j.out
#SBATCH --error=logs/bge-reranker-train-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers ir_datasets python-dotenv --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ir_datasets télécharge les triplets MS MARCO — /tmp trop petit, on redirige
mkdir -p $HOME/tmp $HOME/.ir_datasets
export TMPDIR=$HOME/tmp
export IR_DATASETS_HOME=$HOME/.ir_datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache HuggingFace vers $SLURM_TMPDIR (espace local du nœud, créé automatiquement par SLURM)
export HF_HOME=$SLURM_TMPDIR/huggingface_cache
mkdir -p $HF_HOME

echo "=============================="
echo "Job BGE reranker train | Début : $(date)"
echo "Node                   : $SLURM_NODELIST"
echo "=============================="

# 5000 triplets → 10 000 paires (5000 pos + 5000 neg), 1 époque
# batch_size=2, max_seq_length=256 pour tenir dans la mémoire GPU
python scripts/train_cross_encoder.py \
    --model              BAAI/bge-reranker-base \
    --max_train_triplets 5000 \
    --epochs             1 \
    --batch_size         2 \
    --lr                 2e-5 \
    --max_seq_length     256 \
    --seed               42 \
    --output_dir         results/cross_encoder/training/bge_reranker_ft

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
