#!/bin/bash
# Job SLURM — fine-tuning léger BGE-large sur MS MARCO (bi-encoder)
# Soumettre avec : sbatch scripts/dense/submit_train_slurm.sh

#SBATCH --job-name=bge-train
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bge-train-%j.out
#SBATCH --error=logs/bge-train-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers ir_datasets --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# ir_datasets télécharge et met en cache les données
mkdir -p $HOME/tmp $HOME/.ir_datasets
export TMPDIR=$HOME/tmp
export IR_DATASETS_HOME=$HOME/.ir_datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================="
echo "Job BGE train | Début : $(date)"
echo "Node          : $SLURM_NODELIST"
echo "=============================="

python scripts/dense/train_biencoder.py \
    --model BAAI/bge-base-en-v1.5 \
    --max_train_samples 20000 \
    --epochs 1 \
    --batch_size 2 \
    --lr 2e-5 \
    --max_seq_length 128 \
    --seed 42 \
    --output_dir results/dense/training/bge_base_biencoder_ft \
    --save_training_examples

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
