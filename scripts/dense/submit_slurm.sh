#!/bin/bash
# Script SLURM — BGE-large dense retrieval sur le corpus complet MS MARCO
# Soumettre avec : sbatch scripts/dense/submit_slurm.sh

#SBATCH --job-name=bge-dense
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dense-%j.out
#SBATCH --error=logs/dense-%j.err

# module load <your-site-modules>
module load gcc arrow

pip install sentence-transformers --user --quiet

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=============================="
echo "Job BGE dense | Début : $(date)"
echo "=============================="

# Évaluation BGE-large off-the-shelf sur corpus complet
python scripts/run_dense_eval.py

echo "=============================="
echo "Fin : $(date)"
echo "=============================="
