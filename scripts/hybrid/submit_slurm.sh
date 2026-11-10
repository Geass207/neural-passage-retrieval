#!/bin/bash
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_rrf
#SBATCH --output=results/hybrid/evaluation/slurm-%j.out

# Chargement du module du cours
# module load <your-site-modules>

# Configuration du cache HuggingFace
export HF_HOME=$SLURM_TMPDIR/huggingface_cache

# Création du dossier de sortie global si besoin
mkdir -p results/hybrid/evaluation/

# Lancement de l'évaluation avec les deux fichiers JSON requis
python scripts/run_rrf_eval.py \
    --bm25-run results/baselines/bm25_msmarco_dev/run.json \
    --dense-run results/dense/evaluation/dense_bge_base_ft_subset10_msmarco_dev/run.json