#!/bin/bash
# Job SLURM dédié au BM25 baseline — CPU seulement, beaucoup de RAM
# Soumettre avec : sbatch scripts/reranker/submit_bm25_slurm.sh
# À faire UNE SEULE FOIS avant le job Mistral

#SBATCH --job-name=bm25-baseline
#SBATCH --partition=batch_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bm25-%j.out
#SBATCH --error=logs/bm25-%j.err

# module load <your-site-modules>
module load gcc arrow

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=============================="
echo "Job BM25 | Début : $(date)"
echo "=============================="

# Supprime l'index corrompu si présent
rm -f data/msmarco/bm25_index.pkl

python scripts/run_bm25_eval.py

echo "=============================="
echo "BM25 terminé : $(date)"
echo "=============================="
