#!/usr/bin/env bash
#SBATCH --job-name=cg-weak-scaling
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/weak_scaling_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

ROWS_PER_RANK=${1:-15625}
D=${2:-200}
OUT=${3:-results/weak_scaling_carc_${SLURM_JOB_ID}.jsonl}

mkdir -p "$(dirname "$OUT")"

for P in 1 2 4 8 16 32; do
  N=$(( ROWS_PER_RANK * P ))
  echo "running P=$P" >&2
  OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
    python scripts/benchmark.py --n "$N" --d "$D" --reps 20 >> "$OUT" || echo "P=$P failed" >&2
done