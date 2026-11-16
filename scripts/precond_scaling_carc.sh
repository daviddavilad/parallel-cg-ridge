#!/usr/bin/env bash
#SBATCH --job-name=cg-precond-scaling
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/precond_scaling_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

OUT=results/precond_scaling_carc_${SLURM_JOB_ID}.jsonl
mkdir -p "$(dirname "$OUT")"

for N in 50000 500000; do
  for P in 1 2 4 8 16 32; do
    echo "running N=$N P=$P" >&2
    OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
      python scripts/benchmark.py --n "$N" --d 500 --cond 1e6 --lam 0 \
        --precond sparse --s_factor 2 --zeta 1 --reps 5 >> "$OUT" || echo "N=$N P=$P failed" >&2
  done
done