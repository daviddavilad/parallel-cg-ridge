#!/usr/bin/env bash
#SBATCH --job-name=cg-precond-shared-r
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/precond_shared_r_carc_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

OUT=results/precond_shared_r_carc_${SLURM_JOB_ID}.jsonl
mkdir -p "$(dirname "$OUT")"

for P in 1 2 4 8 16 32; do
  for MODE in replicated root shared; do
    echo "running P=$P qr=$MODE" >&2
    OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
      python scripts/benchmark.py --n 50000 --d 500 --cond 1e6 --lam 0 \
        --precond sparse --s_factor 2 --zeta 1 --qr "$MODE" --reps 5 \
        >> "$OUT" || echo "P=$P qr=$MODE failed" >&2
  done
done