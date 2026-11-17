#!/usr/bin/env bash
#SBATCH --job-name=cg-precond-qr-mode
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/precond_qr_mode_carc_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

OUT=results/precond_qr_mode_carc_${SLURM_JOB_ID}.jsonl
mkdir -p "$(dirname "$OUT")"

# Main comparison: both QR modes back to back at each P, d = 500
for N in 50000 500000; do
  for P in 1 2 4 8 16 32; do
    for MODE in replicated root; do
      echo "running N=$N P=$P qr=$MODE" >&2
      OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
        python scripts/benchmark.py --n "$N" --d 500 --cond 1e6 --lam 0 \
          --precond sparse --s_factor 2 --zeta 1 --qr "$MODE" --reps 5 \
          >> "$OUT" || echo "N=$N P=$P qr=$MODE failed" >&2
    done
  done
done

# Cache test: d = 300, replicated, n = 50000
for P in 1 2 4 8 16 32; do
  echo "running cache test d=300 P=$P" >&2
  OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
    python scripts/benchmark.py --n 50000 --d 300 --cond 1e6 --lam 0 \
      --precond sparse --s_factor 2 --zeta 1 --qr replicated --reps 5 \
      >> "$OUT" || echo "d=300 P=$P failed" >&2
done