#!/usr/bin/env bash
#SBATCH --job-name=cg-strong-scaling
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/strong_scaling_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

N=${1:-500000}
D=${2:-200}
OUT=${3:-results/strong_scaling_carc.jsonl}

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for P in 1 2 4 8 16 32; do
  echo "running P=$P" >&2
  OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
    python scripts/benchmark.py --n "$N" --d "$D" --reps 5 >> "$OUT" || echo "P=$P failed" >&2
done