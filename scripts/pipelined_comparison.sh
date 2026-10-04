#!/usr/bin/env bash
#SBATCH --job-name=cg-pipelined-comparison
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --mem=0             # request all node memory
#SBATCH --output=logs/pipelined_comparison_%j.out

module load python/3.13.0-xnav openmpi/4.1.6-2tgn
source .venv/bin/activate
set -euo pipefail

N=${1:-500000}
D=${2:-200}
OUT=${3:-results/pipelined_comparison_${SLURM_JOB_ID}.jsonl}
mkdir -p "$(dirname "$OUT")"

for P in 1 2 4 8 16 32; do
  for SOLVER in cg pipelined; do
    echo "running P=$P solver=$SOLVER" >&2
    OMP_NUM_THREADS=1 timeout 600 srun --mpi=pmi2 -n "$P" \
      python scripts/benchmark.py --n "$N" --d "$D" --reps 5 --solver "$SOLVER" >> "$OUT" \
      || echo "P=$P $SOLVER failed" >&2
  done
done