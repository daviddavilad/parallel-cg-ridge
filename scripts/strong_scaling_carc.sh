#!/usr/bin/env bash
set -euo pipefail

N=${1:-500000}
D=${2:-200}
OUT=${3:-results/strong_scaling_carc.jsonl}

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for P in 1 2 4 8 16 32; do
  echo "running P=$P" >&2
  OMP_NUM_THREADS=1 srun --mpi=pmi2 -n "$P" \
    python scripts/benchmark.py --n "$N" --d "$D" --reps 5 >> "$OUT"
done