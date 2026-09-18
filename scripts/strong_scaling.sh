#!/usr/bin/env bash
# scripts/sweep.sh
set -euo pipefail

N=${1:-100000}
D=${2:-200}
OUT=${3:-results/strong_scaling.jsonl}

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for P in 1 2 4 8 16; do
  echo "running P=$P" >&2
  OMP_NUM_THREADS=1 mpirun -n "$P" --oversubscribe \
    uv run python scripts/benchmark.py --n "$N" --d "$D" --reps 5 >> "$OUT"
done