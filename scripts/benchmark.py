import argparse
import numpy as np
import json

from mpi4py import MPI
from parallel_cg_ridge.cg import cg
from parallel_cg_ridge.cg_mpi import (
    assemble_rhs,
    make_distributed_ridge_operator,
    row_partition,
)
from parallel_cg_ridge.data import make_ridge_problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100_000)
parser.add_argument("--d", type=int, default=200)
parser.add_argument("--lam", type=float, default=1e-3)
parser.add_argument("--cond", type=float, default=100.0)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

X, y, _ = make_ridge_problem(n=args.n, d=args.d, cond=args.cond, seed=0)

start, stop = row_partition(args.n, comm)
X_local = X[start:stop]
y_local = y[start:stop]

apply_A, timings = make_distributed_ridge_operator(X_local, args.lam, comm)
b = assemble_rhs(X_local, y_local, comm)

for _ in range(args.warmup):
    timings.update(compute=0.0, comm=0.0, calls=0)
    cg(apply_A, b, tol=1e-10, maxiter=10 * args.d)

comm.Barrier()

records = []

for _ in range(args.reps):
    timings.update(compute=0.0, comm=0.0, calls=0)
    comm.Barrier()

    t0 = MPI.Wtime()
    w, iters, res, converged = cg(apply_A, b, tol=1e-10, maxiter=10 * args.d)
    t1 = MPI.Wtime()

    wall = t1 - t0
    wall_max = comm.reduce(wall, op=MPI.MAX, root=0)
    comp_max = comm.reduce(timings["compute"], op=MPI.MAX, root=0)
    comm_max = comm.reduce(timings["comm"], op=MPI.MAX, root=0)

    if rank == 0:
        records.append({
            "wall": wall_max,
            "compute": comp_max,
            "comm": comm_max,
            "iters": iters,
            "converged": converged,
        })

if rank == 0:
    result = {
        "P": size,
        "n": args.n,
        "d": args.d,
        "lam": args.lam,
        "cond": args.cond,
        "reps": args.reps,
        "wall": float(np.median([r["wall"] for r in records])),
        "compute": float(np.median([r["compute"] for r in records])),
        "comm": float(np.median([r["comm"] for r in records])),
        "iters": records[0]["iters"],
        "converged": all(r["converged"] for r in records),
    }
    print(json.dumps(result))