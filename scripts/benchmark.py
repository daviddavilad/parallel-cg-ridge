import argparse
import numpy as np
import json
import os
import sys

from mpi4py import MPI
from parallel_cg_ridge.cg import cg
from parallel_cg_ridge.cg_mpi import (
    assemble_rhs,
    make_distributed_ridge_operator,
    scatter_rows,
)
from parallel_cg_ridge.data import make_ridge_problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cpus: list[int]
if sys.platform == "linux":
    os.sched_setaffinity(0, {rank})
    cpus = sorted(os.sched_getaffinity(0))
else:
    cpus = []

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100_000)
parser.add_argument("--d", type=int, default=200)
parser.add_argument("--lam", type=float, default=1e-3)
parser.add_argument("--cond", type=float, default=100.0)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--reps", type=int, default=5)
args = parser.parse_args()

if rank == 0:
    X, y, _ = make_ridge_problem(n=args.n, d=args.d, cond=args.cond, seed=0)
else:
    X, y = None, None

X_local = scatter_rows(X, comm)
y_local = scatter_rows(y, comm)
del X, y

apply_A, timings = make_distributed_ridge_operator(X_local, args.lam, comm)
b = assemble_rhs(X_local, y_local, comm)

for _ in range(args.warmup):
    timings.update(compute=0.0, comm=0.0, calls=0)
    cg(apply_A, b, tol=1e-10, maxiter=10 * args.d)

cpus_per_rank = comm.gather(cpus, root=0)

comm.Barrier()

records = []

for _ in range(args.reps):
    timings.update(compute=0.0, comm=0.0, calls=0)
    comm.Barrier()

    t0 = MPI.Wtime()
    w, iters, res, converged = cg(apply_A, b, tol=1e-10, maxiter=10 * args.d)
    t1 = MPI.Wtime()

    wall = t1 - t0
    walls = comm.gather(wall, root=0)
    comps = comm.gather(timings["compute"], root=0)
    comms = comm.gather(timings["comm"], root=0)

    if rank == 0:
        records.append({
            "wall": max(walls),
            "compute": max(comps),
            "comm": max(comms),
            "compute_per_rank": comps,
            "comm_per_rank": comms,
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
        "compute_per_rank": np.median([r["compute_per_rank"] for r in records], axis=0).tolist(),
        "comm_per_rank": np.median([r["comm_per_rank"] for r in records], axis=0).tolist(),
        "iters": records[0]["iters"],
        "converged": all(r["converged"] for r in records),
        "cpus_per_rank": cpus_per_rank,
    }
    print(json.dumps(result))