import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # must come after

from mpi4py import MPI
from parallel_cg_ridge.cg import (
    cg, 
    pcg,
)
from parallel_cg_ridge.cg_mpi import (
    assemble_rhs,
    make_distributed_ridge_operator,
    make_split_ridge_operator,
    scatter_rows,
    make_distributed_sketch_preconditioner,
    allocate_node_shared,
)
from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.pipelined import pipelined_cg
from parallel_cg_ridge.precond import sparse_sign_sketch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = node_comm.Get_rank()

cpus: list[int]
if sys.platform == "linux":
    os.sched_setaffinity(0, {local_rank})
    cpus = sorted(os.sched_getaffinity(0))
else:
    cpus = []

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100_000)
parser.add_argument("--d", type=int, default=200)
parser.add_argument("--lam", type=float, default=1e-3)
parser.add_argument("--cond", type=float, default=100.0)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--reps", type=int, default=5)
parser.add_argument("--solver", choices=["cg", "pipelined"], default="cg")
parser.add_argument("--precond", choices=["none", "sparse"], default="none")
parser.add_argument("--s_factor", type=int, default=2)
parser.add_argument("--zeta", type=int, default=1)
parser.add_argument("--qr", choices=["replicated", "root", "shared"], default="replicated")

args = parser.parse_args()

if args.precond != "none" and args.solver == "pipelined":
    parser.error("preconditioning is only implemented for --solver cg")

if args.qr != "replicated" and args.precond == "none":
    parser.error("--qr root/shared requires --precond sparse")

if args.reps < 1 or args.warmup < 0:
    parser.error("--reps must be positive and --warmup non-negative")

s = args.s_factor * args.d if args.precond == "sparse" else 0

if args.precond == "sparse":
    if args.s_factor < 1:
        parser.error("--s_factor must be at least 1")
    if not (1 <= args.zeta <= s // 2):
        parser.error("--zeta must be between 1 and s // 2")

# Generate and distribute the problem
if rank == 0:
    X, y, _ = make_ridge_problem(n=args.n, d=args.d, cond=args.cond, seed=0)
else:
    X, y = None, None

X_local = scatter_rows(X, comm)
y_local = scatter_rows(y, comm)
del X, y

# Generate and distribute a fixed sparse sketch
if args.precond == "sparse":
    if rank == 0:
        positions, signs = sparse_sign_sketch(args.n, s, args.zeta, np.random.default_rng(1))
    else:
        positions, signs = None, None

    positions_local = scatter_rows(positions, comm)
    signs_local = scatter_rows(signs, comm)

    del positions, signs
else:
    positions_local = None
    signs_local = None

# Set up the matrix-free operator
if args.solver == "cg":
    apply_A, timings = make_distributed_ridge_operator(X_local, args.lam, comm)
else:
    start_A, finish_A, timings = make_split_ridge_operator(X_local, args.lam, comm)

b = assemble_rhs(X_local, y_local, comm)

# One-time shared-memory allocation
leader_comm = MPI.COMM_NULL
R_shared = None
win = None
shared_alloc = 0.0

if args.qr == "shared":
    leader_comm = comm.Split(0 if local_rank == 0 else MPI.UNDEFINED, key=rank)

    comm.Barrier()

    t0 = MPI.Wtime()
    R_shared, win = allocate_node_shared((args.d, args.d), node_comm)
    alloc_time = MPI.Wtime() - t0

    alloc_times = comm.gather(alloc_time, root=0)

    if rank == 0:
        shared_alloc = max(alloc_times)

# Preconditioner setup and solve
def setup():
    if args.precond == "none":
        return None, {"sketch": 0.0, "comm": 0.0, "qr": 0.0, "bcast": 0.0, "node_sync": 0.0}

    kwargs = {"qr": args.qr}

    if args.qr == "shared":
        kwargs.update({"R_shared": R_shared, "node_comm": node_comm, "leader_comm": leader_comm, "shared_win": win})

    return make_distributed_sketch_preconditioner(X_local, args.lam, s, positions_local, signs_local, comm, **kwargs)


def timed(f):
    """Time applications of the inverse preconditioner."""
    def wrapped(r):
        t0 = MPI.Wtime()
        out = f(r)
        timings["precond"] += MPI.Wtime() - t0
        return out

    return wrapped


def solve(apply_Minv):
    maxiter = 10 * args.d
    tol = 1e-10

    if args.solver == "pipelined":
        return pipelined_cg(start_A, finish_A, b, tol=tol, maxiter=maxiter)

    if apply_Minv is None:
        # Preserve the original unpreconditioned CG path.
        return cg(apply_A, b, tol=tol, maxiter=maxiter)

    return pcg(apply_A, timed(apply_Minv), b, tol=tol, maxiter=maxiter)


def reset_timings():
    timings.update(compute=0.0, comm=0.0, calls=0, precond=0.0)

# Warmup
for _ in range(args.warmup):
    reset_timings()
    apply_Minv, _ = setup()
    solve(apply_Minv)

cpus_per_rank = comm.gather(cpus, root=0)

# Timed repetitions
records = []

for _ in range(args.reps):
    reset_timings()

    # Phase 1: preconditioner setup
    comm.Barrier()

    t0 = MPI.Wtime()
    apply_Minv, setup_timings = setup()
    setup_wall = MPI.Wtime() - t0

    if args.precond == "none":
        setup_wall = 0.0

    # Phase 2: iterative solve
    comm.Barrier()

    t0 = MPI.Wtime()
    w, iters, res, converged = solve(apply_Minv)
    solve_wall = MPI.Wtime() - t0

    # Gather rank-local measurements.
    setup_walls = comm.gather(setup_wall, root=0)
    solve_walls = comm.gather(solve_wall, root=0)

    comps = comm.gather(timings["compute"], root=0)
    comms = comm.gather(timings["comm"], root=0)
    precond_times = comm.gather(timings["precond"], root=0)

    sketches = comm.gather(setup_timings["sketch"], root=0)
    setup_comms = comm.gather(setup_timings["comm"], root=0)
    qrs = comm.gather(setup_timings["qr"], root=0)
    bcasts = comm.gather(setup_timings["bcast"], root=0)
    node_syncs = comm.gather(setup_timings["node_sync"], root=0)

    iteration_counts = comm.gather(iters, root=0)
    convergence_flags = comm.gather(converged, root=0)

    if rank == 0:
        max_setup = max(setup_walls)
        max_solve = max(solve_walls)

        records.append({
            "wall": max_solve,
            "setup_wall": max_setup,
            "total_wall": max_setup + max_solve,
            "compute": max(comps),
            "comm": max(comms),
            "precond_time": max(precond_times),
            "sketch": max(sketches),
            "setup_comm": max(setup_comms),
            "qr": max(qrs),
            "bcast": max(bcasts),
            "node_sync": max(node_syncs),
            "compute_per_rank": comps,
            "comm_per_rank": comms,
            "iters": max(iteration_counts),
            "converged": all(convergence_flags),
            "ranks_agree": len(set(iteration_counts)) == 1,
        })

# Summary output
if rank == 0:
    def median(field):
        return float(np.median([r[field] for r in records]))

    result = {
        "P": size,
        "n": args.n,
        "d": args.d,
        "lam": args.lam,
        "cond": args.cond,
        "reps": args.reps,
        "solver": args.solver,
        "precond": args.precond,
        "s": s,
        "zeta": args.zeta if args.precond == "sparse" else None,
        "qr_mode": args.qr if args.precond == "sparse" else None,
        "shared_alloc": float(shared_alloc),

        # Solve-time metrics (existing meaning preserved)
        "wall": median("wall"),
        "compute": median("compute"),
        "comm": median("comm"),

        # New setup and preconditioning metrics
        "setup_wall": median("setup_wall"),
        "total_wall": median("total_wall"),
        "sketch": median("sketch"),
        "setup_comm": median("setup_comm"),
        "qr": median("qr"),
        "bcast": median("bcast"),
        "node_sync": median("node_sync"),
        "precond_time": median("precond_time"),

        "compute_per_rank": np.median([r["compute_per_rank"] for r in records], axis=0).tolist(),
        "comm_per_rank": np.median([r["comm_per_rank"] for r in records], axis=0).tolist(),

        "iters": records[0]["iters"],
        "converged": all(r["converged"] for r in records),
        "ranks_agree": all(r["ranks_agree"] for r in records),
        "cpus_per_rank": cpus_per_rank,
    }

    print(json.dumps(result))

if win is not None:
    win.Unlock_all()
    win.Free()

if leader_comm != MPI.COMM_NULL:
    leader_comm.Free()

node_comm.Free()
