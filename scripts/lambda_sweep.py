import argparse
import json
import numpy as np
import time

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.operators import make_ridge_operator
from parallel_cg_ridge.cg import (
    pcg,
)
from parallel_cg_ridge.precond import make_sketch_preconditioner

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20_000)
parser.add_argument("--d", type=int, default=500)
parser.add_argument("--cond", type=float, default=1e6)
parser.add_argument("--maxiter_multiplier", type=int, default=20)
parser.add_argument("--precond", choices=["none", "gaussian"], default="none")
parser.add_argument("--s_factor", type=int, default=4)
args = parser.parse_args()

if args.precond == "gaussian" and args.s_factor < 1:
    parser.error("--s_factor must be at least 1")

X, y, _ = make_ridge_problem(n=args.n, d=args.d, cond=args.cond, seed=0)
ev = np.linalg.eigvalsh(X.T @ X)
b = X.T @ y

lams = list(np.logspace(0, -9, 19)) + [0.0]

for lam in lams:
    kappa = (ev.max() + lam) / (ev.min() + lam)
    apply_A = make_ridge_operator(X, lam)

    # Preconditioner setup
    setup_time = 0.0
    s = 0

    if args.precond == "gaussian":
        s = args.s_factor * args.d
        t0 = time.perf_counter()
        apply_Minv = make_sketch_preconditioner(X, lam, s, seed=1)
        setup_time = time.perf_counter() - t0

    else:
        apply_Minv = lambda r: r.copy()

    # Time only the iterative solve
    t0 = time.perf_counter()

    x, iters, rel_res, converged = pcg(apply_A, apply_Minv, b, tol=1e-10, maxiter=args.maxiter_multiplier * args.d)
    solve_time = time.perf_counter() - t0

    # Verify the true residual (outside the solve timer)
    true_res = np.linalg.norm(b - apply_A(x)) / np.linalg.norm(b)

    print(json.dumps({
        "lambda": float(lam),
        "kappa": float(kappa),
        "iters": iters,
        "rel_res": float(rel_res),
        "true_res": float(true_res),
        "converged": converged,
        "n": args.n,
        "d": args.d,
        "cond": args.cond,
        "precond": args.precond,
        "s": s,
        "setup_time": setup_time,
        "solve_time": solve_time,
    }))