import argparse
import json
import numpy as np

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.operators import make_ridge_operator
from parallel_cg_ridge.cg import cg

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20_000)
parser.add_argument("--d", type=int, default=500)
parser.add_argument("--cond", type=float, default=1e6)
parser.add_argument("--maxiter_multiplier", type=int, default=20)
args = parser.parse_args()

X, y, _ = make_ridge_problem(n=args.n, d=args.d, cond=args.cond, seed=0)
ev = np.linalg.eigvalsh(X.T @ X)
b = X.T @ y

lams = list(np.logspace(0, -9, 19)) + [0.0]

for lam in lams:
    kappa = (ev.max() + lam) / (ev.min() + lam)
    apply_A = make_ridge_operator(X, lam)
    x, iters, rel_res, converged = cg(apply_A, b, tol=1e-10, maxiter=args.maxiter_multiplier * args.d)
    true_res = np.linalg.norm(b - apply_A(x)) / np.linalg.norm(b)
    print(json.dumps({
        "lambda": float(lam),
        "kappa": float(kappa),
        "iters": iters,
        "rel_res": float(rel_res),
        "true_res": float(true_res),
        "converged": converged,
        "n": args.n, "d": args.d, "cond": args.cond,
    }))