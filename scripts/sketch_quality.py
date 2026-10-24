import json

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.precond import (
    make_sketch_preconditioner,
    preconditioned_condition_number,
)

def main():
    n = 20_000
    d = 500
    s = 2 * d
    cond = 1e6
    lam = 0.0

    # Generate the design matrix once.
    X, _, _ = make_ridge_problem(n=n, d=d, cond=cond, seed=0)

    A = X.T @ X

    arms = [
        ("gaussian", "gaussian", None),
        ("sparse_zeta_1", "sparse", 1),
        ("sparse_zeta_2", "sparse", 2),
        ("sparse_zeta_4", "sparse", 4),
    ]

    for arm, kind, zeta in arms:
        for seed in range(1, 31):

            kwargs = {"kind": kind}

            if zeta is not None:
                kwargs["zeta"] = zeta

            apply_Minv = make_sketch_preconditioner(X, lam, s, seed=seed, **kwargs)

            kappa = preconditioned_condition_number(A, apply_Minv)

            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "kappa": float(kappa),
            }))

if __name__ == "__main__":
    main()