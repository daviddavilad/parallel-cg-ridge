import numpy as np
from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.precond import make_sketch_preconditioner

def test_sketch_preconditioner_reduces_conditioning():
    n, d, s, lam = 2000, 50, 200, 0.0
    X, _, _ = make_ridge_problem(n=n, d=d, cond=1e6, seed=0)
    A = X.T @ X + lam * np.eye(d)

    apply_Minv = make_sketch_preconditioner(X, lam, s, seed=1)
    MinvA = np.column_stack([apply_Minv(A[:, j]) for j in range(d)])

    ev = np.sort(np.abs(np.linalg.eigvals(MinvA)))
    kappa_pre = ev[-1] / ev[0]

    assert kappa_pre < 20          # predicted ≈ 9 at s = 4d
    assert np.linalg.cond(A) > 1e5 # sanity: the problem really is ill-conditioned