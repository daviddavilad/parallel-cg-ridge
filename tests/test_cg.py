import numpy as np

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.operators import make_ridge_operator
from parallel_cg_ridge.cg import (
    cg, 
    pcg,
)

def test_cg_matches_direct_solve():
    """Test that the CG solver matches the direct solve for a ridge regression problem."""
    n, d, cond, lam = 200, 20, 10.0, 1e-3

    X, y, w_true = make_ridge_problem(n = n, d = d, cond = cond, seed = 0, noise_std = 0.0)
    apply_A = make_ridge_operator(X, lam)
    b = X.T @ y

    w_cg, iters, rel_res, converged = cg(apply_A, b)

    A_dense = X.T @ X + lam * np.eye(d)
    w_direct = np.linalg.solve(A_dense, b)

    assert converged, "CG did not converge"
    assert np.allclose(w_cg, w_direct, rtol=1e-8)
    assert rel_res < 1e-10, f"Relative residual is too high: {rel_res}"

def test_iterations_increase_with_conditioning():
    n, d, lam = 200, 20, 1e-3

    iters = {}
    for cond in [10.0, 1e4]:
        X, y, _ = make_ridge_problem(n=n, d=d, cond=cond, seed=0, noise_std=0.0)
        apply_A = make_ridge_operator(X, lam)
        b = X.T @ y
        _, k, _, converged = cg(apply_A, b, maxiter=10 * d)
        assert converged
        iters[cond] = k

    assert iters[1e4] > iters[10.0]

def test_pcg_identity_matches_cg():
    X, y, _ = make_ridge_problem(n=200, d=20, cond=10.0, seed=0, noise_std=0.0)
    apply_A = make_ridge_operator(X, 1e-2)
    b = X.T @ y

    x_cg, it_cg, _, _ = cg(apply_A, b, tol=1e-10, maxiter=200)
    x_pcg, it_pcg, _, _ = pcg(apply_A, lambda r: r.copy(), b, tol=1e-10, maxiter=200)

    assert it_cg == it_pcg
    assert np.allclose(x_cg, x_pcg, rtol=1e-12, atol=1e-14)