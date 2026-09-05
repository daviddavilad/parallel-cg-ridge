import numpy as np


def cg(apply_A, b, tol=1e-10, maxiter=None):
    """
    Conjugate Gradient method for solving Ax = b.

    Parameters:
    - apply_A: A function that applies the matrix A to a vector.
    - b: The right-hand side vector.
    - tol: Tolerance for convergence. It is relative to the norm of b. The method stops when the norm of the residual is less than tol * ||b||.
    - maxiter: Maximum number of iterations. Must be at least 1. If None, defaults to the size of b.

    Returns:
    - x: The solution vector.
    - num_iter: The number of iterations performed.
    - residual_norm: The norm of the final residual. (Relative residual norm: ||r|| / ||b|| )
    - converged: A boolean indicating whether the method converged.
    """
    n = len(b)
    if maxiter is None:
        maxiter = n

    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")

    x = np.zeros(n)
    r = b.copy() # r = b - apply_A(x) with x = np.zeros(n) computes b - 0, so r = b
    p = r.copy()
    rsold = np.dot(r, r)
    b_norm = np.linalg.norm(b)
    converged = False

    for i in range(maxiter):
        Ap = apply_A(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)

        if np.sqrt(rsnew) / b_norm < tol:
            converged = True
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, i + 1, np.sqrt(rsnew) / b_norm, converged # caller inspects the flag to determine if the solution is valid