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

def pcg(apply_A, apply_Minv, b, tol=1e-10, maxiter=None):
    """
    Solve Ax = b using preconditioned conjugate gradient.

    Based on Algorithm 1 of Ghysels & Vanroose (2014).
    apply_Minv(r) returns M^{-1}r.

    Returns (x, iters, rel_res, converged).
    """
    n = b.shape[0]

    if maxiter is None:
        maxiter = n

    x = np.zeros_like(b)
    r = b.copy()

    b_norm = np.linalg.norm(b)

    if b_norm == 0:
        return x, 0, 0.0, True

    # Initial preconditioned residual
    u = apply_Minv(r)

    # Initial search direction
    p = u.copy()

    # Scalar needed for alpha and beta
    rho = np.dot(r, u)

    rel_res = np.linalg.norm(r) / b_norm
    iters = 0
    converged = False

    for i in range(maxiter):
        s = apply_A(p)

        alpha = rho / np.dot(p, s)

        x += alpha * p
        r -= alpha * s

        rel_res = np.linalg.norm(r) / b_norm

        iters = i + 1

        if rel_res < tol:
            converged = True
            break

        u_new = apply_Minv(r)
        rho_new = np.dot(r, u_new)

        beta = rho_new / rho

        p = u_new + beta * p

        # Update scalar history
        rho = rho_new

    return x, iters, rel_res, converged