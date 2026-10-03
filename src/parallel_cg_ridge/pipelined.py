import numpy as np

def pipelined_cg(start_A, finish_A, b, tol=1e-10, maxiter=None):
    """
    Solve the linear system Ax = b using the pipelined conjugate gradient method.
    This is algorithm 3 - unpreconditioned pipelined conjugate gradient method - from
    Ghysels & Vanroose (2014), Hiding global synchronization latency in the 
    preconditioned Conjugate Gradient algorithm.

    The algorithm is adapted to the ridge regression problem, where A is 
    symmetric positive definite and b is the right-hand side vector.
    """
    n = b.shape[0]

    if maxiter is None:
        maxiter = n

    x = np.zeros_like(b)
    r = b.copy()

    b_norm = np.linalg.norm(b)

    # Zero-RHS case
    if b_norm == 0:
        return (x, 0, 0.0, True)

    # w_0 = A r_0
    req = start_A(r)
    w = finish_A(req, r)

    # Previous recurrence vectors
    s = np.zeros_like(b)
    p = np.zeros_like(b)
    z = np.zeros_like(b)

    # Scalar history needed for i > 0
    gamma_prev = 0.0
    alpha_prev = 0.0

    rel_res = np.linalg.norm(r) / b_norm
    iters = 0
    converged = False

    for i in range(maxiter):
        gamma = np.dot(r, r)
        delta = np.dot(r, w)

        rel_res = np.sqrt(gamma) / b_norm

        if rel_res < tol:
            converged = True
            break

        if i == 0:
            beta = 0.0
            alpha = gamma / delta
        else:
            beta = gamma / gamma_prev
            alpha = gamma / (delta - beta * gamma / alpha_prev)

        req = start_A(w)

        # Update recurrence vectors
        s = w + beta * s
        p = r + beta * p
        x += alpha * p
        r -= alpha * s

        q = finish_A(req, w)
        z = q + beta * z
        w -= alpha * z

        gamma_prev = gamma
        alpha_prev = alpha
        iters = i + 1

    return (x, iters, rel_res, converged)