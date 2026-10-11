import numpy as np
from scipy.linalg import solve_triangular

def make_sketch_preconditioner(X, lam, s, seed=0):
    """
    Gaussian sketch-and-precondition.

    Returns apply_Minv, where M = R.T @ R
    approximates X.T @ X + lam * I.
    """
    n, d = X.shape

    if lam < 0:
        raise ValueError("lam must be non-negative")
    if s < d:
        raise ValueError("s must be at least d")

    rng = np.random.default_rng(seed)

    # Gaussian sketch
    S = rng.standard_normal((s, n)) / np.sqrt(s)

    # Sketch the design matrix
    SX = S @ X

    # Incorporate ridge regularization
    B = np.vstack([SX, np.sqrt(lam) * np.eye(d)])

    # QR factorization
    R = np.linalg.qr(B, mode="r")

    def apply_Minv(r):
        # Step 1: Solve R.T @ z = r
        z = solve_triangular(R, r, lower=False, trans="T")

        # Step 2: Solve R @ u = z
        u = solve_triangular(R, z, lower=False)

        return u

    return apply_Minv