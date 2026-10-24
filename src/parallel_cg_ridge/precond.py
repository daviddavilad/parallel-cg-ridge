import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix

def make_sketch_preconditioner(X, lam, s, seed=0, kind="gaussian", zeta=2):
    """
    Sketch-and-precondition with a Gaussian or sparse sign sketch.

    Two kinds of sketching are used: Gaussian and sparse. 
    kind="sparse" uses a sparse sign sketch: each column of S has zeta nonzeros,
    each ±1/sqrt(zeta), in distinct rows. zeta = 1 is CountSketch.

    Returns apply_Minv, where M = R.T @ R
    approximates X.T @ X + lam * I.
    """
    n, d = X.shape

    if lam < 0:
        raise ValueError("lam must be non-negative")
    if s < d:
        raise ValueError("s must be at least d")
    
    if kind not in ("gaussian", "sparse"):
        raise ValueError("kind must be 'gaussian' or 'sparse'")

    if kind == "sparse" and not (1 <= zeta <= s // 2):
        raise ValueError("zeta must be between 1 and s // 2")

    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        S = rng.standard_normal((s, n)) / np.sqrt(s)

    else:
        # Generate zeta candidate row positions per column
        positions = rng.integers(0, s, size=(n, zeta))

        # Resample columns containing duplicate row positions
        while True:
            sorted_positions = np.sort(positions, axis=1)

            collisions = np.any(sorted_positions[:, 1:] == sorted_positions[:, :-1], axis=1)

            if not np.any(collisions):
                break

            n_collisions = np.count_nonzero(collisions)

            positions[collisions] = rng.integers(0, s, size=(n_collisions, zeta))

        rows = positions.ravel()
        cols = np.repeat(np.arange(n), zeta)

        vals = (rng.choice([-1.0, 1.0], size=zeta * n) / np.sqrt(zeta))

        S = csc_matrix((vals, (rows, cols)), shape=(s, n))

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

def preconditioned_condition_number(A, apply_Minv):
    """
    Compute the spectral condition number of M^{-1}A.

    M^{-1}A is similar to an SPD matrix when A and M are SPD,
    so its eigenvalues are real and positive in exact arithmetic.
    """
    d = A.shape[0]

    MinvA = np.column_stack([apply_Minv(A[:, j]) for j in range(d)])

    ev = np.sort(np.abs(np.linalg.eigvals(MinvA)))

    return float(ev[-1] / ev[0])