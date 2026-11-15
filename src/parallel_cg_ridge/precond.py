"""Sketch-based preconditioners for ridge regression."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix

def sparse_sign_sketch(n, s, zeta, rng):
    """Draw a sparse sign sketch in column-oriented array form.

    Returns ``(positions, signs)``, both with shape ``(n, zeta)``.
    Row ``j`` of each array describes column ``j`` of the sketch matrix:
    ``positions[j]`` contains distinct row indices in ``[0, s)``, and
    ``signs[j]`` contains *unscaled* independent values from ``{-1, +1}``.
    The normalization ``1 / sqrt(zeta)`` is applied by
    ``sparse_sign_matrix``, not here.

    Positions (including collision resampling) are drawn before signs to
    preserve the established seed-to-sketch mapping.
    """
    if not (1 <= zeta <= s // 2):
        raise ValueError("zeta must be between 1 and s // 2")

    positions = rng.integers(0, s, size=(n, zeta))

    while True:
        sorted_positions = np.sort(positions, axis=1)
        collisions = np.any(sorted_positions[:, 1:] == sorted_positions[:, :-1], axis=1)

        if not np.any(collisions):
            break

        n_collisions = np.count_nonzero(collisions)
        positions[collisions] = rng.integers(0, s, size=(n_collisions, zeta))

    signs = rng.choice([-1.0, 1.0], size=zeta * n).reshape(n, zeta)

    return positions, signs

def sparse_sign_matrix(positions, signs, s):
    """Build an ``s x n_local`` CSC sketch from aligned column-wise arrays."""
    n_local, zeta = positions.shape

    rows = positions.ravel()
    cols = np.repeat(np.arange(n_local), zeta)
    vals = signs.ravel() / np.sqrt(zeta)

    return csc_matrix((vals, (rows, cols)), shape=(s, n_local))

def qr_preconditioner(SX, lam):
    """Return an inverse-operator callback for ``M = R.T @ R``."""
    if lam < 0:
        raise ValueError("lam must be non-negative")

    d = SX.shape[1]
    B = np.vstack([SX, np.sqrt(lam) * np.eye(d)])
    R = np.linalg.qr(B, mode="r")

    def apply_Minv(r):
        z = solve_triangular(R, r, lower=False, trans="T")
        
        return solve_triangular(R, z, lower=False)

    return apply_Minv

def make_sketch_preconditioner(X, lam, s, seed=0, kind="gaussian", zeta=2):
    """Return a Gaussian or sparse-sign sketch-based ridge preconditioner.

    For ``kind='sparse'``, each sketch column has ``zeta`` nonzeros
    of magnitude ``1/sqrt(zeta)`` at distinct row positions.
    ``zeta=1`` is CountSketch.

    The returned callback applies ``M^{-1}``, where
    ``M = R.T @ R`` approximates ``X.T @ X + lam * I``.
    """
    n, d = X.shape

    if s < d:
        raise ValueError("s must be at least d")
    
    if kind not in ("gaussian", "sparse"):
        raise ValueError("kind must be 'gaussian' or 'sparse'")

    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        S = rng.standard_normal((s, n)) / np.sqrt(s)

    else:
        positions, signs = sparse_sign_sketch(n, s, zeta, rng)
        S = sparse_sign_matrix(positions, signs, s)

    SX = S @ X
    return qr_preconditioner(SX, lam)

def preconditioned_condition_number(A, apply_Minv):
    """Return eigenvalue-based condition number for SPD-preconditioned A."""
    d = A.shape[0]

    MinvA = np.column_stack([apply_Minv(A[:, j]) for j in range(d)])

    ev = np.sort(np.abs(np.linalg.eigvals(MinvA)))

    return float(ev[-1] / ev[0])