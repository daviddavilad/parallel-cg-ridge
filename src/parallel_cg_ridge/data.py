import numpy as np


def make_ridge_problem(
    n: int,
    d: int,
    cond: float,
    seed: int,
    noise_std: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic ridge regression problem with a prescribed spectrum.

    Builds ``X = U @ Sigma @ V.T`` with orthonormal ``U`` and ``V`` and
    geometrically spaced singular values from 1 down to ``1 / sqrt(cond)``.
    Since ``X.T @ X = V @ Sigma**2 @ V.T``, its eigenvalues are exactly the
    squared singular values, so the condition number of ``X.T @ X`` equals
    ``cond``. Controlling the spectrum this way makes the relationship between
    conditioning and CG iteration count directly testable.

    Parameters
    ----------
    n : int
        Number of samples (rows of ``X``). Must be at least ``d``.
    d : int
        Number of features (columns of ``X``).
    cond : float
        Condition number of ``X.T @ X`` (not of ``X``, whose condition number
        is ``sqrt(cond)``). Must be at least 1.
    seed : int
        Seed for the random number generator, for reproducibility.
    noise_std : float, optional
        Standard deviation of the Gaussian noise added to ``y``. Set to 0 for
        a noiseless problem where ``w_true`` is exactly recoverable.

    Returns
    -------
    X : ndarray of shape (n, d)
        Design matrix with the prescribed singular-value spectrum.
    y : ndarray of shape (n,)
        Targets, ``X @ w_true`` plus Gaussian noise.
    w_true : ndarray of shape (d,)
        The planted weight vector used to generate ``y``.

    Raises
    ------
    ValueError
        If ``n < d`` or ``cond < 1``.
    """

    if n < d:
        raise ValueError("n must be greater than or equal to d")

    if cond < 1.0:
        raise ValueError("cond must be at least 1")
    
    rng = np.random.default_rng(seed)

    # Build U from QR of an n x d Gaussian matrix
    A = rng.standard_normal((n, d))
    U, _ = np.linalg.qr(A, mode="reduced")

    # Build V from QR of a d x d Gaussian matrix
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))

    # Choose d singular values from 1 to 1/sqrt(cond)
    singular_values = np.geomspace(
        1.0,
        1.0 / np.sqrt(cond),
        d,
    )
    
    # Construct X = U Sigma V^T
    Sigma = np.diag(singular_values)
    X = U @ Sigma @ V.T

    # Draw planted weights w_true
    w_true = rng.standard_normal(d)

    # Generate y = X @ w_true + noise
    noise = noise_std * rng.standard_normal(n)
    y = X @ w_true + noise

    return X, y, w_true