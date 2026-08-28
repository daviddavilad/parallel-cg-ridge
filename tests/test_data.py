"""Test the condition number of generated ridge regression problems."""

from parallel_cg_ridge.data import make_ridge_problem
import numpy as np


def test_condition_numbers():
    """Check the condition number of generated ridge regression problems."""
    for cond in [1, 10, 100, 1e4]:
        X, y, w_true = make_ridge_problem(
            n=500,
            d=20,
            cond=cond,
            seed=42,
        )

        actual = np.linalg.cond(X.T @ X)
        print(f"Requested: {cond:>8.0f} | Actual: {actual:.4f}")
        assert np.isclose(actual, cond, rtol=1e-6), f"Condition number mismatch: requested {cond}, got {actual}"