import numpy as np


def make_ridge_operator(X: np.ndarray, lam: float):
    def apply_A(p: np.ndarray) -> np.ndarray:
        return X.T @ (X @ p) + lam * p

    return apply_A