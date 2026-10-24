import numpy as np
import pytest

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.precond import (
    make_sketch_preconditioner,
    preconditioned_condition_number,
)

@pytest.mark.parametrize(
    "kind,zeta",
    [
        ("gaussian", None),
        ("sparse", 2),
        ("sparse", 4),
    ],
)
def test_sketch_preconditioner_reduces_conditioning(kind, zeta):
    n, d, s, lam = 2000, 50, 200, 0.0

    X, _, _ = make_ridge_problem(n=n, d=d, cond=1e6, seed=0)

    A = X.T @ X + lam * np.eye(d)

    kwargs = ({"kind": kind} if zeta is None else {"kind": kind, "zeta": zeta})

    apply_Minv = make_sketch_preconditioner(X, lam, s, seed=1, **kwargs)

    kappa_pre = preconditioned_condition_number(A, apply_Minv)

    print(f"{kind} zeta={zeta}: kappa_pre = {kappa_pre:.2f}")

    assert kappa_pre < 20
    assert np.linalg.cond(A) > 1e5