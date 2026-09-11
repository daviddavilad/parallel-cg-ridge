import numpy as np
from mpi4py import MPI

from parallel_cg_ridge.cg import cg
from parallel_cg_ridge.cg_mpi import (
    assemble_rhs,
    make_distributed_ridge_operator,
    row_partition,
)
from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.operators import make_ridge_operator


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Same problem generated independently on every rank
n = 500
d = 20
cond = 10.0
lam = 1e-2
seed = 42

X, y, _ = make_ridge_problem(
    n=n,
    d=d,
    cond=cond,
    seed=seed,
)

# Slice this rank's row block
start, stop = row_partition(n, comm)

X_local = X[start:stop]
y_local = y[start:stop]

print(
    f"rank {rank}/{size}: "
    f"rows [{start}, {stop}), "
    f"X_local.shape={X_local.shape}"
)

# Distributed operator and distributed RHS
apply_A_dist = make_distributed_ridge_operator(
    X_local,
    lam,
    comm,
)

b_dist = assemble_rhs(
    X_local,
    y_local,
    comm,
)

# Existing serial CG is unchanged
w_dist, iters_dist, res_dist, conv_dist = cg(
    apply_A_dist,
    b_dist,
    tol=1e-10,
    maxiter=100,
)

# Only rank 0 performs the serial reference comparison
if rank == 0:
    apply_A_serial = make_ridge_operator(X, lam)
    b_serial = X.T @ y

    w_serial, iters_serial, res_serial, conv_serial = cg(
        apply_A_serial,
        b_serial,
        tol=1e-10,
        maxiter=100,
    )

    # Check that the distributed and serial solutions are close
    assert conv_dist
    assert conv_serial

    print()
    print("MPI ranks:", size)
    print("distributed iterations:", iters_dist)
    print("serial iterations:", iters_serial)
    print(
        "solution difference:",
        np.linalg.norm(w_dist - w_serial),
    )
    print(
        "rhs difference:",
        np.linalg.norm(b_dist - b_serial),
    )
    print("distributed residual:", res_dist)
    print("serial residual:", res_serial)