import numpy as np
from mpi4py import MPI
from parallel_cg_ridge.cg_mpi import row_partition, scatter_rows
from parallel_cg_ridge.data import make_ridge_problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

X, y, _ = make_ridge_problem(n=1003, d=20, cond=10.0, seed=0)

X_local = scatter_rows(X if rank == 0 else None, comm)
y_local = scatter_rows(y if rank == 0 else None, comm)

start, stop = row_partition(1003, comm)
assert np.array_equal(X_local, X[start:stop])
assert np.array_equal(y_local, y[start:stop])

ok = comm.gather(True, root=0)
if rank == 0:
    print(f"scatter correct on {comm.Get_size()} ranks")