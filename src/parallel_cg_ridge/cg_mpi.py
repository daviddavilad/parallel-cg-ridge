import numpy as np

from mpi4py import MPI

def row_partition(n: int, comm) -> tuple[int, int]:
    """Return the [start, stop) row range owned by this MPI rank."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    base = n // size
    remainder = n % size

    local_n = base + (1 if rank < remainder else 0)

    start = rank * base + min(rank, remainder)
    stop = start + local_n

    return start, stop

def make_distributed_ridge_operator(X_local, lam, comm):
    d = X_local.shape[1]
    recvbuf = np.empty(d)          # preallocated, reused every iteration

    def apply_A(p):
        u = X_local @ p                     # local, no communication
        partial = X_local.T @ u             # local partial d-vector
        comm.Allreduce(partial, recvbuf, op=MPI.SUM)
        return recvbuf + lam * p

    return apply_A

def assemble_rhs(X_local, y_local, comm):
    partial = X_local.T @ y_local
    b = np.empty_like(partial)
    comm.Allreduce(partial, b, op=MPI.SUM)
    return b