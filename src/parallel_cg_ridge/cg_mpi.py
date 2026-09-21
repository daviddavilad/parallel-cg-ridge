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

def row_counts(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Row counts and row offsets for every rank."""
    base = n // size
    remainder = n % size

    counts = np.zeros(size, np.int64)
    offsets = np.zeros(size, np.int64)

    for rank in range(size):
        counts[rank] = base + (1 if rank < remainder else 0)
        offsets[rank] = rank * base + min(rank, remainder)
    
    return counts, offsets

def make_distributed_ridge_operator(X_local, lam, comm):
    d = X_local.shape[1]
    recvbuf = np.empty(d)          # preallocated, reused every iteration

    timings = {"compute": 0.0, "comm": 0.0, "calls": 0}

    def apply_A(p):
        t0 = MPI.Wtime()
        u = X_local @ p                     # local, no communication
        partial = X_local.T @ u             # local partial d-vector
        timings["compute"] += MPI.Wtime() - t0
        timings["calls"] += 1

        t0 = MPI.Wtime()
        comm.Allreduce(partial, recvbuf, op=MPI.SUM)
        timings["comm"] += MPI.Wtime() - t0

        return recvbuf + lam * p

    return apply_A, timings

def assemble_rhs(X_local, y_local, comm):
    partial = X_local.T @ y_local
    b = np.empty_like(partial)
    comm.Allreduce(partial, b, op=MPI.SUM)
    return b