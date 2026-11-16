import numpy as np

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

from .precond import sparse_sign_matrix, qr_preconditioner

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

def scatter_rows(A, comm, root=0):
    """Distribute row blocks of a C-contiguous NumPy array."""
    rank = comm.Get_rank()

    if rank == root:
        assert A.flags["C_CONTIGUOUS"]
        meta = (A.shape, A.dtype)
    else:
        meta = None

    # Broadcast metadata so every rank knows the shape and dtype.
    shape, dtype = comm.bcast(meta, root=root)

    mpi_dtype = from_numpy_dtype(dtype)

    # Determine the receiving rank's row range.
    start, stop = row_partition(shape[0], comm)

    local_shape = (stop - start,) + shape[1:]
    local = np.empty(local_shape, dtype=dtype)

    if rank == root:
        counts, offsets = row_counts(shape[0], comm.Get_size())

        # Scatterv counts elements, not rows.
        row_elems = int(np.prod(shape[1:]))

        elem_counts = counts * row_elems
        elem_offsets = offsets * row_elems

        sendbuf = [A, elem_counts, elem_offsets, mpi_dtype]
    else:
        sendbuf = None

    comm.Scatterv(sendbuf, [local, mpi_dtype], root=root)

    return local

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

def make_split_ridge_operator(X_local, lam, comm):
    """Returns (start_A, finish_A). start_A launches the reduction; finish_A waits."""
    d = X_local.shape[1]
    recvbuf = np.empty(d)          # preallocated, reused every iteration

    timings = {"compute": 0.0, "comm": 0.0, "calls": 0}
    state = {"req": None, "partial": None}

    def start_A(p):
        assert state["req"] is None, "previous reduction not awaited"
        t0 = MPI.Wtime()
        u = X_local @ p
        state["partial"] = X_local.T @ u
        timings["compute"] += MPI.Wtime() - t0
        timings["calls"] += 1

        state["req"] = comm.Iallreduce(state["partial"], recvbuf, op=MPI.SUM)
        return state["req"]

    def finish_A(req, p):
        t0 = MPI.Wtime()
        req.Wait()
        timings["comm"] += MPI.Wtime() - t0
        state["req"] = None
        return recvbuf + lam * p

    return start_A, finish_A, timings

def assemble_rhs(X_local, y_local, comm):
    partial = X_local.T @ y_local
    b = np.empty_like(partial)
    comm.Allreduce(partial, b, op=MPI.SUM)
    return b

def make_distributed_sketch_preconditioner(X_local, lam, s, positions_local, signs_local, comm):
    """
    Construct a distributed sparse-sign ridge preconditioner.

    The sketch is supplied externally and is independent of rank count.

    Returns (apply_Minv, timings).
    """
    n_local, d = X_local.shape

    if s < d:
        raise ValueError("s must be at least d")

    if positions_local.shape != signs_local.shape:
        raise ValueError("positions and signs must have matching shapes")

    if positions_local.shape[0] != n_local:
        raise ValueError("sketch columns must match local X rows")

    # Local sketch multiplication
    t0 = MPI.Wtime()

    S_local = sparse_sign_matrix(positions_local, signs_local, s)

    partial = np.ascontiguousarray(S_local @ X_local)

    sketch_time = MPI.Wtime() - t0

    # Global sketch assembly
    SX = np.empty_like(partial)

    t0 = MPI.Wtime()

    comm.Allreduce(partial, SX, op=MPI.SUM)

    comm_time = MPI.Wtime() - t0

    # Replicated QR factorization
    t0 = MPI.Wtime()

    apply_Minv = qr_preconditioner(SX, lam)

    qr_time = MPI.Wtime() - t0

    timings = {"sketch": sketch_time, "comm": comm_time, "qr": qr_time}

    return apply_Minv, timings
