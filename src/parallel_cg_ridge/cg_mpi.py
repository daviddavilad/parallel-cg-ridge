import numpy as np

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

from .precond import (
    sparse_sign_matrix,
    qr_preconditioner,
    sketch_qr_factor,
    triangular_preconditioner,
)

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

def allocate_node_shared(shape, node_comm):
    """Allocate one float64 shared array per node."""
    dtype = np.dtype(np.float64)
    itemsize = dtype.itemsize

    node_rank = node_comm.Get_rank()
    nbytes = int(np.prod(shape)) * itemsize if node_rank == 0 else 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)

    win.Lock_all(MPI.MODE_NOCHECK)

    buf, _ = win.Shared_query(0)

    R_shared = np.ndarray(shape=shape, dtype=dtype, buffer=buf)

    return R_shared, win

def make_distributed_sketch_preconditioner(X_local, lam, s, positions_local, signs_local, comm, qr="replicated", R_shared=None, node_comm=None, leader_comm=None, shared_win=None):
    """
    Construct a distributed sparse-sign ridge preconditioner.

    qr="replicated": Allreduce SX, then QR on every rank.
    qr="root": Reduce SX, QR on rank 0, broadcast R.
    qr="shared": Reduce SX, QR on rank 0, leaders Bcast R into one shared copy per node.

    Returns (apply_Minv, timings).
    """
    n_local, d = X_local.shape
    rank = comm.Get_rank()

    if s < d:
        raise ValueError("s must be at least d")

    if qr not in ("replicated", "root", "shared"):
        raise ValueError("qr must be 'replicated', 'root', or 'shared'")

    if qr == "shared":
        if R_shared is None or node_comm is None or shared_win is None:
            raise ValueError("shared mode requires a shared buffer, window and node communicator")
        if R_shared.shape != (d, d) or R_shared.dtype != np.float64:
            raise ValueError("R_shared must be a float64 d x d array")
        if node_comm.Get_rank() == 0 and leader_comm is None:
            raise ValueError("node leaders require leader_comm")
        if rank == 0 and node_comm.Get_rank() != 0:
            raise ValueError("global rank 0 must be its node's leader")

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
    t0 = MPI.Wtime()

    if qr == "replicated":
        SX = np.empty_like(partial)
        comm.Allreduce(partial, SX, op=MPI.SUM)
    else:
        SX = np.empty_like(partial) if rank == 0 else None
        comm.Reduce(partial, SX, op=MPI.SUM, root=0)

    comm_time = MPI.Wtime() - t0

    # QR factorization
    qr_time = 0.0
    bcast_time = 0.0
    node_sync_time = 0.0

    if qr == "replicated":
        t0 = MPI.Wtime()
        apply_Minv = qr_preconditioner(SX, lam)
        qr_time = MPI.Wtime() - t0

    elif qr == "root":
        if rank == 0:
            t0 = MPI.Wtime()
            R = np.ascontiguousarray(sketch_qr_factor(SX, lam))
            qr_time = MPI.Wtime() - t0
        else:
            R = np.empty((d, d), dtype=X_local.dtype)

        t0 = MPI.Wtime()
        comm.Bcast(R, root=0)
        bcast_time = MPI.Wtime() - t0

        apply_Minv = triangular_preconditioner(R)

    else:
        # Only global rank 0 factorizes.
        if rank == 0:
            t0 = MPI.Wtime()

            # Safe to overwrite R_shared: rank 0 reaches this point only after its
            # Reduce has completed, which requires every rank to have entered setup,
            # i.e. finished any previous solve that reads R_shared.
            R_shared[:] = sketch_qr_factor(SX, lam)
            
            qr_time = MPI.Wtime() - t0

        # Only node leaders communicate the factor.
        if node_comm.Get_rank() == 0:
            t0 = MPI.Wtime()
            leader_comm.Bcast(R_shared, root=0)
            bcast_time = MPI.Wtime() - t0

            # Publish the leader's writes.
            shared_win.Sync()

        # Wait until each node's leader has received R, then synchronize
        # the shared view before reading it. Non-leaders wait here for
        # rank 0's QR and the leaders' Bcast.
        t0 = MPI.Wtime()
        node_comm.Barrier()
        shared_win.Sync()
        node_sync_time = MPI.Wtime() - t0

        apply_Minv = triangular_preconditioner(R_shared)

    timings = {"sketch": sketch_time, "comm": comm_time, "qr": qr_time, "bcast": bcast_time, "node_sync": node_sync_time}

    return apply_Minv, timings
