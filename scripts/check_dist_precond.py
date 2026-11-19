import numpy as np
from mpi4py import MPI

from parallel_cg_ridge.data import make_ridge_problem
from parallel_cg_ridge.cg_mpi import (
    scatter_rows,
    allocate_node_shared,
    make_distributed_sketch_preconditioner,
)
from parallel_cg_ridge.precond import (
    sparse_sign_sketch,
    sparse_sign_matrix,
    qr_preconditioner,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n, d = 1003, 20
s = 2 * d
cond = 10.0

# Only rank 0 generates the global design matrix.
if rank == 0:
    X, _, _ = make_ridge_problem(n=n, d=d, cond=cond, seed=0)
else:
    X = None

# Scatter the design matrix once.
X_local = scatter_rows(X, comm)

# All ranks independently generate the same test vector.
r = np.random.default_rng(99).standard_normal(d)

# Real node grouping.
real_node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

# Simulated nodes: ranks 0-1, 2-3, etc.
# Valid only when all processes are on one physical machine.
fake_node_comm = comm.Split(rank // 2, key=rank)

shared_cases = {}

for label, nc in (("shared-real", real_node_comm), ("shared-fake", fake_node_comm)):
    leader_comm = comm.Split(0 if nc.Get_rank() == 0 else MPI.UNDEFINED, key=rank)

    R_shared, win = allocate_node_shared((d, d), nc)

    shared_cases[label] = (nc, leader_comm, R_shared, win)

for zeta in (1, 2):

    # Rank 0 generates the global sparse sketch.
    if rank == 0:
        positions, signs = sparse_sign_sketch(n, s, zeta, np.random.default_rng(1))
    else:
        positions = None
        signs = None

    # Distribute matching columns of the sketch.
    positions_local = scatter_rows(positions, comm)
    signs_local = scatter_rows(signs, comm)

    for lam in (0.0, 1e-3):
        for qr_mode in ("replicated", "root", "shared-real", "shared-fake"):

            if qr_mode.startswith("shared"):
                nc, lc, R_shared, win = shared_cases[qr_mode]

                kwargs = {
                    "qr": "shared",
                    "R_shared": R_shared,
                    "node_comm": nc,
                    "leader_comm": lc,
                    "shared_win": win,
                }
            else:
                kwargs = {"qr": qr_mode}

            # Construct the distributed preconditioner.
            apply_Minv, _ = make_distributed_sketch_preconditioner(X_local, lam, s, positions_local, signs_local, comm, **kwargs)

            u = apply_Minv(r)

            # Gather the results from every rank.
            us = comm.gather(u, root=0)

            status = None

            if rank == 0:
                # Serial reference using the exact same sketch.
                S = sparse_sign_matrix(positions, signs, s)
                SX = S @ X

                apply_Minv_ref = qr_preconditioner(SX, lam)
                u_ref = apply_Minv_ref(r)

                # Check 1: distributed vs serial.
                serial_ok = np.allclose(u, u_ref, rtol=1e-12, atol=0)

                rel_err = (np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref))

                # Check 2: all MPI ranks agree bitwise.
                ranks_ok = all(np.array_equal(us[0], other) for other in us[1:])

                status = (serial_ok, ranks_ok, rel_err)

            # Give every rank the same result before asserting.
            serial_ok, ranks_ok, rel_err = comm.bcast(status, root=0)

            assert serial_ok, (
                f"Serial mismatch: qr={qr_mode}, zeta={zeta}, "
                f"lam={lam}, rel_err={rel_err:.3e}"
            )

            assert ranks_ok, (
                f"MPI ranks disagree: qr={qr_mode}, zeta={zeta}, lam={lam}"
            )

            if rank == 0:
                print(
                    f"P={comm.Get_size()}, "
                    f"qr={qr_mode}, zeta={zeta}, lam={lam:g}: "
                    f"PASS (rel_err={rel_err:.3e}, ranks bitwise equal)",
                    flush=True,
                )

for nc, lc, R_shared, win in shared_cases.values():
    win.Unlock_all()
    win.Free()

    if lc != MPI.COMM_NULL:
        lc.Free()

    nc.Free()
