import sys

from parallel_cg_ridge.results import load_results

rows = load_results(sys.argv[1:])

for r in rows:
    p, w = r["P"], r["wall"]
    it = r["iters"]

    c = r.get("compute_per_rank")
    solver = r.get("solver", "cg")
    comm = r["comm"]
    compute = r["compute"]

    # Normalize by iterations
    ms_per_iter = 1000 * w / it
    comm_per_iter = 1000 * comm / it

    cpr = r.get("cpus_per_rank")
    cpus = [x[0] if x else None for x in cpr] if cpr else "—"

    n = r["n"]
    print(f"solver={solver}  n={n:>7}  P={p:>2}  iters={it:>3}  wall={w:.3f}  compute={compute:.3f}  comm={comm:.3f}")