import sys
import json

paths = sys.argv[1:]

for path in paths:
    print(f"== {path}")

    with open(path) as f:
        rows = [json.loads(line) for line in f if line.startswith("{")]

    for r in rows:
        p, w = r["P"], r["wall"]
        it = r["iters"]

        c = r.get("compute_per_rank")
        if c:
            stats = f"min={min(c):.3f}  max={max(c):.3f}  ratio={max(c)/min(c):.2f}"
        else:
            stats = "min=—  max=—  ratio=—"

        cpr = r.get("cpus_per_rank")
        cpus = [x[0] if x else None for x in cpr] if cpr else "—"

        n = r["n"]
        print(f"n={n:>7}  P={p:>2}  iters={it:>3}  wall={w:.3f}  {stats}")