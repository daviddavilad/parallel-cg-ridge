import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_cg_ridge.results import load_results, efficiency, speedup

which = sys.argv[1]
paths = sys.argv[2:]

if which == "efficiency":
    data = efficiency(load_results(paths))
    ylabel = "Iteration-normalized parallel efficiency"
    outname = "efficiency.png"
elif which == "speedup":
    data = speedup(load_results(paths))
    ylabel = "Iteration-normalized speedup"
    outname = "speedup.png"
else:
    raise SystemExit("usage: plot_scaling.py [efficiency|speedup] FILES...")

fig, ax = plt.subplots()
all_p = set()

for n, pairs in sorted(data.items()):
    ps = [p for p, e in pairs]
    es = [e for p, e in pairs]
    ax.plot(ps, es, marker="o", label=f"n = {n:,}")
    all_p.update(ps)

ticks = sorted(all_p)
ax.set_xscale("log", base=2)
ax.set_xticks(ticks)
ax.set_xticklabels([str(p) for p in ticks])

if which == "efficiency":
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal")
else:
    ax.set_yscale("log", base=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(p) for p in ticks])
    ax.plot(ticks, ticks, "--", color="gray", linewidth=1, label="ideal")

ax.set_xlabel("MPI ranks")
ax.set_ylabel(ylabel)
ax.legend()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
output = figures_dir / outname
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)