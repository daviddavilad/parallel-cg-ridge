import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_cg_ridge.results import load_results, efficiency

paths = sys.argv[1:]

rows = load_results(paths)
eff_by_n = efficiency(rows)

fig, ax = plt.subplots()
all_p = set()

for n, pairs in sorted(eff_by_n.items()):
    ps = [p for p, e in pairs]
    es = [e for p, e in pairs]
    ax.plot(ps, es, marker="o", label=f"n = {n:,}")
    all_p.update(ps)

ticks = sorted(all_p)
ax.set_xscale("log", base=2)
ax.set_xticks(ticks)
ax.set_xticklabels([str(p) for p in ticks])

ax.set_xlabel("MPI ranks")
ax.set_ylabel("Iteration-normalized parallel efficiency")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal")
ax.legend()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
output = figures_dir / "efficiency.png"
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)