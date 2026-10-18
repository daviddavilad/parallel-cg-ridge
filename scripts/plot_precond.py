"""Compare unpreconditioned and sketch-preconditioned CG across the λ sweep."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parallel_cg_ridge.results import load_results

rows = load_results(sys.argv[1:])

# All arms must come from the same problem
problems = {(r["n"], r["d"], r["cond"]) for r in rows}
if len(problems) != 1:
    raise ValueError("All arms must use the same n, d, and cond")
n, d, cond = problems.pop()

# Group rows by arm: (preconditioner, sketch size, zeta)
arms = {}
for r in rows:
    arms.setdefault((r["precond"], r["s"], r.get("zeta")), []).append(r)

fig, (ax_iters, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))

markers = ["o", "s", "^", "v", "D"]

for i, ((precond, s, zeta), group) in enumerate(sorted(arms.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0))):
    # Log axes cannot show λ = 0
    group = sorted((r for r in group if r["lambda"] > 0), key=lambda r: r["lambda"])

    lams = [r["lambda"] for r in group]
    iters = [r["iters"] for r in group]
    total = [r["setup_time"] + r["solve_time"] for r in group]

    if precond == "none":
        label = "none"
    elif precond == "gaussian":
        label = f"Gaussian, s = {s // d}d"
    else:
        label = f"sparse, s = {s // d}d, ζ = {zeta}"

    marker = markers[i % len(markers)]
    style = dict(marker=marker, markersize=6)

    line, = ax_iters.plot(lams, iters, label=label, **style)
    color = line.get_color()
    ax_time.plot(lams, total, color=color, label=label, **style)

    if precond != "none":
        setup = np.median([r["setup_time"] for r in group])
        ax_time.axhline(setup, color=color, linestyle="--", alpha=0.4)

# σ²_min = 1/cond holds because the generator normalizes σ²_max = 1
sigma_min_sq = 1.0 / cond

for ax in (ax_iters, ax_time):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Regularization λ")
    ax.axvline(
        sigma_min_sq, color="gray", linestyle=":",
        label=r"$\sigma^2_{\min}$" if ax is ax_iters else None,
    )

# Legend entry for the dashed setup floors, without drawing anything
ax_time.plot([], [], color="gray", linestyle="--", alpha=0.4, label="setup time")

ax_iters.set_title("Iterations")
ax_iters.set_ylabel("CG iterations")
ax_iters.legend()

ax_time.set_title("Total time")
ax_time.set_ylabel("Setup + solve time (s)")
ax_time.legend()

fig.tight_layout()
output = Path("figures/precond_sparse.png")
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {output}")