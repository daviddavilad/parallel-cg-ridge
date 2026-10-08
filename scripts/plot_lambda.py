"""Plot CG iterations against kappa(A) from the regularization lambda sweep"""

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_cg_ridge.results import load_results

outname = "lambda_sweep.png"

data = load_results(["results/final/lambda_sweep.jsonl"])
fig, ax = plt.subplots()

data = sorted(data, key=lambda r: r["kappa"])
kappa = np.array([r["kappa"] for r in data])
iters = np.array([r["iters"] for r in data])
d = data[0]["d"]

ax.plot(kappa, iters, marker="o", label="measured")

ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)

ax.axhline(d, color="gray", linestyle="--", linewidth=1, label="d (exact-arithmetic limit)")

k_ref = iters[0] * np.sqrt(kappa / kappa[0])
ax.plot(kappa, k_ref, ":", color="gray", linewidth=1.5, label="√κ reference")

ax.set_xlabel("Condition number κ(A)")
ax.set_ylabel("Iterations")
ax.legend()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
output = figures_dir / outname
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)