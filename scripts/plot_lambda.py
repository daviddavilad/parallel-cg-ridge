"""Plot CG iterations against kappa(A) from the regularization lambda sweep"""

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from parallel_cg_ridge.results import load_results

outname = "lambda_sweep.png"

data = load_results(["results/final/lambda_sweep.jsonl"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.tight_layout()

data = sorted(data, key=lambda r: r["kappa"])
kappa = np.array([r["kappa"] for r in data])
iters = np.array([r["iters"] for r in data])
lam = np.array([r["lambda"] for r in data])
d = data[0]["d"]

# First subplot: iterations vs condition number
ax1.plot(kappa, iters, marker="o", label="measured")

ax1.set_title("Iterations vs. κ(A)")
ax1.set_xscale("log", base=10)
ax1.set_yscale("log", base=10)

ax1.axhline(d, color="gray", linestyle="--", linewidth=1, label="d (exact-arithmetic limit)")

k_ref = iters[0] * np.sqrt(kappa / kappa[0])
ax1.plot(kappa, k_ref, ":", color="gray", linewidth=1.5, label="√κ reference")

ax1.set_xlabel("Condition number κ(A)")
ax1.set_ylabel("Iterations")
ax1.legend()

# Second subplot: iterations vs regularization parameter
pos = lam > 0
ax2.plot(lam[pos], iters[pos], marker="o", label="measured")
ax2.set_title("Iterations vs. λ")
ax2.invert_xaxis()

ax2.set_xscale("log", base=10)
ax2.set_yscale("log", base=10)
ax2.set_xlabel("Regularization parameter λ")
ax2.set_ylabel("Iterations")

sigma2_min = 1.0 / data[0]["cond"]   # valid because the generator normalizes σ²_max = 1
ax2.axvline(sigma2_min, color="gray", linestyle="-.", linewidth=1, label=r"$\sigma^2_{\min}$")
iters_ols = iters[lam == 0][0]
ax2.axhline(iters_ols, color="gray", linestyle=":", linewidth=1.5, label="λ = 0 (OLS)")
ax2.axhline(d, color="gray", linestyle="--", linewidth=1, label="d (exact-arithmetic limit)")

ax2.legend(loc="lower right")

# Save the figure to results/
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
output = figures_dir / outname
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)