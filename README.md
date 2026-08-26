# Parallel Matrix-Free Conjugate Gradient for Large-Scale Ridge Regression

Matrix-free distributed-memory conjugate gradient for large-scale ridge regression, with MPI strong/weak scaling studies.

## Problem

Ridge regression minimizes `‖Xw − y‖² + λ‖w‖²`, which reduces to the symmetric positive-definite system
 
```
(XᵀX + λI) w = Xᵀy
```

where `X` is `n × d` (n samples, d features) and `λ > 0`. Because the system is SPD, conjugate gradient is the natural Krylov solver. CG is also, equivalently, the optimization method that minimizes the quadratic `½wᵀAw − bᵀw`. Therefore, CG lies at the insersecion of numerical linear algebra (NLA) and convex optimization.

## Hypothesis

As the number of MPI ranks grows, at what point does the cost of the per-iteration global reduction overtake the shrinking local compute. Can a communication-avoiding variant (i.e. pipelined CG) push that crossover to higher rank counts?

This question is focused on parallel computing rather than on statistical and ML modeling. Ridge regression is a well-understood symmetric positive-definite system whose conditioning can be controlled exactly. Thus, the main subject of study is the solver's parallel behavior.

We hypothesize that strong-scaling efficiency will fall once the `O(log P)` reduction latency becomes comparable to the `O(nd/P)` local compute, and that the crossover rank count will shift upwards as problem size (n) scales. Additionally, we expect pipelined CG to underperform standard CG at low rank counts (where its extra vector dominates and there is little latency to hide), and to outperform it beyond the crossover threshold.

## Methodology

**Matrix-free.** `XᵀX` is never formed. Each iteration applies the operator as two matrix–vector products:
 
```
A·p = Xᵀ(X p) + λp
```

**Distribution.** `X` is partitioned by rows across MPI ranks and the `d`-dimensional vectors are replicated on every rank. Consequently, the CG inner products are local and the only global communication is a single `Allreduce` per iteration, assembling `Xᵀu` from per-rank partial sums inside the operator.

**Scaling.** Local compute is `O(nd/P)` and decreases with rank count, while the reduction is bounded by latency and grows roughly as `log P`. The strong- and weak-scaling experiments on CARC are designed to identify where this crossover occurs.

**Communication-avoiding comparison.** Pipelined CG reformulates the recurrence so the reduction is issued non-blocking (`Iallreduce`) and overlapped with the operator application, at the cost of extra vector work and reduced numerical stability. As a result, a good observation becomes comparing pipelined CG against standard CG, which would quantify where that trade pays off.

## Implementation

## Reproducing

Requires an MPI implementation (Open MPI) and Python 3.12+.
 
```bash
sudo apt install libopenmpi-dev openmpi-bin   # or equivalent
uv sync
```

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.12 |
| Numerics | NumPy |
| Parallelism | mpi4py over Open MPI (distributed memory) |
| Environment | uv |
| Testing / QA | pytest, ruff, mypy |
| Reference solvers | SciPy (`cg`, direct solve) for correctness baselines |
| Cluster | UNM CARC |
| GPU extension (planned) | C++ / CUDA for the MPI + GPU hybrid |

## Repository structure

```
parallel-cg-ridge/
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── parallel_cg_ridge/
│       ├── __init__.py
│       ├── data.py          # synthetic problem generation (prescribed spectrum)
│       ├── operators.py     # matrix-free ridge operator
│       ├── cg.py            # serial conjugate gradient
│       └── cg_mpi.py        # distributed conjugate gradient
├── scripts/                 # scaling drivers, CARC job scripts
├── tests/                   # correctness tests
├── docs/                    # design decisions and project notes
├── results/                 # timing outputs, gitignored
└── report/
    ├── main.tex
    └── figures/
```

## Status and roadmap

In development, Fall 2026. Current state: project scaffolding.
 
Planned deliverables:
 
- [ ] Serial matrix-free CG, verified against a direct solve of the normal equations
- [ ] Distributed MPI CG with consistent results across rank counts
- [ ] Strong- and weak-scaling studies with a compute-vs-communication timing breakdown
- [ ] Standard vs. pipelined CG comparison
- [ ] Analysis of λ, conditioning, and CG iteration count

## References

- Hestenes & Stiefel (1952), *Methods of conjugate gradients for solving linear systems*
- Saad (2003), *Iterative Methods for Sparse Linear Systems*
- Ghysels & Vanroose (2014), *Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm*
- Cools, Cornelis & Vanroose (2019), *Numerically stable recurrence relations for the communication-hiding pipelined Conjugate Gradient method*