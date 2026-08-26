# Design

Notes on the implementation. Written as source material for the method section of the final report.

## Matrix-free operator

Ridge regression requires solving `(XᵀX + λI) w = Xᵀy`. The naive approach forms `XᵀX` explicitly, which costs `O(nd²)` and produces a `d × d` matrix. For the problem sizes where ridge regression becomes interesting, this cost is both dominant and unnecessary. On the other hand, CG only ever needs the *action* of the operator on a vector, never the operator itself.

The operator is therefore applied by association:

```
A·p = Xᵀ(X p) + λp
```
 
Two matrix–vector products with `X` and a scaled vector add, per iteration, at `O(nd)` cost.

## Solver interface

`cg()` accepts a *callable* that applies the operator to a vector, never a matrix. The consequence is that the serial solver, the MPI solver, and any future GPU version share one implementation of the CG recurrence; so only the operator changes.

## Data distribution
 
`X` (`n × d`) is partitioned **by rows**, so rank `r` holds a contiguous block `X_r` of `n_r` samples along with the matching entries of `y`. The `d`-dimensional vectors (`w`, the residual, the search direction) are **replicated** on every rank.
 
Tracing an operator application:
 
1. `X_r · p` — `p` is replicated and `X_r` is local, so this requires **no communication**. The result is rank `r`'s block of an `n`-vector.
2. `X_rᵀ · u_r` — produces a *partial* `d`-vector. The true result is the sum of these partials across all ranks, which requires a single **`Allreduce`** of size `d`.

Two consequences arise for this project:

- The CG inner products (`rᵀr`, `pᵀAp`) are computed on replicated vectors and therefore cost **nothing** to communicate.
- The one global reduction lives **inside the operator**, not in the dot products.

### Alternative considered
 
If `d` were large relative to `n`, distributing the feature vectors as well would be the natural choice; the dot products would then also require reductions, matching the textbook distributed-CG setting. For the regime of interest here (`n ≫ d`), replicating the `d`-vectors is both cheaper and simpler, and it is what produces the communication structure described above.

## Cost model
 
Per iteration:
 
- **Local compute:** `O(n·d/P)` — decreases as ranks are added.
- **Communication:** one `Allreduce` of `d` doubles, latency-bound, growing roughly as `O(log P)`.

At small `P` compute dominates and scaling looks close to ideal. As `P` grows, the shrinking compute and growing reduction cost cross over, and parallel efficiency degrades.

## Pipelined CG, not s-step

Both reduce communication cost, but in different ways, and only pipelined CG fits this problem.

**s-step CG** builds a Krylov basis `[p, Ap, …, Aˢp]` via a matrix-powers kernel, which avoids communication by exploiting the **sparsity and locality** of `A` — a finite-difference stencil touches only neighboring grid points, so several applications can share one enlarged halo exchange. Here `A = XᵀX + λI` is effectively dense and its cost is a *global* reduction, not neighbor communication. Each of the `s` applications would still need its own `Allreduce`, so this method does not improve performance materially.

**Pipelined CG** hides a single global reduction by issuing it non-blocking (`Iallreduce`) and overlapping it with the operator application. That matches this structure directly: one reduction per iteration, with local work (`X_r · p`) available to overlap it against.

Costs / Caveats: extra vector work per iteration, and reduced numerical stability from the reformulated recurrences. Therefore, the implementation will follow the stable variants of Cools et al. (2019), and will monitor both the recursive residual and the true `‖b − Aw‖`.

*(Terminology: "asynchronous CG" sometimes refers to this family, but also to chaotic relaxation, where processes never synchronize. Define the term explicitly in the report.)*

## Synthetic problem generation

Synthetic data is primary for the scaling and conditioning experiments because the results are easier to benchmark, as compared to using a real dataset. Synthetic data also provides control that a real dataset cannot provide:

- `X = UΣVᵀ` with a **prescribed singular-value spectrum**, so `κ(XᵀX) = (σ_max/σ_min)²` is set exactly. CG's iteration count is governed by conditioning, making the λ–conditioning–convergence relationship directly testable.
- Iteration count can be **held fixed across rank counts**, isolating parallel cost in strong scaling rather than confounding it with convergence differences.
- Problem size dials to whatever makes distributed memory meaningful.

`X` is generated **on the fly, per rank, from a deterministic seed** (each rank builds its own row block locally). This eliminates data I/O, which would otherwise become the very bottleneck the study aims to measure.

A ground-truth `w*` is planted (`y = Xw* + noise`), so recovery of `w*` checks correctness against the regression problem itself, independent of the residual norm.

A real dataset will be used and implemented for validation.

## Correctness strategy

- **Serial:** direct solve of the normal equations (Cholesky) on a small problem, plus SciPy's `cg`.
- **Distributed:** results consistent with the serial baseline at any rank count.
- **Reproducibility caveat:** floating-point addition isn't associative, so `Allreduce` sums partials in a rank-dependent order. Results won't be bit-identical across `P`, and iteration counts may differ slightly. Expected behavior (worth measuring and discussing rather than suppressing).

## Implementation notes

- Use the uppercase `mpi4py` buffer API (`Allreduce`, `Iallreduce`) on NumPy arrays. The lowercase pickle-based calls are far slower and would corrupt timings.
- Timers must separate local compute from communication, so the crossover can be identified directly rather than inferred from the runtime.