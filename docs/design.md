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

### Sizing the scaling experiments

The correctness runs used `n = 500`, `d = 20`, and terminated in exactly 20 iterations at every rank count, not because the tolerance was met, but because CG's finite-termination property bounds it at `d` steps in exact arithmetic.

This matters for experiment design. When iteration count is pinned by finite termination, it cannot vary with rank count, so the reordering effects of the `Allreduce` never reach the stopping test. The scaling runs need `d` large enough that CG stops on tolerance well before exhausting the Krylov space, otherwise the timing regime being measured is not the one the cost model describes.

### Interpreting the scaling curve

Plotting runtime against rank count produces a U-shaped curve: time falls while the shrinking `O(nd/P)` compute dominates, reaches a minimum, then rises as the growing `O(log P)` reduction takes over. The minimum locates the crossover.

The minimum is not necessarily the operating point of interest. Near it, large increases in `P` buy small reductions in time, so parallel efficiency is poor. The practically useful point is often the *knee* (where the curve begins to flatten), which captures most of the available speedup at substantially better efficiency. Both are worth analyzing: the time-optimal configuration and the efficiency-optimal one answer different questions depending on the practical application.

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

### BLAS thread oversubscription

Timing runs must pin BLAS to one thread per MPI rank (`OMP_NUM_THREADS=1`). NumPy's OpenBLAS backend otherwise sizes its thread pool for the whole machine independently in every rank, so `P` ranks each spawn a full-machine pool and the node is massively oversubscribed.

Measured on the desktop at `n = 100000`, `d = 200`, `P = 4`: unpinned wall time 6.77 s versus 0.62 s pinned, a 10.8x penalty. The damage appears mostly in the communication column (3.56 s versus 0.075 s) because a rank cannot enter the reduction until its local compute finishes, so descheduled threads show up as arrival skew rather than as compute time. This is a useful reminder that measured `Allreduce` time includes load-imbalance skew, not only network cost.

### First strong-scaling results (desktop)

`n = 100000`, `d = 200`, `lam = 1e-3`, 5 repetitions, one BLAS thread per rank. CPU: AMD Ryzen 7 8700F, 8 physical cores, 2 threads per core.

| P | wall (s) | compute (s) | comm (s) | efficiency |
|---|---|---|---|---|
| 1 | 1.456 | 1.451 | 0.002 | 1.00 |
| 2 | 0.830 | 0.820 | 0.011 | 0.88 |
| 4 | 0.587 | 0.564 | 0.053 | 0.62 |
| 8 | 0.553 | 0.497 | 0.092 | 0.33 |
| 16 | 0.937 | 0.494 | 0.519 | 0.10 |

Note: efficiency is measured as

$$
E(P) = \frac{T(1)}{P \cdot T(P)}.
$$

The predicted U-shape appears, with the runtime minimum at `P = 8`. The knee is at `P = 4`: moving from 4 to 8 ranks reduces wall time by 6% while halving parallel efficiency, so `P = 4` is the better operating point. Its efficiency of 0.62 falls in the 50-60% range suggested as a practical target.

Two results that contradict the cost model:

**Local compute does not scale as `1/P`.** Measured compute speedup saturates at about 2.9x and is essentially flat from `P = 8` to `P = 16`. The dense matrix-vector product loads 8 bytes per 2 flops, so it is memory-bandwidth bound, and all cores share one memory controller. The `1/P` term in the cost model holds only while the machine has spare bandwidth.

**The `P = 16` point measures SMT contention, not scaling.** With 8 physical cores, `P = 16` places two ranks per core. Compute cannot improve, and the 5.6x jump in measured communication reflects hyperthread pairs arriving at the reduction at unpredictable times rather than any increase in network cost.

Local sweeps should therefore be capped at `P = 8`. Whether the bandwidth plateau persists on distributed-memory hardware, where each node has its own memory controller, is a question for the work on CARC. To be developed further.

### Rank placement and measurement variance

Early CARC runs gave bimodal timings at P=8 and P=16: each run landed on either a fast value or a slow one, with nothing in between. Per-rank timings showed that in the slow runs some ranks computed up to 2x slower than others, and that the faster ranks spent the difference waiting inside the `Allreduce`.

The cause was rank placement. Hopper nodes have two sockets of 16 cores, each with its own memory controller. Unpinned ranks are placed by the kernel, and an uneven split across sockets leaves the crowded socket's ranks sharing less memory bandwidth - which this bandwidth-bound solver feels directly. The effect never appeared at P=32, where every core is used and the split is necessarily even.

`srun --cpu-bind=cores` had no effect here; every rank still reported all 32 CPUs. Pinning explicitly with `os.sched_setaffinity(0, {rank})` worked. CPU numbering alternates between sockets, so pinning rank r to CPU r splits ranks evenly for any P. The pin happens before the data is scattered, so each rank's block is allocated in its own socket's memory.

With pinning, per-rank compute varies by at most 6% and three independent runs agree within 2%. All reported timings use this configuration.

### Prediction: pipelined CG in this layout

Standard distributed CG performs one blocking `Allreduce` per iteration, inside the operator. Pipelined CG (Ghysels & Vanroose Algorithm 3) issues that reduction non-blocking and overlaps it with local work. In the row-distributed, replicated-vector layout the only work available for the overlap window is four AXPYs on `d`-vectors: `s`, `p`, `x` and `r`. At `d = 200` that is roughly 800 flops, against a reduction latency of tens of microseconds.

Pipelined CG also performs `k + 1` operator applications for `k` iterations, since `w_0 = A r_0` must be computed before the loop, and adds four vector updates per iteration relative to standard CG.

Prediction, recorded before measurement: exposed communication time (the duration of `Wait`) will be essentially unchanged, because the overlap window is too small to cover the reduction latency. Total runtime will be slightly worse than standard CG, by the cost of the extra operator application and the extra AXPYs. Iteration counts should match standard CG closely; the paper reports about 3% more on average across the test matrices they used for experiments.

### Result: pipelined CG does not improve on standard CG in this layout

Measured on Hopper, `d = 200`, 20 repetitions per configuration, three independent jobs (4328402, 4328403, 4328404) on three different nodes, one BLAS thread per rank, ranks pinned by node-local index.

Across all 54 paired comparisons (three problem sizes, six rank counts, three runs) pipelined CG was slower than standard CG in every case, typically by 2-4% in wall time. Iteration counts were consistently one to two higher (91 against 93 at `P = 1`, about 2%), close to the 3% average reported by Ghysels and Vanroose across their test matrices.

Overlap does occur. At `n = 50000`, `P = 32`, exposed communication falls from 7 ms to 5 ms, consistently across all three runs. It is simply never enough to offset the cost: pipelined CG performs `k + 1` operator applications for `k` iterations and four additional vector updates per iteration.

This confirms the prediction recorded before measurement. The structural reason is the one given in the  Ghysels & Vanroose reading note: the method hides a dot-product reduction behind an independent operator application, but in this layout the dot products require no communication at all, and the only reduction sits inside the operator with every subsequent quantity depending on its result.

### Prediction: weak scaling

Weak scaling fixes the work per rank and grows the problem with the rank count: `n = 15625 * P`, so each rank holds 15,625 rows at every `P`. Ideal behavior is constant runtime, since each rank performs the same local work regardless of how many ranks there are.

Two effects should push runtime up as `P` grows. The `Allreduce` cost grows as `log P`, though the strong-scaling runs showed exposed communication is only a few percent of runtime at these sizes. The larger effect should be memory bandwidth: at `P = 1` a single core has the node's memory controllers to itself, while at `P = 32` all 32 cores share them, and the dense matrix-vector product is bandwidth-bound. The earlier single-node results showed compute time per rank rising with occupancy for exactly this reason.

Prediction, recorded before measurement: weak-scaling efficiency will degrade substantially, with most of the degradation attributable to memory bandwidth rather than communication. The degradation should be visible in the `compute` column, not only in `wall`, which distinguishes it from a communication-driven explanation.

### Result: weak scaling

Measured on Hopper, 15,625 rows per rank, `d = 200`, three independent jobs (IDs 4328414, 4328415, 4328416), one BLAS thread per rank, ranks pinned by node-local index.

| P | wall (s) | compute (s) | comm (s) | efficiency |
|---|---|---|---|---|
| 1 | 0.218 | 0.215 | 0.001 | 1.00 |
| 2 | 0.229 | 0.222 | 0.003 | 0.95 |
| 4 | 0.326 | 0.321 | 0.007 | 0.67 |
| 8 | 0.382 | 0.372 | 0.020 | 0.57 |
| 16 | 0.461 | 0.450 | 0.029 | 0.47 |
| 32 | 0.714 | 0.703 | 0.046 | 0.31 |

Medians of three runs, which agree within about 2%. Efficiency is $`T(1)/T(P)`$, since the work per rank is constant.

The prediction holds. Of the 0.50 s increase in wall time from `P = 1` to `P = 32`, compute accounts for 0.49 s; communication reaches only 6% of wall time. Each rank performs identical work at every `P`, so the 3x rise in per-rank compute time is memory-bandwidth contention, not communication.

Regarding the dynamics, runtime is nearly flat from `P = 1` to `P = 2` and jumps 45% at `P = 4`. Because rank `r` is pinned to CPU `r` and CPU numbering alternates between sockets, `P = 2` places one rank on each socket and neither socket's memory controllers are shared. Contention begins at `P = 4`, when each socket first holds two ranks. The controlling variable is ranks per memory domain.

The `P = 32` configuration (`n = 500000`) coincides with the `P = 32` point of the strong-scaling study, and both measure 0.71 s.

Limitation: this is weak scaling within a single node, so it measures intra-node bandwidth contention rather than distributed-memory scaling. A multi-node weak scaling run at fixed ranks per node is the natural follow-up.

### Prediction: regularization, conditioning and CG iterations

The convergence rate of CG depends on the condition number of `A = XᵀX + λI`:

$$
\kappa(A) = \frac{\sigma_{\max}^2 + \lambda}{\sigma_{\min}^2 + \lambda}.
$$

For the test problem (`n = 20000`, `d = 500`, generator `cond = 1e6`), the eigenvalues of `XᵀX` span `σ²_min = 1e-6` to `σ²_max = 1`. Regularization improves conditioning whenever `λ` is comparable to or larger than `σ²_min`: for `λ ≫ σ²_min`, `κ(A) ≈ 1/λ`; for `λ ≪ σ²_min`, `κ(A) ≈ κ(XᵀX) = 1e6` and stops changing.

The standard bound gives roughly `½ √κ ln(2/ε)` iterations, about `12 √κ` at `ε = 1e-10`. In exact arithmetic CG also terminates in at most `d = 500` iterations, and the bound reaches 500 at `κ(A) ≈ 1700`, i.e. `λ ≈ 6e-4`.

Prediction, recorded before measurement:

1. `λ` above ~`6e-4`: iterations grow like `√κ(A)`, a slope of ½ on a log-log plot of iterations against `κ(A)`.
2. `λ` from ~`6e-4` down to ~`1e-6`: the bound exceeds `d`. Iterations flatten near `d = 500`, as finite termination caps them.
3. `λ` below ~`1e-6`, including `λ = 0`: `κ(A)` is fixed at about `1e6`, so iterations are flat.

The `√κ` bound depends only on the extreme eigenvalues, while CG responds to the whole spectrum, so the generator's spacing of singular values will shift the curve.

Two key points worth having in mind. The attainable accuracy is roughly `κ · ε_machine ≈ 1e-10`, at the tolerance itself, so the recursively updated residual may keep shrinking while the true residual `‖b − Ax‖ / ‖b‖` stalls, we record both. And `maxiter = 20d`, so that the cap does not determine the iteration count.

### Result: regularization, conditioning and CG iterations

Single run, `n = 20000`, `d = 500`, generator `cond = 1e6` (`σ²_min = 1e-6`, `σ²_max = 1`), tolerance `1e-10`, `maxiter = 10000`. Data: `results/final/lambda_sweep.jsonl`.

| λ | κ(A) | iterations |
|---|---|---|
| 1 | 2.0 | 13 |
| 1e-1 | 11 | 34 |
| 1e-2 | 101 | 91 |
| 1e-3 | 1,000 | 240 |
| 1e-4 | 9,902 | 626 |
| 1e-5 | 90,910 | 1,608 |
| 1e-6 | 500,000 | 3,285 |
| 1e-7 | 909,091 | 4,253 |
| 1e-8 | 990,099 | 4,376 |
| 0 | 1,000,000 | 4,365 |

Every run converged, and the true residual `‖b − Ax‖ / ‖b‖` agrees with CG's recursively updated residual to about six significant digits throughout, so the attainable-accuracy issue did not show up in the results.

**Regime 1 (√κ growth): correct, with a slightly lower slope.** On a log-log scale, iterations grow with slope 0.42–0.44 in `κ(A)`, a little below the ½ of the classical bound. The bound is pessimistic by a factor of 1.6–2.7: it predicts about 12,000 iterations at `κ = 1e6`, against 4,365 measured.

**Regime 2 (flattening at d = 500): incorrect.** Nothing happens at `d`. The iteration count crosses 500 near `λ = 2e-4` and continues on the same slope to about 3,300, then flattens only because `κ(A)` itself stops changing. At `λ = 0`, CG takes 4,365 iterations on a 500-dimensional problem, 8.7x the exact-arithmetic limit.

The explanation is finite precision. Termination in at most `d` steps depends on the search directions remaining exactly `A`-orthogonal, and rounding destroys that orthogonality. Greenbaum (1989) showed that finite-precision CG behaves like exact CG applied to a larger matrix whose eigenvalues cluster around those of `A`, so its convergence is governed by conditioning rather than dimension. The data show this directly: the `√κ` trend ignores `d`.

**Regime 3 (flat below σ²_min): correct.** Below `λ = 1e-7` the counts lie between 4,253 and 4,376. The 0.25% decrease at the smallest `λ` values is within the variation of a single deterministic run and is not a trend.

Implication for communication cost: iteration counts span a factor of 336 across this sweep, and in the row-distributed layout each iteration costs one global reduction. Reducing `κ(A)` through preconditioning is therefore the most direct lever on communication, which motivates the sketch-preconditioned experiment.

Reference: A. Greenbaum, "Behavior of slightly perturbed Lanczos and conjugate-gradient recurrences," *Linear Algebra and its Applications* 113 (1989), 7–63.