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

**Regime 1 (√κ growth): correct, with a slightly lower slope.** On a log-log scale, iterations grow with slope 0.42–0.44 in `κ(A)` (for κ ≳ 10), a little below the ½ of the classical bound. The bound is pessimistic by a factor of 1.6–2.7: it predicts about 12,000 iterations at `κ = 1e6`, against 4,365 measured.

**Regime 2 (flattening at d = 500): incorrect.** Nothing happens at `d`. The iteration count crosses 500 near `λ = 2e-4` and continues on the same slope to about 3,300, then flattens only because `κ(A)` itself stops changing. At `λ = 0`, CG takes 4,365 iterations on a 500-dimensional problem, 8.7x the exact-arithmetic limit.

The explanation is finite precision. Termination in at most `d` steps depends on the search directions remaining exactly `A`-orthogonal, and rounding destroys that orthogonality. Greenbaum (1989) showed that finite-precision CG behaves like exact CG applied to a larger matrix whose eigenvalues cluster around those of `A`, so its convergence is governed by conditioning rather than dimension. The data show this directly: the `√κ` trend ignores `d`.

**Regime 3 (flat below σ²_min): correct.** Below `λ = 1e-7` the counts lie between 4,253 and 4,376. The 0.25% decrease at the smallest `λ` values is within the variation of a single deterministic run and is not a trend.

Implication for communication cost: iteration counts span a factor of 336 across this sweep, and in the row-distributed layout each iteration costs one global reduction. Reducing `κ(A)` through preconditioning is therefore the most direct lever on communication, which motivates the sketch-preconditioned experiment.

Reference: A. Greenbaum, "Behavior of slightly perturbed Lanczos and conjugate-gradient recurrences," *Linear Algebra and its Applications* 113 (1989), 7–63.

### Prediction: sketch-preconditioned CG

Method: sketch `X` with a Gaussian matrix `S` of size `s × n`, QR-factor the stacked matrix `[SX; √λ I] = QR`, and precondition CG with `M = RᵀR`, so that `M ≈ XᵀX + λI = A`. Applying `M⁻¹` is two triangular solves with `R`.

Setup: the same problem as the regularization sweep (`n = 20000`, `d = 500`, `cond = 1e6`), the same tolerance `1e-10`, and the same `λ` grid.

For a Gaussian sketch, the singular values of the sketched matrix lie roughly in `[1 − √(d/s), 1 + √(d/s)]`, so the preconditioned condition number is about

$$
\kappa(M^{-1}A) \approx \left(\frac{1 + \sqrt{d/s}}{1 - \sqrt{d/s}}\right)^2,
$$

which gives

- `s = 2d` → `κ ≈ 34`
- `s = 4d` → `κ ≈ 9`

In the unpreconditioned sweep, `κ = 32` took 55 iterations and `κ = 11` took 34.

Prediction, recorded before measurement:

1. With preconditioning, iteration counts are roughly constant across the whole `λ` grid, about 30 to 55 depending on `s`, against 13 to 4,365 without. At `λ = 0` that is a reduction of roughly 100×, and in the row-distributed layout, roughly 100× fewer global reductions.
2. For large `λ`, preconditioning is worse. At `λ = 1`, plain CG needs only 13 iterations because `κ(A) = 2`, below what the preconditioner achieves. The crossover should fall where `κ(A)` matches the preconditioned condition number, between about `λ = 3e-2` and `λ = 1e-1` depending on `s`.

Iteration counts are not the whole cost. The sketch and QR are paid once: about `2snd` flops for a dense Gaussian sketch and `O(sd²)` for the QR. At `s = 2d` the sketch alone costs about as much as 500 iterations of matrix-vector products, so wall time is recorded separately. With a dense Gaussian sketch, setup may consume much of the savings at moderate `λ`, which motivates the sparse sign sketch as the next step.

### Result: sketch-preconditioned CG

Same problem and grid as the regularization sweep, Gaussian sketch with `s = 2d` and `s = 4d`, one BLAS thread. Data: `results/final/lambda_sweep_*`.

| λ | none | s=2d | s=4d | none (s) | s=2d (s) | s=4d (s) |
|---|---|---|---|---|---|---|
| 1 | 13 | 10 | 9 | 0.065 | 0.52 | 1.04 |
| 1e-2 | 91 | 22 | 17 | 0.45 | 0.57 | 1.13 |
| 1e-3 | 240 | 28 | 20 | 1.23 | 0.60 | 1.05 |
| 1e-5 | 1,590 | 42 | 27 | 7.9 | 0.73 | 1.08 |
| 0 | 4,380 | 58 | 31 | 21.4 | 0.80 | 1.09 |

Iteration counts in the first three columns. Total time (setup plus solve) in the last three.

**Reduction at λ = 0: confirmed.** Iterations fall 75× (`s = 2d`) and 141× (`s = 4d`), so the row-distributed solver would need that many fewer global reductions. Wall time falls 27× and 20×.

**Plateau height: confirmed.** Both arms plateau below `λ ≈ 1e-7`. The measured `κ(M⁻¹A) = 7.7` at `s = 4d` corresponds, in the unpreconditioned sweep, to about 30 iterations (the plateau is 31). The predicted `κ ≈ 34` at `s = 2d` corresponds to 55 (the plateau is 58).

**Constant iterations across λ: refuted.** Iterations rise from about 10 at `λ = 1` to the plateau. The preconditioner is built from `[SX; √λ I]`, and the `√λ I` block is exact, so when `λ` dominates `XᵀX` the sketch error is negligible and `κ(M⁻¹A)` approaches 1. As `λ` shrinks, sketch error dominates and `κ(M⁻¹A)` rises to its asymptotic value.

**Preconditioning loses at large λ: refuted in iterations, but confirmed in wall time.** Preconditioned CG never needs more iterations, for the reason above. But setup costs 0.46 s (`s = 2d`) and 0.94 s (`s = 4d`), so plain CG is faster for `λ` above about `5e-3` and `1.5e-3` respectively (a smaller crossover than predicted), because setup is expensive.

**Setup dominates.** At `s = 4d` setup is 85% of total time, which is why `s = 2d` is faster overall at `λ = 0` despite twice the iterations. The dense Gaussian sketch costs `O(snd)`. A sparse sign sketch would cut this to `O(ζnd)`.

**Note on reproducibility.** The unpreconditioned arm differs from the original sweep by 1–3% at high `κ` (e.g. 616 versus 626 iterations at `λ = 1e-4`). The original used multi-threaded BLAS, and this run used one thread. Summation order changes rounding, and at high `κ` CG's iteration count is sensitive to it.

### Prediction: sparse sign sketch preconditioning

The Gaussian-preconditioned sweep showed setup dominating total time: 0.46 s at `s = 2d`, almost all of it the dense product `SX`, which costs about `2snd` flops. A sparse sign sketch places `ζ` nonzeros per column of `S`, each `±1/√ζ`, so `SX` costs about `2ζnd`. At `s = 1000` and `ζ = 2` that is roughly 500× cheaper. The remaining setup is the QR of the `(s + d) × d` stacked matrix, about `2(s + d)d²` flops, around 4% of the Gaussian sketch cost.

Setup: the same problem, tolerance and `λ` grid as before, `s = 2d`, with `ζ ∈ {1, 2, 4}`.

Prediction, recorded before measurement:

1. **Setup time** falls from about 0.46 s to under 0.1 s for every `ζ`.
2. **Iteration counts.** `ζ = 2` and `ζ = 4` match the Gaussian arm within a few iterations (plateau ≈ 58), and `ζ = 1` needs noticeably more.
3. **Wall-time crossover** with unpreconditioned CG moves to larger `λ`, since the setup floor drops by several times: preconditioning should pay off from about `λ ≈ 1e-2` or above, rather than `5e-3`.

The open question is item 2. In my Randomized NLA research project (H8 in github.com/daviddavilad/randomized-numerical-linear-algebra), I measured low-rank approximation accuracy, where `ζ ≥ 2` matched Gaussian. Preconditioning instead requires the sketch to preserve the geometry of the entire column space of `X`, and the OSNAP theory guarantees that only for `ζ` growing like `log d` and `s` above roughly `d log d`. Notice that `s = 2d` with `ζ = 2` lies below those guarantees.

### Result: sparse sign sketch preconditioning

Same problem, tolerance and `λ` grid as before, `s = 2d`, one BLAS thread, one sketch draw per arm. Data: `results/final/lambda_sweep_sparse_s2_z*.jsonl`.

| | setup (s) | iters, λ = 1 | iters, λ = 1e-4 | iters, λ = 0 | total at λ = 0 (s) |
|---|---|---|---|---|---|
| none | — | 13 | 616 | 4,380 | 21.4 |
| Gaussian | 0.46 | 10 | 35 | 58 | 0.80 |
| sparse, `ζ = 1` | 0.025 | 10 | 35 | 57 | 0.36 |
| sparse, `ζ = 2` | 0.031 | 10 | 35 | 58 | 0.45 |
| sparse, `ζ = 4` | 0.033 | 10 | 35 | 60 | 0.36 |

**Setup time: confirmed.** 0.025–0.035 s for every `ζ`, about 15× below the Gaussian sketch.

**Iteration counts: partly refuted.** `ζ = 2` and `ζ = 4` match the Gaussian arm within two iterations at every `λ`, as predicted. But `ζ = 1` matches as well, contrary to the expectation and the worst-case OSNAP bounds: a CountSketch with `s = 2d` preconditions this problem as well as a dense Gaussian sketch. One-iteration differences are within the variation of a single sketch draw.

The likely reason is coherence. The worst-case guarantees are driven by matrices whose leverage is concentrated in a few rows, which a one-nonzero sketch can miss. The test matrix is `X = UΣVᵀ` with Haar-random `U`, so leverage is spread evenly across rows (the most favorable case for sparse sketches). The claim supported here is therefore limited: on incoherent problems `ζ = 1` suffices. Behavior on coherent problems is untested.

**Wall-time crossover: confirmed.** With `ζ = 1`, plain CG is faster only at `λ = 1` (0.065 s against 0.087 s); the two tie near `λ ≈ 0.3`, and preconditioned CG is faster at every smaller `λ`. The prediction was `λ ≈ 1e-2` or above. At `λ = 0`, sparse preconditioning is 59× faster end to end than plain CG.

Solve times vary by up to about 25% between arms with identical iteration counts (0.33 s against 0.42 s), which is run-to-run machine variation. Therefore, iteration counts are the reliable comparison.

### Prediction: sketch quality across seeds

The sparse-sketch sweep found `ζ = 1` as effective as a Gaussian sketch, but on the evidence of one sketch draw per arm and iteration counts, which are a coarse measure: CG iterations grow roughly like `√κ`, so a preconditioner with a 20% higher condition number would cost only about 10% more iterations, which at 58 iterations is within single-draw variation. This experiment measures `κ(M⁻¹A)` directly.

Setup: the same problem as the sweeps (`n = 20000`, `d = 500`, `cond = 1e6`), `λ = 0`, `s = 2d`, 30 independent sketch draws per arm, for the Gaussian sketch and sparse sign sketches with `ζ = 1, 2, 4`.

Prediction, recorded before measurement:

1. The Gaussian arm's mean `κ(M⁻¹A)` lies near, and somewhat below, the asymptotic estimate of 34, as the `s = 4d` test (7.7 against a predicted 9) came in below its estimate.
2. `ζ = 2` and `ζ = 4` match the Gaussian mean within two standard errors.
3. `ζ = 1` also matches the Gaussian mean within two standard errors, so the single-draw sweep result holds.

"Matches" means the difference in means is within two standard errors of zero. A difference that is statistically clear but small, such as under 10% in `κ`, would still be reported as a real but practically minor penalty.

#### Result

Run: `scripts/sketch_quality.py` (n = 20000, d = 500, s = 2d, κ(XᵀX) = 1e6, λ = 0), 30 seeds per arm. Data: `results/final/sketch_quality.jsonl`.

| arm            | mean κ(M⁻¹A) | SD   | SE   | diff vs Gaussian | diff / SE_combined |
|----------------|--------------|------|------|------------------|--------------------|
| Gaussian       | 32.24        | 0.99 | 0.18 | —                | —                  |
| sparse, ζ = 1  | 32.29        | 1.11 | 0.20 | +0.05 (+0.2%)    | 0.19               |
| sparse, ζ = 2  | 32.51        | 1.22 | 0.22 | +0.27 (+0.8%)    | 0.94               |
| sparse, ζ = 4  | 32.66        | 1.36 | 0.25 | +0.41 (+1.3%)    | 1.34               |

All three predictions were confirmed:

1. The Gaussian mean (32.2) is close to the asymptotic estimate ((1 + √(d/s)) / (1 − √(d/s)))² ≈ 34.0 and slightly below it.
2. ζ = 2 and ζ = 4 are within two standard errors of the Gaussian mean.
3. ζ = 1 is within two standard errors as well, and it is the closest of the three (0.19 SE). Every difference is far below the 10% threshold for a practical difference.

This rules out the concern that iteration counts were too coarse to detect a ζ = 1 penalty. Measuring κ directly, the experiment could have detected a difference of about 0.55 (two combined standard errors, ≈ 1.7% of κ), and none appeared.

Things this data does not support:

- The means go up with ζ (32.29 → 32.51 → 32.66), but every step is within noise. This shows no detectable difference.
- The sparse arms have slightly larger spreads (SD 1.1–1.4 vs 0.99) and a few high draws (37.3 at ζ = 2, 36.4 at ζ = 4). With 30 draws this is an observation, not a finding.

Scope: this is one fixed X with Haar-random singular vectors, so it is incoherent by construction. Sparse sketches with small ζ are known to do worse on coherent matrices, so this result does not show that ζ = 1 is generally as good as a Gaussian sketch. The coherent-matrix experiment would test that.

### Conditioning or precision? The plateau at cond = 1e4

The unpreconditioned sweep at `cond = 1e6` flattens below `λ ≈ 1e-7`. Two explanations fit:

- **Conditioning.** Once `λ ≪ σ²_min = 1/cond`, adding `λ` no longer changes `κ(A) = (1 + λ)/(1/cond + λ)`, so the iteration count stops changing. The plateau location scales with `1/cond`.
- **Precision.** If arithmetic were single precision, `λp` would be lost to rounding once `λ` fell below roughly `ε_single · ‖XᵀX‖ ≈ 6e-8`. The plateau location would be fixed by machine epsilon, independent of `cond`.

At `cond = 1e6` both explanations put the plateau near `1e-7`, so that sweep cannot tell them apart. The solver runs in float64 (`ε ≈ 1e-16`), so the precision explanation is not expected to apply. This experiment tests that directly.

Run: `scripts/lambda_sweep.py --cond 1e4 --precond none`, same `n`, `d`, `λ` grid, tolerance, and seed as the `cond = 1e6` sweep.

Prediction:

1. **Location.** The plateau moves to `σ²_min = 1e-4`. Using the shape of the `cond = 1e6` curve, iterations at `λ = 1e-4` are about 75% of the plateau value, and are flat (within 3%) for `λ ≲ 3e-6`.
2. **Height.** The plateau is at `κ(A) ≈ 1e4`. The `cond = 1e6` sweep needed about 630 iterations at `κ ≈ 1e4` (`λ = 1e-4`), so the plateau should be roughly 450–800 iterations, about 7x below the `cond = 1e6` plateau of ~4,370.
3. **Large λ.** For `λ ≳ 1e-2`, `κ(A)` is nearly the same for both values of `cond`, so iteration counts agree within about 10%.

Falsification: if the `cond = 1e4` sweep stays on the `cond = 1e6` curve down to `λ ≈ 1e-7` and flattens there, the plateau is set by precision, not conditioning.

#### Result

Run: `scripts/lambda_sweep.py --cond 1e4 --precond none`, same `n`, `d`, `λ` grid, tolerance, and seed as the `cond = 1e6` sweep. Data: `results/final/lambda_sweep_none_cond1e4.jsonl`.

The plateau moves with `cond`. The conditioning explanation is confirmed and the precision explanation is refuted.

**Location: confirmed.** The plateau is about 721 iterations.

| λ               | iterations | % of plateau |
|-----------------|------------|--------------|
| 1e-4 (= σ²_min) | 532        | 74%          |
| 1e-5            | 690        | 96%          |
| 3e-6            | 708        | 98%          |
| ≤ 1e-6          | 713–722    | flat         |

The prediction was about 75% at `σ²_min` and flat within 3% below `3e-6`. Under the precision explanation, at `λ = 1e-5` this run would still follow the `cond = 1e6` curve, at about 1,600 iterations. It took 690.

**Height: confirmed.** The plateau of 721 is inside the predicted 450–800 range. The ratio to the `cond = 1e6` plateau (~4,370) is 6.1x, a little below the predicted ~7x. That corresponds to an effective slope of about 0.39 between the two plateaus.

**Large λ: confirmed.** For `λ ≥ 1e-2`, iteration counts differ by 0–5.5% between the two sweeps.

**Observation.** For `λ ≤ 1e-2`, the `cond = 1e4` problem needs a few percent more iterations than the `cond = 1e6` problem at equal or slightly smaller `κ(A)`. The same holds at the plateau: 721 iterations at `κ = 1e4`, against 626 when the `cond = 1e6` problem had `κ ≈ 1e4` (`λ = 1e-4`). A plausible explanation is the shape of the spectrum. Eigenvalues of `XᵀX` below `λ` cluster near `λ` once it is added, and CG handles a tight cluster almost like a single eigenvalue. At `λ = 1e-2`, about 67% of the `cond = 1e6` spectrum is in that cluster, against 50% for `cond = 1e4`. The `√κ` bound is worst-case and ignores how the eigenvalues are distributed. This is a hypothesis and has not been tested.

**Iterations exceed `d` at moderate conditioning.** The plateau of 721 is above `d = 500` even at `κ = 1e4`, so loss of orthogonality (Greenbaum 1989) affects convergence at moderate conditioning too, not only at `κ = 1e6`.

### Design: distributed sketch-preconditioned CG

The serial preconditioner builds `R` from the QR of `[SX; √λ I]`. To distribute it, split the sketch by columns to match the row partition of `X`: `S = [S_1 | S_2 | … | S_P]`, where `S_r` is `s × n_r`. Then SX = Σ_r S_r X_r

Setup on each rank:

1. Compute `S_r X_r` (`s × d`) locally.
2. `Allreduce` the partial sketches, so every rank holds `SX`.
3. Compute the QR of `[SX; √λ I]` locally, so every rank holds its own copy of `R`.

`apply_Minv` is two triangular solves with `R` on replicated `d`-vectors, so it needs no communication. PCG still performs exactly one `Allreduce` per iteration, inside the operator. The preconditioner adds one `Allreduce` in total, during setup.

#### Decision: sparse sketches only

A Gaussian `S_r` is `s × n_r`, which for `s = 2d` is twice the size of `X_r`, and computing `S_r X_r` costs `O(s n_r d)`, the compute of roughly `d` CG iterations. A sparse sign sketch costs `O(ζ n_r d)`, under one iteration. The sketch-quality experiment found `ζ = 1` as effective as a Gaussian sketch on this problem, so dropping the Gaussian option costs nothing measurable.

#### Decision: the same sketch for every P

If each rank drew its own random sketch, `S` would change with `P`, and so would `κ(M⁻¹A)` and the iteration count, confounding the strong-scaling measurements. Instead, rank 0 generates the global `positions` (`n × ζ`) and `signs` and scatters them with `X`. This adds `O(nζ)` data, small next to `X`'s `O(nd)`, and makes `S` identical for every `P`. Each rank builds its `S_r` as a sparse `s × n_r` matrix from its slice.

#### Decision: every rank computes the QR

The QR of the `(s + d) × d` matrix costs `O(sd²)` and is independent of `n` and `P`. Computing it on every rank avoids a second communication step (QR on rank 0, then broadcast `R`). The cost is a serial term in setup that does not shrink with `P`. Distributing this QR is not worthwhile. TSQR needs local blocks with at least `d` rows, and `[SX; √λ I]` is only `3d × d`, so at `P = 32` each block would have about 47 rows. Each level of the TSQR reduction tree also factors a `2d × d` stack at about the cost of the full QR. The better options are to factor once and broadcast `R`, or to share one `R` per node.

#### Setup cost

| component           | cost              | scales with P      |
|---------------------|-------------------|--------------------|
| local sketch        | `O(ζ n d / P)`    | shrinks as `1/P`   |
| `Allreduce` of `SX` | `s·d` doubles     | grows as `log P`   |
| QR                  | `O(s d²)`         | constant (serial)  |

For `d = 500`, `s = 2d`: the `Allreduce` moves `5×10⁵` doubles (4 MB) once, and the QR is about `7×10⁸` flops on every rank. As `P` grows, the local sketch shrinks and the QR remains, so setup time should approach a floor set by the QR.

#### Correctness test

For the same `positions`, `signs`, and `λ`, the distributed `apply_Minv` must match the serial one to about `1e-12` for any `P`. The test compares `M⁻¹r`, not `R`, because `R` is unique only up to the signs of its rows.

Interface:

    make_distributed_sketch_preconditioner(X_local, lam, s, positions_local, signs_local, comm)
        -> apply_Minv, timings {"sketch", "comm", "qr"}

The serial and distributed implementations share three helpers: `sparse_sign_sketch` generates positions and signs, `sparse_sign_matrix` constructs the corresponding CSC matrix from global or local arrays, and `qr_preconditioner` constructs the inverse preconditioner from the completed sketch product. The serial function retains its existing signature, and the distributed implementation scatters the generated positions and signs across ranks. The refactored serial function produces bitwise-identical `M⁻¹r` for the same seed (checked for `ζ = 1` and `ζ = 2`), so all earlier results stay reproducible from the current code.

### Prediction: strong scaling of distributed sketch-preconditioned CG

Run: `scripts/precond_scaling_carc.sh`, Hopper, one node, P = 1, 2, 4, 8, 16, 32. `d = 500`, `cond = 1e6`, `λ = 0`, sparse sign sketch with `s = 2d`, `ζ = 1`, sketch seed fixed so `S` is identical for every `P`. Two problem sizes: `n = 50,000` and `n = 500,000`. We run three independent jobs and measure the median.

Cost model:

- Setup: local sketch `∝ n/P`, one `Allreduce` of `s·d` doubles, and a QR replicated on every rank that does not depend on `P`.
- Per iteration: operator compute `∝ n d / P`, one `Allreduce` of `d` doubles, and `apply_Minv`, which is replicated and does not depend on `P`. On the desktop at `d = 200`, `apply_Minv` cost 11% of operator compute, not the 2% a flop count predicts, so per-call overhead is part of its fixed cost.

Prediction:

1. **Iterations.** The same at every `P` (within ±1, from summation order), and about 55–60 at both sizes, since `κ(M⁻¹A)` depends on `s/d`, not on `n`.
2. **Large problem (`n = 500,000`).** Setup plus `apply_Minv` stay below 5% of total time at every `P`. Total-time scaling has the same shape as the unpreconditioned strong scaling, limited by memory bandwidth.
3. **Small problem (`n = 50,000`).** The fixed costs grow as a fraction of total time with `P`. By `P = 32`, QR plus `apply_Minv` exceed 25% of total time, and speedup of total time levels off at a lower `P` than speedup of the solve alone.
4. **Setup components.** Sketch time falls roughly as `1/P`. Setup `Allreduce` time grows with `P` but stays below the QR. QR time stays roughly constant. It may rise by up to about 50% at `P = 32`, because 32 ranks factor simultaneously and share cache and memory bandwidth.
5. **`apply_Minv` per iteration** stays roughly constant across `P`.

The main claim is item 3: in this design, the replicated QR and triangular solves set a floor on strong scaling for small problems. If the fixed costs stay below 10% of total time at `n = 50,000`, `P = 32`, the floor is not a practical concern at this `d`.

#### Result

Data: `results/final/precond_scaling_carc_4359211.jsonl`, `…_4359213.jsonl`, `…_4359214.jsonl`. Hopper, node `hopper010` (Xeon Gold 6226R, 2 × 16 cores, 22.5 MB L3 per socket), with three independent jobs. Values are medians across the three jobs. Every configuration agrees across jobs within 4%, most within 1%. "Fixed" is QR plus `apply_Minv` as a share of total time. All 36 runs converged with identical iteration counts on every rank.

`n = 50,000`:

| P  | total (ms) | solve | setup | sketch | QR   | `apply_Minv` | fixed | speedup (total) | speedup (solve) |
|----|------------|-------|-------|--------|------|--------------|-------|-----------------|-----------------|
| 1  | 1756       | 1701  | 55    | 24.9   | 29.2 | 26.6         | 3%    | 1.00            | 1.00            |
| 2  | 936        | 890   | 45    | 13.2   | 29.6 | 27.6         | 6%    | 1.88            | 1.91            |
| 4  | 510        | 469   | 41    | 7.4    | 30.9 | 29.5         | 12%   | 3.44            | 3.63            |
| 8  | 309        | 267   | 40    | 4.6    | 32.4 | 35.2         | 22%   | 5.68            | 6.37            |
| 16 | 227        | 180   | 46    | 3.7    | 37.4 | 34.6         | 32%   | 7.74            | 9.43            |
| 32 | 256        | 187   | 69    | 4.3    | 56.2 | 57.5         | 44%   | 6.85            | 9.09            |

`n = 500,000`:

| P  | total (ms) | sketch | QR   | `apply_Minv` | fixed | speedup (total) |
|----|------------|--------|------|--------------|-------|-----------------|
| 1  | 16728      | 242    | 30.2 | 29.2         | 0.4%  | 1.00            |
| 2  | 8763       | 123    | 29.8 | 29.3         | 0.7%  | 1.91            |
| 4  | 4495       | 64.2   | 30.6 | 36.4         | 1.5%  | 3.72            |
| 8  | 2324       | 35.0   | 32.2 | 37.3         | 3.0%  | 7.20            |
| 16 | 1450       | 27.8   | 37.5 | 36.9         | 5.1%  | 11.54           |
| 32 | 1303       | 31.2   | 56.0 | 70.9         | 9.7%  | 12.84           |

**1. Iterations: confirmed.** 58 at every `P` for `n = 50,000`. 57 or 58 depending on `P` for `n = 500,000`, identical across all three jobs. The ±1 variation is deterministic and comes from summation order, not noise.

**2. Large problem below 5%: refuted.** The fixed share is 5.1% at `P = 16` and 9.7% at `P = 32`. The cost model assumed the replicated terms were constant in `P`. At `P = 32` both nearly doubled (items 4 and 5).

**3. Small problem: floor confirmed, earlier levelling not resolved.** By `P = 32` the fixed costs are 44% of total time, above the predicted 25%. Total time is minimized at `P = 16` and increases by 13% at `P = 32`, so adding ranks makes the solver slower. However, solve-only speedup also peaks at `P = 16` (9.43, then 9.09), so this grid of `P` values cannot show that total time levels off at a lower `P`. After the peak, total time degrades faster (−11% vs −4% in speedup). Part of the fixed cost is inside the solve, since `apply_Minv` runs every iteration.

**4. Setup components: partly confirmed.**
- Sketch time falls as `1/P` up to `P = 8` (242 → 35 ms at `n = 500,000`), then stops falling and rises slightly at `P = 32`.
- Setup `Allreduce` grows with `P` (0.7 → 9 ms) and stays below the QR: confirmed.
- QR within +50%: refuted. It rises from 29 ms to 56 ms (+92%) at `P = 32`, the same at both problem sizes, as expected for a cost independent of `n`. The rise is gradual: +11% at `P = 8`, +28% at `P = 16`.

**5. `apply_Minv` per iteration roughly constant: refuted at `P = 32`.** It goes from 0.46 to 0.60 ms through `P = 16`, then 0.99 ms at `P = 32` (`n = 500,000`: 0.51 → 1.24 ms).

**Proposed explanation for 4 and 5 (not tested).** Both replicated terms jump when each socket goes from 8 to 16 ranks, independent of `n`. Each rank holds its own `R` (500 × 500 doubles = 2 MB): 8 copies per socket take 16 MB and fit in the 22.5 MB L3. 16 copies take 32 MB and do not, so at `P = 32` the triangular solves read `R` from DRAM. The QR's working matrix (1500 × 500 = 6 MB per rank) exceeds L3 at fewer ranks per socket, which fits its more gradual rise. This is consistent with the data but not demonstrated. Plain memory-bandwidth contention, as in the weak-scaling result, would also grow with ranks per socket.

**Conclusion.** Replicating the QR and the triangular solves on every rank is cheap at low `P` but sets a floor on strong scaling. At `n = 50,000`, the best total time is at `P = 16`. At `n = 500,000`, the same costs reach 10% of total time by `P = 32` and are rising. The model's assumption that replicated work costs the same at any `P` fails when ranks share cache and memory bandwidth.
