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