# Pipelined conjugate gradient

**Ghysels, P., & Vanroose, W. (2014).** Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm. *Parallel Computing*, 40(7), 224–238. [DOI](https://doi.org/10.1016/j.parco.2013.06.001)

Read: 2026-09-28 (pass 2)

## Problem

Krylov subspace methods suffer from expensive global synchronization steps arising from dot products and norm calculations on distributed-memory parallel machines.

Their fundamental operations are:

- Sparse matrix-vector products (SpMV).
- Vector additions and updates.
- Dot products and norm calculations.

These operations have different communication requirements. For sparse matrices originating from PDE discretizations, SpMV generally requires communication only between neighboring processors. Vector updates can be performed locally.

However, dot products require combining partial results from all processors through a global reduction. This introduces synchronization latency, which increasingly limits scalability as the processor count grows.

**Objective:** Reformulate preconditioned Conjugate Gradient (PCG) to hide global communication latency by overlapping global reductions with useful computations, particularly SpMV and preconditioner applications.

## Approach

### 1. Mathematical foundations: Krylov subspaces

Consider the linear system

$$
Ax=b,
$$

where $A$ is symmetric positive definite (SPD).

Starting from an initial approximation $x_0$, define the initial residual:

$$
r_0=b-Ax_0.
$$

The Krylov subspace of dimension at most $k$ is

$$
\mathcal{K}_k(A,r_0)
=
\operatorname{span}
\left\{
r_0,Ar_0,A^2r_0,\ldots,A^{k-1}r_0
\right\}.
$$

Each iteration potentially introduces another independent direction through repeated applications of $A$.

An important interpretation is that every vector in the Krylov subspace can be represented as

$$
v=p_{k-1}(A)r_0,
$$

where $p_{k-1}$ is a polynomial of degree at most $k-1$.

Thus, Krylov methods construct polynomial approximations using matrix-vector products rather than explicitly computing the inverse of $A$.

### 2. Standard Conjugate Gradient

CG solves $Ax=b$ by minimizing the quadratic objective

$$
f(x)=\frac{1}{2}x^TAx-b^Tx.
$$

Since

$$
\nabla f(x)=Ax-b,
$$

the residual is the negative gradient:

$$
r_k=b-Ax_k=-\nabla f(x_k).
$$

Rather than following the negative gradient directly, CG constructs search directions $p_k$ that are $A$-conjugate:

$$
p_i^TAp_j=0,
\qquad i\neq j.
$$

The resulting iterates satisfy

$$
x_k\in x_0+\mathcal{K}_k(A,r_0).
$$

At each iteration, CG minimizes the $A$-norm of the error over this affine Krylov subspace:

$$
x_k
=
\underset{x\in x_0+\mathcal{K}_k}{\arg\min}
\|x^*-x\|_A,
$$

where

$$
\|e\|_A=\sqrt{e^TAe},
\qquad x^*=A^{-1}b.
$$

The fundamental recurrences are

$$
\alpha_k
=
\frac{r_k^Tr_k}{p_k^TAp_k},
$$

$$
x_{k+1}=x_k+\alpha_kp_k,
$$

$$
r_{k+1}=r_k-\alpha_kAp_k,
$$

$$
\beta_{k+1}
=
\frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k},
$$

$$
p_{k+1}
=
r_{k+1}+\beta_{k+1}p_k.
$$

Interpretation:

- $p_k$ determines the search direction.
- $\alpha_k$ determines the step length.
- $\beta_{k+1}$ determines how the previous search direction contributes to the next one.

CG requires one SpMV per iteration and, in exact arithmetic, converges in at most $N$ iterations for an $N$-dimensional SPD problem.

### 3. Preconditioned Conjugate Gradient

Introduce an SPD preconditioner $M$ and consider

$$
M^{-1}Ax=M^{-1}b.
$$

The objective is to improve the spectral properties of the system, ideally making $M^{-1}A$ closer to the identity matrix.

Define the preconditioned residual

$$
u_k=M^{-1}r_k.
$$

The corresponding Krylov subspace is

$$
\mathcal{K}_k(M^{-1}A,u_0)
=
\operatorname{span}
\left\{
u_0,
M^{-1}Au_0,
\ldots,
(M^{-1}A)^{k-1}u_0
\right\}.
$$

Although $M^{-1}A$ is not necessarily symmetric under the ordinary Euclidean inner product, it is self-adjoint under the $M$-inner product:

$$
\langle x,y\rangle_M=x^TMy.
$$

Standard PCG uses the recurrences

$$
s_k=Ap_k,
$$

$$
\alpha_k
=
\frac{(r_k,u_k)}{(s_k,p_k)},
$$

$$
x_{k+1}=x_k+\alpha_kp_k,
$$

$$
r_{k+1}=r_k-\alpha_ks_k,
$$

$$
u_{k+1}=M^{-1}r_{k+1},
$$

$$
\beta_{k+1}
=
\frac{(r_{k+1},u_{k+1})}
{(r_k,u_k)},
$$

$$
p_{k+1}
=
u_{k+1}+\beta_{k+1}p_k.
$$

Here, $(x,y)=x^Ty$ denotes the Euclidean inner product.

### Where standard PCG synchronizes

Standard PCG requires two global synchronization points per iteration.

The first occurs when computing the dot products needed to calculate $\alpha_k$. The second occurs when calculating the dot product needed for $\beta_{k+1}$.

These operations are sequentially dependent:

1. Calculate $\alpha_k$.
2. Update the solution and residual.
3. Apply the preconditioner.
4. Calculate $\beta_{k+1}$.

Each global reduction requires communication between all participating processors. Even when the individual dot products are relatively inexpensive, global synchronization introduces latency that becomes significant at large processor counts.

### 4. Chronopoulos/Gear reformulation

The Chronopoulos/Gear variant reduces the number of global synchronization points from two to one.

Introduce the auxiliary vector

$$
w_i=Au_i.
$$

Recall the search-direction recurrence:

$$
p_i=u_i+\beta_i p_{i-1}.
$$

Multiplying by $A$ gives

$$
Ap_i=Au_i+\beta_iAp_{i-1}.
$$

Defining $s_i=Ap_i$ yields

$$
s_i=w_i+\beta_i s_{i-1}.
$$

Thus, we can calculate $s_i$ recursively rather than directly applying $A$ to $p_i$.

Next, introduce two scalars:

$$
c_i=(r_i,u_i),
$$

$$
d_i=(w_i,u_i).
$$

The CG coefficients can now be written as

$$
\beta_i=\frac{c_i}{c_{i-1}},
$$

and

$$
\alpha_i
=
\frac{c_i}
{d_i-\dfrac{\beta_ic_i}{\alpha_{i-1}}}.
$$

For the initial iteration:

$$
\beta_0=0,
\qquad
\alpha_0=\frac{c_0}{d_0}.
$$

The expression for $\alpha_i$ follows from the $A$-conjugacy of the search directions.

Crucially, both $c_i$ and $d_i$ can be computed together in a single global reduction.

**Main result:** Chronopoulos/Gear CG preserves mathematical equivalence to standard CG while reducing the number of global synchronization points from two to one.

However, the algorithm still needs to wait for the global reduction before continuing with dependent calculations.

### The pipelined reformulation

The authors extend Chronopoulos/Gear CG by reorganizing the recurrences to overlap the global reduction with the SpMV.

First, consider the unpreconditioned case:

$$
M=I,
\qquad
u_i=r_i.
$$

Recall the existing auxiliary vectors:

$$
w_i=Ar_i,
$$

$$
s_i=Ap_i.
$$

The difficulty is that directly computing

$$
w_{i+1}=Ar_{i+1}
$$

requires knowing $r_{i+1}$, which depends on $\alpha_i$ and therefore on the global reduction.

To eliminate this dependency, use

$$
r_{i+1}=r_i-\alpha_i s_i.
$$

Multiplying by $A$:

$$
Ar_{i+1}
=
Ar_i-\alpha_iAs_i.
$$

Introduce

$$
z_i=As_i.
$$

We obtain the recurrence

$$
w_{i+1}
=
w_i-\alpha_i z_i.
$$

This avoids directly computing $Ar_{i+1}$ after the global reduction.

Next, use

$$
s_i=w_i+\beta_i s_{i-1}.
$$

Multiplying by $A$ gives

$$
As_i=Aw_i+\beta_iAs_{i-1}.
$$

Introduce another auxiliary vector:

$$
q_i=Aw_i.
$$

Therefore,

$$
z_i=q_i+\beta_i z_{i-1}.
$$

The important identities are

$$
\boxed{
\begin{aligned}
q_i &= Aw_i,\\
z_i &= q_i+\beta_i z_{i-1},\\
s_i &= w_i+\beta_i s_{i-1},\\
w_{i+1} &= w_i-\alpha_i z_i.
\end{aligned}
}
$$

The key observation is that $q_i=Aw_i$ can be calculated without knowing the current $\alpha_i$ or $\beta_i$.

Consequently, the algorithm can initiate a non-blocking global reduction for

$$
c_i=(r_i,r_i),
$$

$$
d_i=(w_i,r_i),
$$

while simultaneously computing

$$
q_i=Aw_i.
$$

Once the reduction finishes, the algorithm calculates the coefficients and completes the remaining vector recurrences.

This is the essential pipelining mechanism: **global communication and useful computation are executed concurrently rather than sequentially.**

The method retains one SpMV and one global reduction per iteration, but their execution can overlap.

In exact arithmetic, the reformulation is mathematically equivalent to standard CG.

### Numerical considerations

The pipelined formulation requires additional auxiliary vectors and recurrence relations.

Although these recurrences are mathematically equivalent to the corresponding direct matrix-vector products, floating-point rounding errors are propagated differently.

For example, repeatedly updating

$$
w_{i+1}=w_i-\alpha_i z_i
$$

does not necessarily produce exactly the same floating-point result as directly calculating

$$
w_{i+1}=Ar_{i+1}.
$$

This introduces a potential trade-off between improved parallel performance and numerical stability.

The authors investigate this issue in later sections.

### Pipelined CR variant

The preconditioned operator $M^{-1}A$ is self-adjoint not only under the $M$-inner product used for PCG, but also under the $A$-inner product,

$$
\langle x,y\rangle_A=x^TAy.
$$

Using the $A$-inner product instead changes the quantity minimized by the Krylov iteration. Rather than minimizing the $A$-norm of the error as in CG, the resulting method minimizes

$$
\|e_k\|_{AM^{-1}A}.
$$

Since $r_k=Ae_k$, this can also be written as

$$
\|e_k\|_{AM^{-1}A}^2=r_k^TM^{-1}r_k.
$$

Without preconditioning, $M=I$, so

$$
\|e_k\|_{A^2}=\|r_k\|_2,
$$

meaning that CR minimizes the Euclidean residual norm over the Krylov subspace.

The scalar quantities used by pipelined CR become

$$
c_i=(w_i,u_i),
$$

and

$$
d_i=(m_i,w_i),
$$

where

$$
m_i=M^{-1}w_i.
$$

Because the algorithm no longer requires the original residual $r_i$ or $s_i=Ap_i$ in its main recurrence, those two vector recurrences can be dropped, reducing memory and floating-point work relative to pipelined CG.

The trade-off is that $d_i$ depends on $m_i=M^{-1}w_i$, so the preconditioner must be applied before the global reduction can begin. Therefore, pipe-CR can overlap its global reduction with the SpMV, but not with application of the preconditioner.

Its idealized critical-path cost is therefore

$$
T_{\mathrm{pipe\text{-}CR}}
\approx
\mathrm{PC}+\max(G,\mathrm{SpMV}),
$$

where $G$ is the global reduction latency.

### Comparison with Gropp's asynchronous CG

Gropp's asynchronous CG follows a less aggressive communication-hiding strategy. Rather than reducing the number of global synchronizations to one, it retains two reductions per iteration but overlaps each with useful computation.

One reduction,

$$
d_i=(p_i,s_i),
$$

is overlapped with application of the preconditioner,

$$
q_i=M^{-1}s_i.
$$

The second reduction,

$$
c_{i+1}=(r_{i+1},u_{i+1}),
$$

is overlapped with the matrix-vector product,

$$
w_{i+1}=Au_{i+1}.
$$

Thus, its idealized cost is

$$
T_{\mathrm{Gropp}}
\approx
\max(G,\mathrm{SpMV})
+
\max(G,\mathrm{PC}).
$$

Gropp-CG requires fewer additional recurrences, floating-point operations, and stored vectors than pipe-CG, which tends to give it numerical behavior closer to standard CG. However, because it retains two synchronization phases, it offers less potential communication overlap and therefore worse strong scalability when global reductions become the dominant bottleneck.

The main communication strategies can be summarized as:

- Standard CG: two blocking global reductions.
- Chronopoulos/Gear CG: combine dot products into one global reduction.
- Pipe-CG: one nonblocking reduction overlapped with the preconditioner and SpMV.
- Pipe-CR: one nonblocking reduction overlapped with the SpMV only.
- Gropp-CG: two reductions, with one overlapped with the preconditioner and the other with the SpMV.

## Results

Tests on a broad collection of symmetric positive-definite Matrix Market problems show that the pipelined algorithms generally preserve the convergence behavior of their corresponding standard methods. Pipe-CG required about $3\%$ more iterations than standard CG on average, while pipe-CR required about $4\%$ more iterations than standard CR. Standard CG, single-reduction CG, and Gropp-CG generally produced nearly indistinguishable convergence histories. Pipe-CG and pipe-CR typically converged similarly initially but reached a higher accuracy floor because the additional recurrences propagate floating-point errors differently.

The main finite-precision problem is the growing difference between the recursively updated residual and the true residual. The algorithm updates

$$
r_{i+1}=r_i-\alpha_iAp_i,
$$

while the true residual is

$$
\hat r_i=b-Ax_i.
$$

Because $x_i$ and $r_i$ are updated through separate floating-point recurrences, rounding errors accumulate differently and eventually

$$
r_i\neq b-Ax_i.
$$

The authors therefore experiment with residual replacement, periodically recomputing

$$
r_i\leftarrow b-Ax_i,
$$

together with the corresponding preconditioned and auxiliary quantities. Replacing these quantities every 50 iterations improves the maximum attainable accuracy by at least an order of magnitude in the examples studied. However, the value $50$ is explicitly arbitrary and is not optimal for every system.

Residual replacement cannot simply be performed every iteration because overly frequent replacement can interfere with the superlinear convergence often observed in CG. As the Krylov space grows, CG accumulates spectral information and increasingly approximates important eigen-directions through Ritz values; once difficult eigencomponents are effectively removed, the remaining problem can behave as though it has a much better-conditioned spectrum and convergence accelerates. Repeatedly perturbing the residual recurrence can disturb this accumulated Krylov structure, producing a trade-off between controlling roundoff error and preserving favorable convergence behavior.

In the strong-scaling ice-flow benchmark, pipe-CG achieved a maximum speedup of $2.14\times$ over standard CG and $1.43\times$ over single-reduction CG on 20 nodes. Pipe-CR achieved a maximum speedup of $2.09\times$ over standard CR. Importantly, at 20 nodes pipe-CG used exactly the same number of linear iterations as standard CG and CG1, indicating that the measured improvement came from better parallel execution rather than fewer iterations.

The pipelined methods are most attractive when global synchronization latency represents a significant part of runtime. They are less attractive when a very strong and expensive preconditioner dominates the cost of each iteration. Potentially favorable applications include systems requiring many repeated Krylov solves, latency-bound systems distributed over many processors, coarse-grid multigrid solves, domain decomposition, Schur complement methods, optimization, implicit time integration, and problems for which no near-optimal preconditioner is available.

## Questions and gaps

- **Why can frequent residual replacement destroy CG's superlinear convergence?** CG can accelerate as its growing Krylov subspace captures spectral information about $A$, effectively suppressing difficult eigencomponents and improving the spectrum of the remaining error. Residual replacement corrects floating-point drift but perturbs the recurrence and approximate orthogonality/conjugacy structure responsible for accumulating this information. Too-frequent replacement may therefore partially sacrifice the superlinear phase.

- **How should residual replacement actually be scheduled?** A fixed replacement period $T$, such as $T=50$, cannot generally be optimal because the rate of residual-gap growth depends on the matrix, preconditioner, conditioning, floating-point precision, convergence state, target tolerance, and hardware.

- **What is the correct objective for an "optimal" residual-replacement strategy?** Minimizing the residual gap alone may not minimize total computational cost. A better objective may be time-to-solution subject to a target true-residual tolerance,

$$
\min_{\pi} T_{\mathrm{solve}}(\pi)
\qquad
\text{s.t.}
\qquad
\frac{\|b-Ax_k\|}{\|b\|}
\leq \varepsilon_{\mathrm{target}},
$$

where $\pi$ is an adaptive policy deciding at each iteration whether to continue the recurrence or perform a replacement.

- How do the additional pipelined recurrences amplify local rounding errors, and how does this depend on the spectrum of $A$?

- How much spectral information about $A$ can be extracted cheaply from the CG/Lanczos iteration and used to improve stability or residual-replacement decisions?

- When does the extra arithmetic and memory traffic of pipelining outweigh the communication latency being hidden?

- How do these trade-offs change with the preconditioner and with the ratio between global-reduction latency, SpMV time, and preconditioner time?

## Open problems / extensions of the work

The paper leaves a general residual-replacement strategy for pipelined methods as future work. It also suggests introducing shifts into the pipelined matrix-vector recurrences to improve numerical stability, potentially using spectral information produced by CG to choose good shifts.

Another extension is deeper pipelining. If one SpMV and preconditioner application are not long enough to hide the global reduction, the pipeline could span several Krylov iterations. This increases the amount of communication that can be hidden but also increases recurrence length, memory requirements, arithmetic work, and potential floating-point instability.

The authors also identify the relationship with $s$-step and communication-avoiding Krylov methods. Pipelining hides communication while $s$-step methods reduce its frequency; combining these ideas with effective preconditioners remains difficult.

Another possible optimization is to fuse the preconditioner and SpMV when their structure permits it, reducing the latency and memory movement of the sequential computational part of pipe-CG.

A full finite-precision rounding-error analysis of the pipelined algorithms is left as future work in this paper. However, this has already been published by Cools et al., 2018.

### Follow-up research direction: adaptive residual replacement

Later work has already addressed part of the residual-replacement problem, so the research question should **not** be framed as simply finding a universally optimal fixed $T$. Later analysis of pipelined CG derives estimates of the gap between the recursive and true residual and uses these estimates to trigger automated residual replacement. Modern PETSc includes such a method, `KSPPIPECGRR`, where replacement iterations are dynamically determined.

A more interesting remaining direction is a **performance- and convergence-aware residual-replacement policy** that jointly considers numerical error and the state of Krylov convergence. The decision could depend on quantities such as

$$
\text{estimated residual gap},
\qquad
\text{Ritz-value convergence},
\qquad
\text{loss of orthogonality},
\qquad
\text{observed convergence rate},
$$

as well as

$$
\text{SpMV cost},
\qquad
\text{preconditioner cost},
\qquad
\text{communication latency},
\qquad
\text{target tolerance}.
$$

A particularly interesting theoretical question is whether one can characterize when a residual replacement can be performed without materially disrupting the superlinear convergence phase of CG. This would connect finite-precision numerical linear algebra, spectral information from the Lanczos/CG process, and HPC performance modeling.

**Potential question for office hours:** Can residual replacement in pipelined CG be made adaptive not only to the estimated residual gap, but also to the current spectral/Krylov state, so that it controls floating-point error while preserving superlinear convergence and minimizing total time-to-solution?

## Relevance to my work

The paper assumes local SpMV and global dot products. In the row-distributed, replicated-vector layout used here for $A = X^\top X + \lambda I$, the structure is inverted: the dot products $c_i$ and $d_i$ are computed on replicated $d$-vectors and require no communication at all, while the only global reduction is the `Allreduce` inside the operator that assembles $X^\top(Xp)$ from per-rank partial sums.

Pipelined CG hides the dot-product reduction behind an independent operator application, $q_i = Aw_i$. Here there is no dot-product reduction to hide, and the reduction that does exist sits inside the operator, with every subsequent quantity depending on its result through $z_i = q_i + \beta_i z_{i-1}$ and $w_{i+1} = w_i - \alpha_i z_i$. The only work available to overlap the reduction is the $O(d)$ vector updates for $s_i$, $p_i$, $x_{i+1}$ and $r_{i+1}$ — roughly $800$ flops at $d = 200$, against a reduction latency of tens of microseconds.

The authors make the same point about Chronopoulos/Gear in Section 2.2: "even for small parallel machines the runtime of a single vector update will not be enough to fully cover the latency of the global communication." That is precisely the situation here, and it is why the paper introduces $q_i = Aw_i$ — an option not available when the operator itself contains the reduction.