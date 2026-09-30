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

To be studied.


### Comparison with Gropp's asynchronous CG

To be studied.


## Results

To be completed after studying the numerical experiments and parallel performance benchmarks.


## Questions and gaps

1. How do the additional recurrences affect the propagation of floating-point errors?
2. Under what circumstances does the additional computation outweigh the time saved by overlapping global communication?
3. How does the choice of preconditioner influence the effectiveness of pipelining?
4. Can the pipelined approach preserve the convergence properties of standard CG in finite-precision arithmetic?


## Open problems / extensions of the work

To revisit after reading the remaining sections.

Potential topics to investigate:

- Numerical stability of pipelined recurrences.
- Communication hiding versus communication avoidance.
- Generalization to other Krylov subspace methods.


## Relevance to my work