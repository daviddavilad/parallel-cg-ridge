# Matrix-Free Ridge Operator

## Goal

Show that the matrix-free operator

$$
A p = X^\top (X p) + \lambda p
$$

is mathematically identical to multiplying by the matrix

$$
A = X^\top X + \lambda I,
$$

and explain why the matrix-free formulation is preferred for large-scale scientific computing.

---

## Matrix-Free Operator Identity

The ridge regression system is

$$
(X^\top X + \lambda I) w = X^\top y.
$$

Instead of explicitly forming the matrix

$$
A = X^\top X + \lambda I,
$$

the solver only computes the action of $A$ on a vector $p$.

Starting from the matrix expression,

$$
Ap = (X^\top X + \lambda I)p.
$$

Distributing the matrix-vector product,

$$
Ap = (X^\top X)p + (\lambda I)p.
$$

By associativity of matrix multiplication,

$$
(X^\top X)p = X^\top(Xp).
$$

Since

$$
Ip = p,
$$

the expression becomes

$$
Ap = X^\top(Xp) + \lambda p.
$$

Therefore,

$$
(X^\top X + \lambda I)p = X^\top(Xp)+\lambda p.
$$

This is an exact algebraic identity, not an approximation. The matrix-free formulation computes exactly the same vector as the explicit matrix multiplication, only without ever constructing $X^\top X$.

---

## Computational Cost

Suppose $X \in \mathbb{R}^{n \times d}$.

### Explicit Matrix

Forming

$$
X^\top X
$$

multiplies a $d \times n$ matrix by an $n \times d$ matrix.

The computational cost is

$$
O(nd^2).
$$

The resulting matrix requires

$$
O(d^2)
$$

storage.

Once formed, each matrix-vector product

$$
Ap
$$

costs

$$
O(d^2).
$$

If Conjugate Gradient requires $k$ iterations, the total computational cost is

$$
O(nd^2 + kd^2).
$$

---

### Matrix-Free Formulation

Each application of

$$
Ap = X^\top(Xp)+\lambda p
$$

requires two matrix-vector products.

Computing

$$
Xp
$$

costs

$$
O(nd),
$$

and computing

$$
X^\top(Xp)
$$

also costs

$$
O(nd).
$$

Therefore, each operator application costs

$$
O(nd),
$$

with no additional storage beyond the original data matrix $X$.

Over $k$ CG iterations, the total computational cost is

$$
O(knd).
$$

---

## Why Matrix-Free?

The explicit formulation performs an expensive preprocessing step to construct

$$
X^\top X,
$$

after which each iteration is relatively cheap.

The matrix-free formulation performs no preprocessing, instead computing the operator action directly every iteration.

Although the explicit approach may become competitive if enough iterations are performed to amortize the preprocessing cost, it requires storing a dense $d \times d$ matrix. For large-scale problems, this memory requirement quickly becomes the limiting factor.

In the distributed-memory setting, explicitly forming $X^\top X$ would also require additional global communication across MPI processes before the iterative solve even begins.

For these reasons, the matrix-free formulation is the standard approach for large-scale iterative methods such as Conjugate Gradient.

---

## Connection to the Implementation

This derivation motivates the implementation in

`src/parallel_cg_ridge/operators.py`.

Rather than constructing

$$
A = X^\top X + \lambda I,
$$

the implementation returns a callable operator that computes

$$
Ap = X^\top(Xp)+\lambda p
$$

whenever the Conjugate Gradient solver requests a matrix-vector product.