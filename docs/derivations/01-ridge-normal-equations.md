# Ridge Regression to the Normal Equations

## Goal

Show that ridge regression leads to the linear system

$$
(X^\top X + \lambda I)w = X^\top y,
$$

and prove that the coefficient matrix is symmetric positive definite when $\lambda > 0$. This is the property that allows the Conjugate Gradient method to be used.

---

## Derivation

Ridge regression solves the optimization problem

$$
\min_w \; \|Xw-y\|_2^2 + \lambda \|w\|_2^2,
$$

where

- $X \in \mathbb{R}^{n \times d}$ is the data matrix,
- $y \in \mathbb{R}^n$ is the target vector,
- $w \in \mathbb{R}^d$ is the parameter vector,
- $\lambda > 0$ is the regularization parameter.

Define

$$
f(w) = \|Xw-y\|_2^2 + \lambda\|w\|_2^2.
$$

Expand the squared norms:

$$
f(w) = (Xw-y)^\top(Xw-y) + \lambda w^\top w.
$$

Expanding the first term gives

$$
f(w) = w^\top X^\top Xw - y^\top Xw - w^\top X^\top y + y^\top y + \lambda w^\top w.
$$

The two middle terms are scalars, and a scalar equals its own transpose, so

$$
w^\top X^\top y = (w^\top X^\top y)^\top = y^\top Xw.
$$

They can therefore be combined:

$$
f(w) = w^\top X^\top Xw - 2y^\top Xw + y^\top y + \lambda w^\top w.
$$

Taking the gradient with respect to $w$, and using the fact that $X^\top X$ is symmetric so that $\nabla(w^\top X^\top Xw) = 2X^\top Xw$,

$$
\nabla f(w) = 2X^\top Xw - 2X^\top y + 2\lambda w.
$$

At the minimizer, the gradient is zero:

$$
2X^\top Xw - 2X^\top y + 2\lambda w = 0.
$$

Divide by $2$:

$$
X^\top Xw + \lambda w = X^\top y.
$$

Therefore,

$$
(X^\top X+\lambda I)w = X^\top y.
$$

Define

$$
A = X^\top X + \lambda I, \qquad b = X^\top y.
$$

Then ridge regression is equivalent to solving

$$
Aw = b.
$$

---

## Why $A$ is Symmetric Positive Definite

First, $A$ is symmetric because

$$
A^\top = (X^\top X+\lambda I)^\top = X^\top X + \lambda I = A.
$$

To show positive definiteness, take any nonzero vector $v \in \mathbb{R}^d$. Then

$$
v^\top A v = v^\top(X^\top X+\lambda I)v.
$$

Expanding,

$$
v^\top A v = v^\top X^\top Xv + \lambda v^\top v.
$$

Using

$$
v^\top X^\top Xv = (Xv)^\top(Xv) = \|Xv\|_2^2,
$$

we obtain

$$
v^\top A v = \|Xv\|_2^2 + \lambda\|v\|_2^2.
$$

Both terms are nonnegative. Since $\lambda>0$ and $v\neq0$,

$$
\lambda\|v\|_2^2 > 0.
$$

Therefore,

$$
v^\top A v > 0
$$

for every nonzero $v$, so $A$ is symmetric positive definite.

An important consequence is that this remains true even if $X$ is rank-deficient. The regularization term $\lambda I$ shifts the eigenvalues away from zero and makes the system uniquely solvable.

---

## Conclusion

Ridge regression reduces to the SPD linear system

$$
(X^\top X+\lambda I)w = X^\top y.
$$

Because the coefficient matrix is symmetric positive definite for $\lambda>0$, Conjugate Gradient is a valid solver for the problem.

---

## Connection to the Implementation

The matrix

$$
A = X^\top X + \lambda I
$$

is never formed explicitly in the project.

Instead, the solver only needs the action of $A$ on a vector $p$:

$$
Ap = X^\top(Xp) + \lambda p.
$$

This is the basis of the matrix-free operator implemented in

`src/parallel_cg_ridge/operators.py`.