# Cost Model for Distributed Matrix-Free CG

## Goal

Develop a simple performance model for the distributed ridge operator and use it to predict how runtime should change as the number of MPI ranks increases.

The model separates each iteration into two components:

1. local matrix-vector computation;
2. global communication through `MPI_Allreduce`.

The purpose of this model is not to predict exact wall-clock times. Instead, it gives a falsifiable prediction for the strong-scaling behavior that will later be measured experimentally.

---

## 1. Per-Iteration Computation

Let $X \in \mathbb{R}^{n \times d}$. With $P$ MPI ranks and a row-wise partition, rank $r$ owns $X_r \in \mathbb{R}^{n_r \times d}$, where approximately $n_r \approx n/P$.

Each application of the distributed ridge operator first computes

$$
u_r = X_r p.
$$

A dense matrix-vector product with an $n_r \times d$ matrix requires approximately $2n_r d$ floating-point operations, counting one multiplication and one addition per matrix entry.

The rank then computes

$$
q_r = X_r^\top u_r.
$$

This requires approximately another $2n_r d$ floating-point operations.

Therefore, the total local computational work per rank and per operator application is approximately $4n_r d$. Using $n_r \approx n/P$, the computational work becomes $4nd/P$.

Let $\gamma$ denote the effective time per floating-point operation. Then the computational component of the runtime can be modeled as

$$
T_{\mathrm{comp}}(P) = \gamma \frac{4nd}{P}.
$$

The important feature is the dependence on $P$:

$$
T_{\mathrm{comp}}(P) \propto \frac{1}{P}.
$$

Under ideal strong scaling, the local computational cost therefore decreases linearly as more MPI ranks are added.

---

## 2. Per-Iteration Communication

After each rank computes its local contribution

$$
q_r = X_r^\top(X_rp),
$$

the local vectors must be summed across all MPI ranks.

The distributed operator therefore performs one global reduction,

$$
q = \sum_{r=0}^{P-1} q_r.
$$

The vector being reduced has $d$ entries.

A standard latency-bandwidth model for a tree-based collective is

$$
T_{\mathrm{comm}}(P) \approx \alpha \log_2 P + 8\beta d \log_2 P,
$$

where $\alpha$ is the effective message latency and $\beta$ is the effective transfer time per byte.

The factor $8d$ appears because each entry is a 64-bit floating-point number and therefore occupies 8 bytes.

Equivalently,

$$
T_{\mathrm{comm}}(P) \approx (\alpha + 8\beta d)\log_2 P.
$$

For the values of $d$ considered in this project, the reduction messages are relatively small. If $\alpha \gg 8\beta d$, then communication is approximately latency dominated.

Under this assumption,

$$
T_{\mathrm{comm}}(P) \approx \alpha \log_2 P.
$$

Unlike the computational term, this cost does not decrease as ranks are added. Instead, the collective communication cost grows approximately logarithmically with $P$.

---

## 3. Compute-Communication Crossover

At small values of $P$, local matrix-vector computation should dominate the runtime. As $P$ increases, however, $T_{\mathrm{comp}}(P)$ decreases while $T_{\mathrm{comm}}(P)$ increases.

Define $P^*$ as the approximate compute-communication crossover point satisfying

$$
T_{\mathrm{comp}}(P^*) = T_{\mathrm{comm}}(P^*).
$$

Using the latency-dominated approximation gives

$$
\gamma \frac{4nd}{P^*} = \alpha \log_2 P^*.
$$

Rearranging,

$$
P^*\log_2 P^* = \frac{4\gamma nd}{\alpha}.
$$

Because $P^*$ appears both outside and inside the logarithm, this equation is transcendental. In practice, the crossover can be found numerically or estimated directly from measured timing data.

The equation nevertheless reveals the expected scaling behavior. If $d$, $\gamma$, and $\alpha$ remain fixed, then increasing $n$ increases the right-hand side proportionally. Therefore, the crossover rank should move to larger values as the problem size grows.

Ignoring the slowly varying logarithmic factor, $P^* \sim n$.

This gives a falsifiable prediction for the scaling experiments:

> Increasing the number of observations should move the compute-communication crossover toward larger MPI rank counts.

In particular, doubling $n$ should move the crossover by roughly a factor of two, with a correction caused by the logarithmic dependence on $P^*$.

---

## 4. Total Runtime and the U-Shaped Curve

Suppose Conjugate Gradient requires $k$ iterations to reach the chosen tolerance. Ignoring setup costs, the total runtime can be approximated by

$$
T(P) = k\left(T_{\mathrm{comp}}(P) + T_{\mathrm{comm}}(P)\right).
$$

Under the latency-dominated communication model,

$$
T(P) \approx k\left(\gamma\frac{4nd}{P} + \alpha\log_2 P\right).
$$

The two terms behave in opposite directions. The computational term satisfies $T_{\mathrm{comp}}(P) \propto 1/P$, while the communication term satisfies $T_{\mathrm{comm}}(P) \propto \log P$.

Consequently, adding MPI ranks initially reduces runtime because the reduction in local computation dominates the additional communication cost. At sufficiently large $P$, however, the local computation becomes small while collective communication continues to grow.

The model therefore predicts a U-shaped strong-scaling curve:

- runtime initially decreases rapidly;
- the improvement gradually becomes smaller;
- runtime reaches a minimum;
- additional ranks can eventually increase runtime.

### Ordering of the Crossover and the Minimum

The compute-communication crossover $P^*$ is not the same as the minimum-runtime rank, and the model determines which comes first.

Consider the simplified form

$$
T(P) = \frac{C}{P} + D\ln P,
$$

where $C = 4\gamma nd$ collects the computational constants and $D$ the communication constants. The minimum-runtime point satisfies

$$
\frac{dT}{dP} = -\frac{C}{P^2} + \frac{D}{P} = 0,
$$

which gives

$$
P_{\min} = \frac{C}{D}.
$$

Evaluating the two components at this point, the computational term equals $C/P_{\min} = D$ and the communication term equals $D\ln(C/D)$. Communication therefore already exceeds computation at $P_{\min}$ whenever

$$
\ln\frac{C}{D} > 1,
$$

that is, whenever $C/D > e$. Since $C \propto nd$ is large for any problem size of interest, this condition holds in practice. It follows that

$$
P^* < P_{\min}.
$$

Communication overtakes computation strictly before runtime stops improving. This is worth stating explicitly because it is not obvious: even once the reduction is the larger of the two costs, the computational term is still shrinking quickly enough, as $C/P^2$ in the derivative, to outweigh the logarithmic growth of communication. Wall-clock time continues to improve for a range of rank counts after parallel efficiency has already begun to deteriorate.

### Three Operating Points

The model therefore identifies three distinct rank counts, in increasing order:

1. $P^*$, where communication overtakes local computation;
2. the **knee**, where additional ranks begin producing only small reductions in runtime;
3. $P_{\min}$, where wall-clock time is smallest.

The rank count giving the absolute minimum runtime is not necessarily the most desirable operating point. A practical system may instead choose a smaller rank count near the knee.

The minimum-runtime point answers

> Which value of $P$ gives the smallest wall-clock time?

The knee answers a different engineering question:

> At what point does adding more parallel resources stop providing enough additional speedup to justify their cost?

This distinction will be evaluated experimentally using both runtime and parallel-efficiency measurements.

---

## Connection to the Implementation

The timing instrumentation in `src/parallel_cg_ridge/cg_mpi.py` will separately measure the computational and communication components of the distributed operator.

In particular, timings around

$$
u_r = X_rp
$$

and

$$
q_r = X_r^\top u_r
$$

will estimate the local computational cost, while the timing of `comm.Allreduce(...)` will measure the global communication cost.

The experiments can then compare the observed $T_{\mathrm{comp}}(P)$ and $T_{\mathrm{comm}}(P)$ against the predicted behaviors $T_{\mathrm{comp}}(P) \propto 1/P$ and $T_{\mathrm{comm}}(P) \propto \log P$.

The measured crossover, minimum-runtime point, and practical knee can then be compared across multiple problem sizes.