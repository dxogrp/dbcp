# Quick start

This example factors a nonnegative matrix $A\in\mathbf{R}^{m\times n}$ as
$XY$, where $X\in\mathbf{R}^{m\times k}$ and
$Y\in\mathbf{R}^{k\times n}$:

$$
\begin{array}{ll}
\text{minimize} & \|XY-A\|_F^2 \\
\text{subject to} & X_{ij} \geq 0,\quad i=1,\ldots,m,\quad j=1,\ldots,k\\
& Y_{ij} \geq 0,\quad i=1,\ldots,k,\quad j=1,\ldots,n.
\end{array}
$$

The objective is convex in $X$ for fixed $Y$ and convex in $Y$ for fixed
$X$.

```python
import cvxpy as cp
import numpy as np

import dbcp

rng = np.random.default_rng(10015)
m, n, k = 5, 10, 3
A = rng.random((m, k)) @ rng.random((k, n))

X = cp.Variable((m, k), name="X")
Y = cp.Variable((k, n), name="Y")
X.value = rng.random(X.shape)
Y.value = rng.random(Y.shape)

problem = dbcp.BiconvexProblem(
    cp.Minimize(cp.sum_squares(X @ Y - A)),
    [X],
    [Y],
    [X >= 0, Y >= 0],
)

assert problem.is_dbcp()
value = problem.solve()

print(problem.status)
print("objective =", value)
print("reconstruction error =", np.linalg.norm(X.value @ Y.value - A, "fro") ** 2)
```

The `[X]` and `[Y]` arguments supply the `x_var` and `y_var` variable groups,
while the last argument encodes the nonnegativity constraints. During
the x-subproblem, DBCP optimizes `X` while holding `Y` fixed; during the
y-subproblem it does the reverse. The original CVXPY variables receive the
final numerical values.

Because unset variables are initialized randomly, different starting points
can produce different factorizations. Assign `X.value` and `Y.value` before
`solve()` when a specific warm start is desired.
