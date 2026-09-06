# DBCP: Disciplined Biconvex Programming

DBCP is an extension of [CVXPY](https://github.com/cvxpy/cvxpy)
for (approximately) solving *biconvex optimization problems*
in the form

$$
\begin{array}{ll}
    \text{minimize} & f_0(x, y)\\
    \text{subject to} & f_i(x, y) \leq 0,\quad i = 1, \ldots, m\\
    & h_i(x, y) = 0,\quad i = 1, \ldots, p,
\end{array}
$$

where $x \in X$, $y \in Y$ are the optimization variables.
The functions $f_0, \ldots, f_m$ are biconvex, meaning that for fixed $y$,
the functions $f_i(\cdot, y)$ are convex,
and for fixed $x$, the functions $f_i(x, \cdot)$ are convex.
The functions $h_1, \ldots, h_p$ are biaffine in a similar sense.
In the most general case, biconvex optimization problems are very hard
to solve, but heuristic methods such as *alternating convex search* (ACS)
can often find good solutions in practice.

More theoretical and technical aspects about
biconvex optimization problems can be found in our [accompanying paper](https://haozhu10015.github.io/papers/dbcp.html).

## Installation

DBCP has the following dependencies:

- Python >= 3.12
- [NumPy](https://numpy.org/doc/stable/index.html) >= 2.3.3
- [CVXPY](https://www.cvxpy.org/) >= 1.9

### Using pip

You can install the package using pip:

```shell
pip install dbcp
```

### Development setup

DBCP manages its development environment with [uv](https://docs.astral.sh/uv/).
Before setting up the repository, install uv.

1. Clone the repository:

    ```shell
    git clone https://github.com/dxogrp/dbcp.git
    cd dbcp
    ```

2. Create the virtual environment and install the locked development
   dependencies:

    ```shell
    make sync
    ```

Run `make test` and `make lint` before contributing changes.

## Usage

Here we provide a basic overview of how to use DBCP;
for more details, please refer to our paper
and the [examples](./examples).

### DBCP syntax rule for multiplications

DBCP is based on CVXPY and inherits its syntax rules,
with the following extension for variable multiplications:

1. A valid DBCP convex product expression between
   variables should be one of the form:

   - *affine* \* *affine*
   - *affine-nonneg* \* *convex*
   - *affine-nonpos* \* *concave*
   - *convex-nonneg* \* *convex-nonneg*
   - *concave-nonpos* \* *concave-nonpos*

2. There exists no loop in the variable interaction graph of
  the overall expression, where the edge between two variables
  indicates that they appear in different sides in a
  product expression as described in the above rule.

### Specifying biconvex problems

DBCP provides the `BiconvexProblem` and `BiconvexRelaxProblem`
classes to specify biconvex problems.
Roughly speaking, the difference between these two classes is that
`BiconvexProblem` implements a solver for directly solving
the original biconvex problem, while
`BiconvexRelaxProblem` is used for solving a relaxed version
of the problem with additional slack variables added to the constraints,
so the latter allows solving with infeasible starting points.

As an example, to create a `BiconvexProblem` instance,
one can use the following syntax:

```python
prob = BiconvexProblem(obj, [x_var, y_var], constraints)
```

The argument `obj` is a DBCP-compliant biconvex expression
representing the objective function, `x_var` and `y_var`
are lists of the biconvex optimization variables,
and `constraints` is a list of DBCP-compliant biconvex constraints.
The arguments `x_var` and `y_var` define the variable partition
for the biconvex problem, such that each group will be fixed
when optimizing over the other group during the ACS procedure.

### Verification of biconvexity

After creating a `BiconvexProblem` or `BiconvexRelaxProblem` instance,
one can call its `is_dbcp` method to verify whether the problem
is DBCP-compliant:

```python
prob.is_dbcp()
```

which returns `True` if the problem is DBCP-compliant, and `False` otherwise.

### Solving a biconvex problem

After creating a `BiconvexProblem` instance,
one can call its `solve` method to solve the problem:

```python
prob.solve()
```

The most important optional arguments of the
`BiconvexProblem.solve` method are as follows:

- `lbd`: The regularization parameter of the proximal term.
- `max_iters`: The maximum number of ACS iterations.
- `gap_tolerance`: The tolerance for the gap between the subproblems
  when stopping the ACS procedure.

The `BiconvexRelaxProblem.solve` method has an additional
optional argument `nu` to specify the penalty parameter
for the total slack, i.e.,
violation of the biconvex constraints.

### Problem status

After solving a biconvex problem using the `solve` method,
one can check the problem status using the `status` attribute.

The possible status values for `BiconvexProblem` are as follows:

- `converge`: The ACS procedure converged successfully, i.e.,
  the final gap between the subproblems is within the specified tolerance.
- `converge_inaccurate`: The maximum number of iterations was reached,
  but the final gap between the subproblems is still
  larger than the specified tolerance.

In the second case, one may want to call the `solve` method again.
This will continue the ACS procedure from the last iteration, until
either convergence is achieved or the maximum number of iterations
is reached.

The possible status values for `BiconvexRelaxProblem` are as follows:

- `converge`: The ACS procedure converged successfully
  with a feasible solution (i.e., the total slack is zero)
  to the original problem.
- `converge_infeasible`: The ACS procedure converged successfully,
  but the final solution is still infeasible with nonzero total slack.
- `converge_inaccurate`: The maximum number of iterations was reached
  with a feasible final point,
  but the final gap between the subproblems is still
  larger than the specified tolerance.
- `converge_inaccurate_infeasible`: The maximum number of iterations was reached,
  but the final gap between the subproblems is still
  larger than the specified tolerance,
  and the final solution is still infeasible with nonzero total slack.

When 'infeasible' appears in the status,
one may want to increase the penalty parameter `nu`
and call the `solve` method again.

## Examples

### Basic example

Suppose we are given a matrix $A \in \mathbf{R}^{m \times n}$.
Consider the following nonnegative matrix factorization problem:

$$
\begin{array}{ll}
    \text{minimize} & {\|XY + Z - A\|}_F\\
    \text{subject to} & X_{ij} \geq 0,\quad i = 1, \ldots, m,
        \quad j = 1, \ldots, k\\
    & Y_{ij} \geq 0,\quad i = 1, \ldots, k,\quad j = 1, \ldots, n\\
    & {\|Z\|}_F \leq 1,
\end{array}
$$

with variables $X \in \mathbf{R}^{m \times k}$,
$Y \in \mathbf{R}^{k \times n}$,
and $Z \in \mathbf{R}^{m \times n}$.

To specify and solve this problem using DBCP,
one may use the following code:

```python
import cvxpy as cp
import dbcp

X = cp.Variable((m, k), nonneg=True)
Y = cp.Variable((k, n), nonneg=True)
Z = cp.Variable((m, n))

obj = cp.Minimize(cp.norm(X @ Y + Z - A, "fro"))
constraints = [cp.norm(Z, "fro") <= 1]
prob = dbcp.BiconvexProblem(obj, [[X], [Y]], constraints)

prob.solve()
```

### Other examples

We provide several other examples in the [examples](./examples) directory.
Install their locked dependencies and check every notebook with

```shell
make sync-examples
make check-examples
```

To open the notebooks in your browser, run

```shell
make marimo
```

in the repository folder to start the [Marimo](https://marimo.io/) environment.

## Citation

If you find DBCP useful in your research, please consider citing our paper:

```bibtex
@article{zhu2025dbcp,
  title={Disciplined Biconvex Programming},
  author={Zhu, H. and Boedecker, J.},
  journal={arXiv Preprint arXiv:2511.01813},
  year={2025},
}
```
