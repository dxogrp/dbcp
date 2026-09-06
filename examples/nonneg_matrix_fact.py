import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nonnegative Matrix Factorization
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from dbcp import BiconvexProblem

    np.random.seed(10015)
    return BiconvexProblem, cp, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Suppose we are given a matrix $A \in \mathbf{R}^{m \times n}$, and are interested in solving the problem:
    $$\begin{array}{ll}\text{minimize} & {\|XY - A\|}_F^2 \\ \text{subject to} & X_{ij} \geq 0,\quad i = 1, \ldots, m,\quad j = 1, \ldots, k \\ & Y_{ij} \geq 0,\quad i = 1, \ldots, k,\quad j = 1, \ldots, n,\end{array}$$
    where $X \in \mathbf{R}^{m \times k}$ and $Y \in \mathbf{R}^{k \times n}$ are the problem variables.

    This problem is biconvex in the variables $X$ and $Y$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate problem data
    """)
    return


@app.cell
def _(np):
    m = 5
    n = 10
    k = 5
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    return A, k, m, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the biconvex problem
    """)
    return


@app.cell
def _(A, BiconvexProblem, cp, k, m, n):
    X = cp.Variable((m, k), nonneg=True)
    Y = cp.Variable((k, n), nonneg=True)

    obj = cp.Minimize(cp.sum_squares(X @ Y - A))
    prob = BiconvexProblem(obj, [[X], [Y]])
    prob.solve()
    return


if __name__ == "__main__":
    app.run()
