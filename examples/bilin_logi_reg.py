import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bilinear Logistic Regression
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")

    import cvxpy as cp
    import marimo as mo
    import numpy as np
    from sklearn.datasets import make_classification

    from dbcp import BiconvexProblem

    np.random.seed(10015)
    return BiconvexProblem, cp, make_classification, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Suppose we are given a dataset $(X_i, y_i)$, $i = 1, \ldots, m$, where each sample consists of a feature
    matrix $X_i \in \mathbf{R}^{n \times k}$ and a binary label $y_i \in \{0, 1\}$.
    Our goal is to construct a bilinear classifier

    \[
        \hat{y} = \left\{\begin{array}{ll}
            1 & \mathop{\bf tr}(U^T X V) > 0\\
            0 & \text{otherwise},
        \end{array}\right.
    \]

    where $U \in \mathbf{R}^{n \times r}$ and $V \in \mathbf{R}^{k \times r}$ are the bilinear logistic
    regression coefficients with a predefined (maximum) rank $r$, and $\mathop{\bf tr}(M)$ denotes the trace
    of some square matrix $M$.

    To fit a bilinear logistic regression model to the dataset, we would like to solve the following bilinear
    maximum likelihood estimation problem:

    \[
        \begin{array}{ll}
            \text{maximize} & \sum_{i = 1}^{m} y_i \mathop{\bf tr}(U^T X_i V) -
    \log(1 + \exp(\mathop{\bf tr}(U^T X_i V)))
        \end{array}
    \]

    with variables $U$ and $V$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate problem data
    """)
    return


@app.cell
def _(make_classification):
    m = 300
    n = 20
    k = 10
    r = 5
    ninfo_frac = 0.9
    Xs, ys = make_classification(
        n_samples=m, n_features=n * k, n_informative=int(n * k * ninfo_frac), n_redundant=int(n * k * (1 - ninfo_frac))
    )
    Xs = Xs.reshape(m, n, k)
    return Xs, k, n, r, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the biconvex problem
    """)
    return


@app.cell
def _(BiconvexProblem, Xs, cp, k, n, r, ys):
    U = cp.Variable((n, r))
    V = cp.Variable((k, r))

    obj = 0
    for _X, _y in zip(Xs, ys):
        obj += cp.sum(cp.multiply(_y, cp.trace(U.T @ _X @ V)) - cp.logistic(cp.trace(U.T @ _X @ V)))
    prob = BiconvexProblem(cp.Maximize(obj), [[U], [V]])
    prob.solve(cp.CLARABEL, lbd=1, abs_tol=1e-2)
    return


if __name__ == "__main__":
    app.run()
