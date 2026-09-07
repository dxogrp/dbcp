import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sparse Dictionary Learning
    """)
    return


@app.cell
def _():
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from dbcp import BiconvexProblem

    _example_directory = Path(__file__).resolve().parent
    plt.style.use(_example_directory / "zhlatex.mplstyle")
    figure_directory = _example_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(10015)
    return BiconvexProblem, cp, figure_directory, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We consider the sparse dictionary learning problem, which aims to find a dictionary matrix
    $D \in \mathbf{R}^{m \times k}$ and a sparse code matrix $X \in \mathbf{R}^{k \times n}$, such that the data
    matrix $Y \in \mathbf{R}^{m \times n}$ can be well approximated by their product $DX$, while the matrix $X$
    is sparse and the matrix $D$ has bounded Frobenius norm.
    The dictionary learning problem can be formulated as the following biconvex optimization problem:

    \[
        \begin{array}{ll}
            \text{minimize} & {\|DX - Y\|}_F^2 + \alpha {\|X\|}_1\\
            \text{subject to} & {\|D\|}_F \leq \beta
        \end{array}
    \]

    with variables $D$ and $X$, where $\alpha > 0$ is the sparsity regularization parameter, and $\beta > 0$
    is the bound on the Frobenius norm of the dictionary matrix.
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
    m = 10
    n = 20
    k = 20
    beta = 1

    Y = np.random.randn(m, n)
    return Y, beta, k, m, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the problem
    """)
    return


@app.cell
def _(BiconvexProblem, Y, beta, cp, k, m, n, np):
    D = cp.Variable((m, k))
    X = cp.Variable((k, n))
    alpha = cp.Parameter(nonneg=True)
    obj = cp.Minimize(cp.sum_squares(D @ X - Y) + alpha * cp.norm1(X))
    prob = BiconvexProblem(obj, [D], [X], [cp.norm(D, "fro") <= beta])

    errs = []
    cards = []
    for _a in np.logspace(-5, 0, 50):
        alpha.value = _a
        D.value = None
        X.value = None
        prob.solve(cp.CLARABEL, abs_tol=1e-1)
        errs.append(cp.norm(D @ X - Y, "fro").value / cp.norm(Y, "fro").value)
        cards.append(cp.sum(cp.abs(X).value >= 1e-3).value)
    return cards, errs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(cards, errs, figure_directory, plt):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    axs.plot(cards, errs, marker=".", color="k")
    axs.set_xlabel(r"$\mathbf{card}\,X$")
    axs.set_ylabel("$||DX-Y||_F/||Y||_F$")

    fig.tight_layout()
    fig.savefig(figure_directory / "dict_learning.pdf", bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
