import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Blind Deconvolution
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

    from dbcp import BiconvexProblem, convolve

    _example_directory = Path(__file__).resolve().parent
    plt.style.use(_example_directory / "zhlatex.mplstyle")
    figure_directory = _example_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(10015)
    return BiconvexProblem, convolve, cp, figure_directory, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Blind deconvolution is a technique used to recover some sharp signal or image from a blurred observation
    when the blur itself is unknown.
    It jointly estimates both the original signal and the blur kernel, with some prior knowledge about their structures.

    Suppose we are given a data vector $d \in \mathbf{R}^{m + n - 1}$, which is the convolution of an unknown
    sparse signal $x \in \mathbf{R}^n$ and an unknown smooth vector $y \in \mathbf{R}^m$ with bounded
    $\ell_\infty$-norm (i.e., bounded largest entry).
    Additionally, we have the prior knowledge that both the vectors $x$ and $y$ are nonnegative.
    The corresponding blind deconvolution problem can be formulated as the following biconvex optimization problem:

    \[
        \begin{array}{ll}
            \text{minimize} & {\|x \otimes  y - d\|}_2^2 + \alpha_{\rm sp} {\|x\|}_1 + \alpha_{\rm sm} {\|Dy\|}_2^2\\
            \text{subject to} & x \succeq 0,\quad y \succeq 0\\
            & {\|y\|}_\infty \leq \beta
        \end{array}
    \]

    with variables $x$ and $y$, where $\alpha_{\rm sp}, \alpha_{\rm sm} > 0$ are the regularization parameters
    for the sparsity of $x$ and smoothness of $y$, respectively, and $\beta > 0$ is the bound on the
    $\ell_\infty$-norm of the vector $y$.
    The matrix $D \in \mathbf{R}^{(m - 1) \times m}$ is the first-order difference operator, given by,

    \[
        D = \left[\begin{array}{ccccc}
            1 & -1 &&&\\
            & 1 & -1 &&\\
            && \ddots & \ddots &\\
            &&& 1 & -1
        \end{array}\right] \in \mathbf{R}^{(m - 1) \times m},
    \]

    so that $Dy$ computes the vector of successive differences of $y$.
    The convolution $x \otimes y$ of the vectors $x$ and $y$ is given by

    \[
        {(x \otimes y)}_k = \sum_{i + j = k} x_i y_j,\quad k = 1, \ldots, m + n - 1.
    \]
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
    n = 120
    m = 40

    x0 = np.zeros(n)
    x0[6] = 1
    y0 = np.exp(-np.square(np.linspace(-2, 2, m)) * 2)
    d = np.convolve(x0, y0)
    return d, m, n, x0, y0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the problem
    """)
    return


@app.cell
def _(BiconvexProblem, convolve, cp, d, m, n):
    alpha_sp = 0.1
    alpha_sm = 0.2
    beta = 1

    x = cp.Variable(n, nonneg=True)
    y = cp.Variable(m, nonneg=True)
    obj = cp.Minimize(
        cp.sum_squares(convolve(x, y) - d) + alpha_sp * cp.norm1(x) + alpha_sm * cp.sum_squares(cp.diff(y))
    )
    constr = [cp.norm(y, "inf") <= beta]
    prob = BiconvexProblem(obj, [[x], [y]], constr)
    prob.solve(cp.CLARABEL, abs_tol=1e-5, max_iter=200)
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(d, figure_directory, np, plt, x, x0, y, y0):
    fig, axs = plt.subplots(1, 1, figsize=(6, 4.5))
    axs.plot(x0, linestyle="--", color="C3", linewidth=2)
    axs.plot(y0, linestyle="--", color="C1", linewidth=2)
    axs.plot(d, linestyle="--", color="k", linewidth=2)
    axs.plot(x.value, color="C0", marker=".", markersize=10)
    axs.plot(y.value, color="C2", marker="s")
    axs.plot(np.convolve(x.value, y.value), marker="D", color="C4", zorder=-1)

    axs.legend(
        ["ground truth $x$", "ground truth $y$", "ground truth $d$", "recovered $x$", "recovered $y$", "recovered $d$"],
        frameon=False,
        fontsize=12,
    )
    axs.set_xlim(0, 60)
    axs.set_xlabel("$i$")

    fig.tight_layout()
    fig.savefig(figure_directory / "blind_deconv.pdf", bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
