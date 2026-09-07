import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Constrained $k$-means clustering
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

    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    figure_directory = Path(__file__).resolve().parent / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(10015)
    return BiconvexProblem, cp, figure_directory, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We consider the problem of $k$-means clustering with constraints:

    \[
        \begin{array}{ll}
            \text{minimize} & \sum_{i = 1}^{m} z_i^T ({\|\bar{x}_1 - x_i\|}_2^2, \ldots, {\|\bar{x}_k - x_i\|}_2^2)\\
            \text{subject to} & 0 \preceq z_i \preceq \mathbf{1},\quad \mathbf{1}^T z_i = 1,\quad i = 1, \ldots, m\\
            & {\|\bar{x}_i - \mu_i\|}_2 \leq r_i
        \end{array}
    \]

    with variables $\bar{x}_i \in \mathbf{R}^n$, $i = 1, \ldots, k$, and $z_i \in \mathbf{R}^k$, $i = 1, \ldots, m$.
    The constraints ${\|\bar{x}_i - \mu_i\|}_2 \leq r_i$ can be interpreted as limiting the cluster center
    $\bar{x}_i$ to be within the Euclidean ball with (given) center $\mu_i$ and radius $r_i$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate dataset
    """)
    return


@app.cell
def _(np):
    n = 2
    k = 3
    _m_cluster = [180, 180, 180]
    _m_outlier = 24
    m = np.sum(_m_cluster) + k * _m_outlier

    mu_true = np.array(
        [
            [0.25, -0.15],
            [4.65, 0.20],
            [2.60, 4.30],
        ]
    )
    mus = mu_true + np.random.normal(0, 1, size=mu_true.shape)
    r = 0.5

    # core data points
    _covs = [
        np.array([[0.45, 0.10], [0.10, 0.30]]),
        np.array([[0.35, -0.08], [-0.08, 0.50]]),
        np.array([[0.40, 0.00], [0.00, 0.40]]),
    ]

    _xs = []
    for _p in range(k):
        _xs.append(np.random.multivariate_normal(mu_true[_p], _covs[_p], size=_m_cluster[_p]))
    x_core = np.vstack(_xs)

    # outlier points
    _pull = np.array(
        [
            [3.0, -2.0],
            [-2.5, 2.0],
            [1.5, -3.0],
        ]
    )

    _xs = []
    for _p in range(k):
        _base = mu_true[_p] + _pull[_p]
        _xs.append(_base + np.random.normal(0, 0.35, size=(_m_outlier, n)))

    x_out = np.vstack(_xs)

    # combine
    xs = np.vstack([x_core, x_out])
    return k, m, mus, n, r, xs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify the problem and solve
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standard $k$-means
    """)
    return


@app.cell
def _(BiconvexProblem, cp, k, m, n, xs):
    xbars = cp.Variable((k, n))
    zs = cp.Variable((m, k), nonneg=True)
    _obj = cp.sum(cp.multiply(zs, cp.vstack([cp.sum(cp.square(xs - c), axis=1) for c in xbars]).T))
    _constr = [zs <= 1, cp.sum(zs, axis=1) == 1]
    _prob = BiconvexProblem(cp.Minimize(_obj), [[xbars], [zs]], _constr)
    _prob.solve(cp.CLARABEL, lbd=2)
    return (xbars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $k$-means with constraints
    """)
    return


@app.cell
def _(BiconvexProblem, cp, k, m, mus, n, r, xs):
    xbars_constr = cp.Variable((k, n))
    zs_constr = cp.Variable((m, k), nonneg=True)
    _obj = cp.sum(cp.multiply(zs_constr, cp.vstack([cp.sum(cp.square(xs - c), axis=1) for c in xbars_constr]).T))
    _constr = [zs_constr <= 1, cp.sum(zs_constr, axis=1) == 1]
    for _c, _mu in zip(xbars_constr, mus):
        _constr.append(cp.norm2(_c - _mu) <= r)
    _prob = BiconvexProblem(cp.Minimize(_obj), [[xbars_constr], [zs_constr]], _constr)
    _prob.solve(cp.CLARABEL, lbd=10, abs_tol=1e-1)
    return (xbars_constr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(figure_directory, mus, plt, xbars, xbars_constr, xs):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(xs[:, 0], xs[:, 1], s=20, alpha=0.5)
    axs.scatter(mus[:, 0], mus[:, 1], s=100, color="r", marker="^", label=r"$\mu_i$")
    axs.scatter(
        xbars_constr.value[:, 0],
        xbars_constr.value[:, 1],
        s=100,
        color="g",
        marker="*",
        edgecolors="k",
        linewidth=0.5,
        label=r"$\bar{x}_i^{\rm con}$",
    )
    axs.scatter(xbars.value[:, 0], xbars.value[:, 1], s=50, color="k", marker="x", label=r"$\bar{x}_i^{\rm unc}$")
    for _x, _y in mus:
        axs.add_patch(plt.Circle((_x, _y), 0.5, fill=False, edgecolor="k", linewidth=1, linestyle="dashed"))
    axs.set_xlabel("$x_1$")
    axs.set_ylabel("$x_2$")

    axs.legend(frameon=False, handlelength=0.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_directory / "kmeans_constr.pdf", bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
