import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $k$-means Clustering
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from sklearn.datasets import make_blobs
    from dbcp import BiconvexProblem

    import matplotlib.pyplot as plt

    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    figure_directory = Path(__file__).resolve().parent / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(10015)
    return BiconvexProblem, cp, figure_directory, make_blobs, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Suppose we are given a set of data points $x_i \in \mathbf{R}^n$, $i = 1, \ldots, m$, and we would like to cluster them into $k$ groups, using the $k$-means clustering method.
    This corresponds to the following biconvex optimization problem:

    \[
        \begin{array}{ll}
            \text{minimize} & \sum_{i = 1}^{m} z_i^T ({\|\bar{x}_1 - x_i\|}_2^2, \ldots, {\|\bar{x}_k - x_i\|}_2^2)\\
            \text{subject to} & 0 \preceq z_i \preceq \mathbf{1},\quad \mathbf{1}^T z_i = 1,\quad i = 1, \ldots, m
        \end{array}
    \]

    with variables $\bar{x}_i \in \mathbf{R}^n$, $i = 1, \ldots, k$, and $z_i \in \mathbf{R}^k$, $i = 1, \ldots, m$.

    We can interpret the problem formulation as follows:
    The variables $\bar{x}_1, \ldots, \bar{x}_k$ represent the cluster centroids, and each variable $z_i$ is a soft assignment vector for data point $x_i$, where the $j$th entry of $z_i$ indicates the probability of the sample $x_i$ belonging to cluster $j$.
    Then, the objective function represents the total within-cluster variance, which we would like to minimize.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate problem data
    """)
    return


@app.cell
def _(make_blobs):
    n = 2
    m = 1000
    k = 4
    centers = [[0, 2], [0, -2], [2, 0], [-2, 0]]
    xs, labels = make_blobs(n_samples=m, centers=centers, cluster_std=0.5)
    return k, m, n, xs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the problem
    """)
    return


@app.cell
def _(BiconvexProblem, cp, k, m, n, xs):
    xbars = cp.Variable((k, n))
    zs = cp.Variable((m, k), nonneg=True)
    obj = cp.sum(cp.multiply(zs, cp.vstack([
        cp.sum(cp.square(xs - c), axis=1) for c in xbars
    ]).T))
    constr = [zs <= 1, cp.sum(zs, axis=1) == 1]
    prob = BiconvexProblem(cp.Minimize(obj), [[xbars], [zs]], constr)
    prob.solve()
    return xbars, zs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(figure_directory, np, plt, xbars, xs, zs):
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    _labels = np.argmax(zs.value, axis=-1)
    cmap = plt.get_cmap('tab10', np.unique(_labels).size)
    axs.scatter(xs[:, 0], xs[:, 1], s=10, c=_labels, cmap=cmap)
    axs.scatter(xbars.value[:, 0], xbars.value[:, 1], s=100, color='k', marker='x')
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')

    fig.tight_layout()
    fig.savefig(figure_directory / "kmeans.pdf", bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
