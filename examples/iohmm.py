import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting Input-output Hidden Markov Models
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

    from dbcp import BiconvexRelaxProblem

    _example_directory = Path(__file__).resolve().parent
    plt.style.use(_example_directory / "zhlatex.mplstyle")
    figure_directory = _example_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(10015)
    return BiconvexRelaxProblem, cp, figure_directory, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    We consider the fitting problem of a logistic input-output hidden Markov model (IO-HMM) to some dataset.
    Suppose we are given a dataset $(x(t), y(t))$, $t = 1, \ldots, m$, where each sample consists of an input
    feature vector $x(t) \in \mathbf{R}^n$ and an output label $y(t) \in \{0, 1\}$, generated from a $K$-state
    IO-HMM, according to the following procedure:
    Let $\hat{z}(t) \in \{1, \ldots, K\}$, $t = 1, \ldots, m$, be the state label of the IO-HMM with initial
    state distribution $p_{\rm init} \in \mathbf{R}^K$ with $\mathbf{1}^T p_{\rm init} = 1$ and transition matrix
    $P_{\rm tr} \in \mathbf{R}^{K \times K}$ with $P_{\rm tr} \mathbf{1} = \mathbf{1}$.
    At the time step $t$, the state label $\hat{z}(t)$ is sampled according to

    \[
        \hat{z}(t) \sim \left\{
            \begin{array}{ll}
                {\rm Cat}(p_{\rm init}) & t = 0\\
                {\rm Cat}(p_{\hat{z}(t - 1)}) & t > 0,
            \end{array}\right.
    \]

    where the vector $p_{\hat{z}(t-1)} \in \mathbf{R}^K$ denotes the $\hat{z}(t-1)$th row of the matrix
    $P_{\rm tr}$, and ${\rm Cat}(p)$ denotes the categorical distribution with $p$ being the vector of category
    probabilities.
    Then, given the feature vector $x(t) \in \mathbf{R}^n$, the output $y(t) \in \{0, 1\}$ of this IO-HMM at
    time step $t$ is then generated from a logistic model, i.e.,

    \[
        \mathop{\bf prob}(y(t) = 1) = \frac{1}{1 + \exp(-{x(t)}^T \theta_{\hat{z}(t)})},
    \]

    where $\theta_{\hat{z}(t)} \in \{\theta_1, \ldots, \theta_K\} \subseteq \mathbf{R}^n$ is the coefficient.

    We are interested in recovering the transition matrix $P_{\rm tr}$, the model parameters
    $\theta_1, \ldots, \theta_K$, and the unobserved state labels $\hat{z}(1), \ldots, \hat{z}(m)$, given the
    dataset $(x(t), y(t))$, $t = 1, \ldots, m$.
    Noticing that the transition matrix $P_{\rm tr}$ can be easily estimated from the state labels $\hat{z}(t)$,
    $t = 1, \ldots, m$, we consider the following biconvex optimization problem for fitting the IO-HMM:

    \[
        \begin{array}{ll}
            \text{minimize} & -\sum_{t = 1}^{m} {z(t)}^T {\left(y(t){x(t)}^T \theta_k -
    \log(1 + \exp({x(t)}^T \theta_k))\right)}_{k = 1}^K\\
            &\qquad + \alpha_\theta \sum_{k = 1}^{K} {\|\theta_k\|}^2_2 +
    \alpha_z \sum_{t = 1}^{m - 1} D_{\rm kl}(z(t), z(t + 1))\\
            \text{subject to} & 0 \preceq z(t) \preceq \mathbf{1},\quad \mathbf{1}^T z(t) = 1,\quad t = 1, \ldots, m\\
            & \theta_k \in {\cal C}_k,\quad k = 1, \ldots, K,
        \end{array}
    \]

    where the optimization variables are $\theta_k \in \mathbf{R}^n$, $k = 1, \ldots, K$, and
    $z(t) \in \mathbf{R}^K$, $t = 1, \ldots, m$.
    Note that the variable $z(t)$ is a soft assignment vector for the hidden state label $\hat{z}(t)$, where the
    $k$th entry of $z(t)$ indicates the probability of the state being $k$ at time step $t$, and $\hat{z}(t)$ can
    be estimated as the index of the largest entry of $z(t)$ after solving the problem above.

    Each component of this problem can be interpreted as follows:
    The first term in the objective function is the negative log-likelihood of the observed data under the IO-HMM
    model, given the state assignment probabilities $z(t)$, $t = 1, \ldots, m$, and the model parameters
    $\theta_k$, $k = 1, \ldots, K$.
    The second term is a Tikhonov regularization on the model parameters $\theta_k$, with regularization parameter
    $\alpha_\theta > 0$.
    The third term is a temporal smoothness regularization on the state assignment probabilities, where
    $D_{\rm kl}(p, q)$ denotes the Kullback-Leibler divergence between two probability distributions $p$ and $q$,
    and $\alpha_z > 0$ is the corresponding regularization parameter.
    The constraints on the variables $z(t)$, $t = 1, \ldots, m$, ensure that they are valid probability distributions.
    The sets ${\cal C}_k \subseteq \mathbf{R}^n$, $k = 1, \ldots, K$, are nonempty closed convex sets that encode
    potential prior knowledge about the model parameters $\theta_k$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate problem data

    We consider the case of $n = 2$, and the feature vector for each sample is generated according to

    \[
        x(t) \sim ({\cal U}(-5, 5),\ 1),
    \]

    where ${\cal U}(a, b)$ denotes a uniform distribution over the interval $[a, b]$, and the second entry of
    $x(t)$ is always $1$ to account for the bias term.
    """)
    return


@app.cell
def _(np):
    m = 1800
    n = 2
    K = 3
    coefs = np.array([[-1, 0], [2, 6], [2, -6]])
    p_tr = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])

    xs = np.random.uniform(-5, 5, m)
    xs = np.vstack([xs, np.ones(m)]).T

    ys = np.zeros(m)
    labels = np.zeros(m, dtype=int)

    _s = 0
    for _i, _feat in enumerate(xs):
        ys[_i] = 1 if np.random.uniform() < 1 / (1 + np.exp(-_feat @ coefs[_s])) else 0
        labels[_i] = _s
        _s = np.random.choice(K, p=p_tr[_s])
    return K, coefs, labels, m, n, p_tr, xs, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the problem

    To fully specify the biconvex problem, it is assumed that we are given the following prior knowledge about the
    coefficients:

    \[
        \theta_{1,1} \leq 0,\quad \theta_{2, 1} \geq 0,\quad \theta_{3, 1} \geq 0,
        \quad \theta_{2, 2} \geq \theta_{3, 2},
    \]

    where $\theta_{i, j}$ denotes the $j$th entry of the vector $\theta_i$.
    """)
    return


@app.cell
def _(BiconvexRelaxProblem, K, cp, m, n, xs, ys):
    thetas = cp.Variable((K, n))
    zs = cp.Variable((m, K), nonneg=True)

    alpha_theta = 0.1
    alpha_z = 2

    rs = [-cp.multiply(ys, xs @ thetas[k]) + cp.logistic(xs @ thetas[k]) for k in range(K)]
    obj = cp.Minimize(
        cp.sum(cp.multiply(zs, cp.vstack(rs).T))
        + alpha_theta * cp.sum_squares(thetas)
        + alpha_z * cp.sum(cp.kl_div(zs[:-1], zs[1:]))
    )
    constr = [
        thetas[0][0] <= 0,
        thetas[1][0] >= 0,
        thetas[2][0] >= 0,
        thetas[1][1] >= thetas[2][1],
        zs <= 1,
        cp.sum(zs, axis=1) == 1,
    ]

    prob = BiconvexRelaxProblem(obj, ([zs], [thetas]), constr)
    prob.solve(solver=cp.CLARABEL, nu=1e3, lbd=0.1, abs_tol=1e-3)
    return thetas, zs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results
    """)
    return


@app.cell
def _(K, coefs, figure_directory, labels, m, np, plt, thetas, zs):
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 7))

    axs[0].plot(labels, linestyle="dashed", color="k", linewidth=1, zorder=10)
    axs[0].plot(np.argmax(zs.value, axis=-1), color="r", linewidth=2)

    inputs = np.linspace(-5, 5, m)
    inputs = np.vstack([inputs, np.ones(m)]).T
    for _i in range(K):
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ coefs[_i])), linestyle="dashed", color="k", zorder=10)
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ thetas[_i].value)))

    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$\hat{z}(t)$")
    axs[0].set_yticks([0, 1, 2])
    axs[0].set_yticklabels([1, 2, 3])

    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_ylabel(r"$1/(1 + \exp(-x^T \theta))$")

    fig.tight_layout()
    fig.savefig(figure_directory / "iohmm.pdf", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(K, m, np, p_tr, zs):
    p_tr_hat = np.zeros_like(p_tr)
    z_hat = np.argmax(zs.value, axis=-1)
    for zi in range(K):
        z_idx = np.where(z_hat == zi)[0]
        z_idx = np.delete(z_idx, np.where(z_idx == m - 1)[0])
        _, nz_num = np.unique(z_hat[z_idx + 1], return_counts=True)
        p_tr_hat[zi] = nz_num / len(z_idx)

    print(p_tr_hat)
    return


if __name__ == "__main__":
    app.run()
