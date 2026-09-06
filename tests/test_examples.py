import cvxpy as cp
import numpy as np

from dbcp import BiconvexProblem

np.random.seed(10015)


def test_nmf():
    # Generate data matrix A
    m = 5
    n = 10
    k = 5
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    X = cp.Variable((m, k), nonneg=True)
    Y = cp.Variable((k, n), nonneg=True)

    # Define the biconvex problem
    obj = cp.Minimize(cp.sum_squares(X @ Y - A))
    prob = BiconvexProblem(obj, [[X], [Y]])
    prob.solve()

    # Check that the solution is non-negative
    assert np.all(X.value >= 0)
    assert np.all(Y.value >= 0)

    # Check that the reconstruction error is reasonable
    reconstruction_error = np.linalg.norm(A - X.value @ Y.value, "fro") ** 2
    assert reconstruction_error < 1e-1


def test_blin_logi_reg():
    # Generate synthetic data
    m = 50
    n = 20
    k = 10
    r = 5
    Xs = np.random.randn(m, n, k)
    U_ = np.random.randn(n, r)
    V_ = np.random.randn(k, r)
    logits = np.array([np.trace(U_.T @ X @ V_) for X in Xs])
    ys = (logits > 0).astype(int)

    # Define variables
    U = cp.Variable((n, r))
    V = cp.Variable((k, r))

    # Define the biconvex problem
    U = cp.Variable((n, r))
    V = cp.Variable((k, r))

    obj = 0
    for _X, _y in zip(Xs, ys):
        obj += cp.sum(cp.multiply(_y, cp.trace(U.T @ _X @ V)) - cp.logistic(cp.trace(U.T @ _X @ V)))
    prob = BiconvexProblem(cp.Maximize(obj), [[U], [V]])
    prob.solve(cp.CLARABEL, lbd=10, gap_tolerance=1e-2)

    assert U.value is not None
    assert V.value is not None


def test_kmeans():
    # Generate synthetic data
    n = 2
    m = 40
    k = 4
    centers = [[0, 2], [0, -2], [2, 0], [-2, 0]]
    xs = np.vstack([np.random.randn(m // k, n) + center for center in centers])

    # Define variables
    xbars = cp.Variable((k, n))
    zs = cp.Variable((m, k), nonneg=True)

    # Define the biconvex problem
    obj = cp.sum(cp.multiply(zs, cp.vstack([cp.sum(cp.square(xs - c), axis=1) for c in xbars]).T))
    constr = [zs <= 1, cp.sum(zs, axis=1) == 1]
    prob = BiconvexProblem(cp.Minimize(obj), [[xbars], [zs]], constr)
    prob.solve()

    assert xbars.value is not None
