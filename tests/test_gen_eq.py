import cvxpy as cp
import numpy as np

import dbcp

np.random.seed(10015)


def test_psd():
    # Generate data matrix A
    m = 5
    n = 10
    k = 5
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    X = cp.Variable((m, k))
    Y = cp.Variable((k, n))

    # Define the biconvex problem
    obj = cp.Minimize(cp.sum_squares(X @ Y - A))
    constr = [X >> 0]
    prob = dbcp.BiconvexProblem(obj, [[Y], [X]], constr)

    X.value = np.random.randn(m, k)
    assert not constr[0].value()

    prob.solve()
    assert constr[0].value()
    reconstruction_error = np.linalg.norm(A - X.value @ Y.value, "fro") ** 2
    assert reconstruction_error < 1e-1
