import cvxpy as cp

import dbcp


def test_is_dbcp():
    x = cp.Variable()
    y = cp.Variable()
    obj = cp.Minimize(x * y)

    prob = cp.Problem(obj)
    prob_dbcp = dbcp.BiconvexProblem(obj, [x], [y])

    assert not prob.is_dcp()
    assert prob_dbcp.is_dbcp()
