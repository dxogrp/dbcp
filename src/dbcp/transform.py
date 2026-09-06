import cvxpy as cp
import numpy as np
from cvxpy.constraints.nonpos import Inequality, NonNeg, NonPos
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Equality, Zero


def relax_with_slack(
    prob: cp.Problem,
    nu: cp.Parameter = None,
) -> (cp.Problem, list[cp.Variable]):
    proj_constr = []
    slack_vars = []
    for c in prob.constraints:
        if isinstance(c, Inequality):
            slack_vars.append(cp.Variable(shape=c.shape, nonneg=True))
            proj_constr.append(c.args[0] <= c.args[1] + slack_vars[-1])
        elif isinstance(c, Equality):
            slack_vars.append(cp.Variable(shape=c.shape))
            proj_constr.append(c.args[0] == c.args[1] + slack_vars[-1])
        elif isinstance(c, Zero):
            slack_vars.append(cp.Variable(shape=c.shape))
            proj_constr.append(Zero(c.expr + slack_vars[-1]))
        elif isinstance(c, NonPos):
            slack_vars.append(cp.Variable(shape=c.shape, nonneg=True))
            proj_constr.append(NonNeg(-c.expr + slack_vars[-1]))
        elif isinstance(c, NonNeg):
            slack_vars.append(cp.Variable(shape=c.shape, nonneg=True))
            proj_constr.append(NonNeg(c.expr + slack_vars[-1]))
        elif isinstance(c, PSD):
            slack_vars.append(cp.Variable((), nonneg=True))
            proj_constr.append(PSD(c.expr + slack_vars[-1] * np.eye(c.shape[0])))
        elif isinstance(c, SOC):
            slack_vars.append(cp.Variable(shape=c.shape, nonneg=True))
            proj_constr.append(cp.SOC(c.args[0] + slack_vars[-1], c.args[1], axis=c.axis))
        else:
            raise TypeError(f"Constraint type {type(c)} not supported.")
    if nu is not None:
        if prob.objective.NAME == "minimize":
            proj_obj = prob.objective + cp.Minimize(nu * cp.sum([cp.norm1(s) for s in slack_vars]))
        else:
            proj_obj = prob.objective - cp.Minimize(nu * cp.sum([cp.norm1(s) for s in slack_vars]))
    else:
        proj_obj = cp.Minimize(cp.sum([cp.norm1(s) for s in slack_vars]))
    return cp.Problem(proj_obj, proj_constr), slack_vars
