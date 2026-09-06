from collections.abc import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.nonpos import Inequality, NonNeg, NonPos
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Equality, Zero


def fix_prob(prob: cp.Problem, vars: Iterable[cp.Variable]) -> cp.Problem:
    all_vars = sorted(prob.variables(), key=lambda v: v.id)
    params = []
    for v in all_vars:
        p = cp.Parameter(shape=v.shape, id=v.id, **v.attributes)
        if v.value is not None:
            p.project_and_assign(v.value)
        else:
            p.project_and_assign(np.random.standard_normal(v.shape))
        params.append(p)

    params.sort(key=lambda p: p.id)
    if isinstance(prob, cp.Problem):
        return _fix_prob(prob, vars, params)
    else:
        raise TypeError("Object must be a cvxpy Problem.")


def _fix_prob(prob: cp.Problem, vars: Iterable[cp.Variable], params: list[cp.Parameter]) -> cp.Problem:
    fixed_fn = _fix_expr(prob.objective.expr, vars, params)
    fixed_obj = cp.Minimize(fixed_fn) if prob.objective.NAME == "minimize" else cp.Maximize(fixed_fn)
    fixed_constr = []
    for c in prob.constraints:
        if isinstance(c, Inequality):
            lhs = _fix_expr(c.args[0], vars, params)
            rhs = _fix_expr(c.args[1], vars, params)
            fixed_constr.append(lhs <= rhs)
        elif isinstance(c, Equality):
            lhs = _fix_expr(c.args[0], vars, params)
            rhs = _fix_expr(c.args[1], vars, params)
            fixed_constr.append(lhs == rhs)
        elif isinstance(c, Zero):
            fixed_constr.append(Zero(_fix_expr(c.expr, vars, params)))
        elif isinstance(c, NonPos):
            fixed_constr.append(NonNeg(-_fix_expr(c.expr, vars, params)))
        elif isinstance(c, NonNeg):
            fixed_constr.append(NonNeg(_fix_expr(c.expr, vars, params)))
        elif isinstance(c, PSD):
            fixed_constr.append(PSD(_fix_expr(c.expr, vars, params)))
        elif isinstance(c, SOC):
            t = _fix_expr(c.args[0], vars, params)
            X = _fix_expr(c.args[1], vars, params)
            fixed_constr.append(cp.SOC(t, X, axis=c.axis))
        else:
            raise TypeError(f"Constraint type {type(c)} not supported.")
    new_prob = cp.Problem(fixed_obj, fixed_constr)
    return new_prob


def _fix_expr(
    expr: cp.Expression, vars: Iterable[cp.Variable], params: list[cp.Parameter]
) -> cp.Parameter | cp.Expression:
    vars_id = sorted([v.id for v in vars])
    if isinstance(expr, cp.Variable) and expr.id in vars_id:
        param = [p for p in params if p.id == expr.id][0]
        return param
    elif len(expr.args) == 0:
        return expr
    else:
        fixed_args = []
        for arg in expr.args:
            fixed_args.append(_fix_expr(arg, vars, params))
        return expr.copy(args=fixed_args)
