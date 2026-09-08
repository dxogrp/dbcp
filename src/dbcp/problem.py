import inspect
import warnings
from collections.abc import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

from dbcp.error import DBCPError, InitiationError, SolveError
from dbcp.fix import fix_prob
from dbcp.transform import relax_with_slack

_MISSING = object()
_LEGACY_FIX_VARS_ERROR = "The grouped 'fix_vars' API was removed; pass x_var and y_var as separate arguments."


def _validate_tolerance(name: str, value: float) -> None:
    """Validate a convergence tolerance."""
    try:
        is_valid = bool(np.isfinite(value)) and value >= 0
    except (TypeError, ValueError):
        is_valid = False
    if not is_valid:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_positive_penalty(value: float) -> None:
    """Validate the penalty parameter."""
    try:
        is_valid = bool(np.isfinite(value)) and value > 0
    except (TypeError, ValueError):
        is_valid = False
    if not is_valid:
        raise ValueError("nu must be finite and positive.")


def _objective_gap_within_tolerance(
    u: float,
    v: float,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """Return whether two objective values meet the convergence criterion."""
    return bool(np.abs(u - v) <= abs_tol + rel_tol * max(np.abs(u), np.abs(v)))


def _normalize_variable_group(
    name: str,
    variables: Iterable[cp.Variable],
    problem_variable_ids: set[int],
) -> tuple[cp.Variable, ...]:
    """Materialize and validate one of the two fixed-variable groups."""
    if isinstance(variables, cp.Variable):
        raise TypeError(
            f"{name} must be an iterable of CVXPY Variables; "
            f"pass [{name.removesuffix('_var')}] rather than a single Variable."
        )
    try:
        group = tuple(variables)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of CVXPY Variables.") from error

    if any(not isinstance(variable, cp.Variable) for variable in group):
        raise TypeError(f"{name} must contain only CVXPY Variables.")

    missing = [variable.name() for variable in group if variable.id not in problem_variable_ids]
    if missing:
        raise ValueError(f"Every variable in {name} must appear in the problem; missing: {missing}.")
    return group


def _looks_like_legacy_fix_vars(value: object) -> bool:
    """Return whether a value has the former two-group constructor shape."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(
        isinstance(group, (list, tuple)) and all(isinstance(variable, cp.Variable) for variable in group)
        for group in value
    )


class BiconvexProblem(cp.Problem):
    """A biconvex problem solved by proximal alternating convex search.

    The supplied variable groups define the two fixed convex subproblems.
    """

    def __init__(
        self,
        biconvex_objective,
        x_var: Iterable[cp.Variable] | object = _MISSING,
        y_var: Iterable[cp.Variable] | object = _MISSING,
        constraints: list[Constraint] | None = None,
        *,
        fix_vars: object = _MISSING,
    ) -> None:
        """Initialize a BiconvexProblem instance.

        Parameters
        ----------
        biconvex_objective : cp.Objective
            The biconvex objective function.
        x_var : Iterable[cp.Variable]
            Variables optimized in the x-problem and fixed in the y-problem.
        y_var : Iterable[cp.Variable]
            Variables optimized in the y-problem and fixed in the x-problem.
        constraints : list[Constraint] | None
            The problem constraints.
        """
        if fix_vars is not _MISSING or _looks_like_legacy_fix_vars(x_var):
            raise TypeError(_LEGACY_FIX_VARS_ERROR)
        if x_var is _MISSING:
            raise TypeError("x_var is required.")
        if y_var is _MISSING:
            raise TypeError("y_var is required; pass x_var and y_var as separate arguments.")

        super().__init__(biconvex_objective, constraints)
        problem_variable_ids = {variable.id for variable in self.variables()}
        x_group = _normalize_variable_group("x_var", x_var, problem_variable_ids)
        y_group = _normalize_variable_group("y_var", y_var, problem_variable_ids)
        if {variable.id for variable in x_group} & {variable.id for variable in y_group}:
            raise ValueError("x_var and y_var must be disjoint.")
        self.fix_vars = (x_group, y_group)

        self._x_prob = fix_prob(self, self.fix_vars[1])
        self._y_prob = fix_prob(self, self.fix_vars[0])

        self._penalty_nu: cp.Parameter | None = None
        self._penalty_prob: cp.Problem | None = None
        self._penalty_slack_vars: tuple[cp.Variable, ...] = ()
        self._penalty_x_prob: cp.Problem | None = None
        self._penalty_y_prob: cp.Problem | None = None

        self._value: float | None = None
        self._status: str | None = None

    @property
    def x_prob(self) -> cp.Problem:
        """The direct x-problem with y-variables fixed."""
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        """The direct y-problem with x-variables fixed."""
        return self._y_prob

    @property
    def penalty_prob(self) -> cp.Problem:
        """The lazily constructed slack-penalized problem."""
        self._ensure_penalty_problems()
        assert self._penalty_prob is not None
        return self._penalty_prob

    @property
    def penalty_x_prob(self) -> cp.Problem:
        """The slack-penalized x-problem with y-variables fixed."""
        x_prob, _ = self._ensure_penalty_problems()
        return x_prob

    @property
    def penalty_y_prob(self) -> cp.Problem:
        """The slack-penalized y-problem with x-variables fixed."""
        _, y_prob = self._ensure_penalty_problems()
        return y_prob

    @property
    def slack_vars(self) -> tuple[cp.Variable, ...]:
        """The slack variables in the lazily constructed penalty problem."""
        self._ensure_penalty_problems()
        return self._penalty_slack_vars

    @property
    def x_prob_(self) -> cp.Problem:
        """The x-problem with y-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        self._update_fixed_values(self._x_prob, self.fix_vars[1])
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        """The y-problem with x-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        self._update_fixed_values(self._y_prob, self.fix_vars[0])
        return self._y_prob

    @staticmethod
    def _update_fixed_values(
        problem: cp.Problem,
        fixed_variables: tuple[cp.Variable, ...],
    ) -> None:
        """Update fixed-variable parameters from the variables' current values."""
        variables_by_id = {variable.id: variable for variable in fixed_variables}
        for parameter in problem.parameters():
            variable = variables_by_id.get(parameter.id)
            if variable is not None and variable.value is not None:
                parameter.project_and_assign(variable.value)

    def _ensure_penalty_problems(self) -> tuple[cp.Problem, cp.Problem]:
        """Construct the slack-penalized subproblems on first use."""
        if self._penalty_prob is None:
            self._penalty_nu = cp.Parameter((), nonneg=True)
            penalty_prob, slack_vars = relax_with_slack(self, self._penalty_nu)
            self._penalty_prob = penalty_prob
            self._penalty_slack_vars = tuple(slack_vars)
            self._penalty_x_prob = fix_prob(penalty_prob, self.fix_vars[1])
            self._penalty_y_prob = fix_prob(penalty_prob, self.fix_vars[0])
        assert self._penalty_x_prob is not None
        assert self._penalty_y_prob is not None
        return self._penalty_x_prob, self._penalty_y_prob

    def _project(self, solver, proj_max_iter) -> None:
        print("Initiation start...")
        for v in self.variables():
            if v.value is None:
                v.project_and_assign(np.random.standard_normal(v.shape))
        if all([c.value() for c in self.constraints]):
            print("All constraints satisfied.")
        else:
            print("Finding a feasible initial point...")
            print("-" * 85)
            print(f"{'iter':<7} {'residual':<20}")
            print("-" * 65)
            proj_prob, _ = relax_with_slack(self)
            xproj_prob = fix_prob(proj_prob, self.fix_vars[1])
            yproj_prob = fix_prob(proj_prob, self.fix_vars[0])
            i = 0
            while True:
                self._update_fixed_values(xproj_prob, self.fix_vars[1])
                xproj_prob.solve(solver=solver)
                self._update_fixed_values(yproj_prob, self.fix_vars[0])
                yproj_prob.solve(solver=solver)

                print(f"{i:<7} {yproj_prob.value:<20.9f}")
                if all([c.value() for c in self.constraints]):
                    print("-" * 65)
                    print(f"Found feasible point in {i + 1} iterations.")
                    break
                else:
                    i += 1
                if i == proj_max_iter:
                    raise InitiationError("Cannot find a feasible point. Try different initial values.")

    def solve(
        self,
        solver: str = cp.SCS,
        lbd: float = 0.1,
        max_iter: int = 100,
        abs_tol: float = 1e-6,
        *args,
        rel_tol: float = 1e-6,
        mode: str = "direct",
        nu: float | None = None,
        slack_tol: float | None = None,
        **kwargs,
    ) -> float | None:
        """Solve the biconvex problem using direct or penalty ACS.

        Parameters
        ----------
        solver : str
            The cvxpy Solver to use for solving the convex subproblems.
        lbd : float
            The regularization parameter of the proximal term.
        max_iter : int
            The maximum number of ACS iterations.
        abs_tol : float
            The absolute tolerance for the gap between x- and y-problems.
        rel_tol : float
            The relative tolerance for the gap between x- and y-problems.
        mode : {"direct", "penalty"}
            The solution mode. Direct mode finds a feasible initial point; penalty
            mode adds penalized slacks and permits an infeasible start.
        nu : float | None
            The finite, strictly positive penalty applied to total slack in penalty
            mode. The effective default is 1. A non-None value is invalid in direct
            mode; None is equivalent to omission.
        slack_tol : float | None
            The total-slack tolerance in penalty mode. The effective default is
            1e-6. A non-None value is invalid in direct mode; None is equivalent to
            omission.
        *args
            Additional positional arguments forwarded to each alternating
            subproblem solve, but not to feasible initialization.
        **kwargs
            Additional keyword arguments forwarded to each alternating subproblem
            solve. ``proj_max_iter`` instead configures feasible initialization,
            and ``method`` retains its CVXPY meaning.
        """
        self._status = None
        self._value = None

        if "gap_tolerance" in kwargs:
            raise TypeError("'gap_tolerance' was removed; use 'abs_tol' and 'rel_tol'.")
        if "slack_tolerance" in kwargs:
            raise TypeError("'slack_tolerance' was removed; use 'slack_tol'.")
        cvxpy_method = kwargs.get("method")
        if (
            isinstance(cvxpy_method, str)
            and cvxpy_method in ("direct", "penalty")
            and cvxpy_method not in cp.Problem.REGISTERED_SOLVE_METHODS
        ):
            raise TypeError("The DBCP 'method' selector was renamed to 'mode'; use mode='direct' or mode='penalty'.")
        proj_max_iter = kwargs.pop("proj_max_iter", 10)
        _validate_tolerance("abs_tol", abs_tol)
        _validate_tolerance("rel_tol", rel_tol)
        if not isinstance(mode, str) or mode not in ("direct", "penalty"):
            raise ValueError("mode must be either 'direct' or 'penalty'.")
        is_penalty = mode == "penalty"
        if is_penalty:
            penalty_nu = 1 if nu is None else nu
            penalty_slack_tol = 1e-6 if slack_tol is None else slack_tol
            _validate_positive_penalty(penalty_nu)
            _validate_tolerance("slack_tol", penalty_slack_tol)
        else:
            invalid_options = [name for name, value in (("nu", nu), ("slack_tol", slack_tol)) if value is not None]
            if invalid_options:
                options = " and ".join(invalid_options)
                raise ValueError(f"{options} may only be used with mode='penalty'.")
            penalty_nu = 1
            penalty_slack_tol = 1e-6

        if not self.is_dbcp():
            raise DBCPError("Problem does not follow DBCP rules.")

        print(f"{' DBCP Summary ':=^{85}}")
        if mode == "direct":
            self._project(solver, proj_max_iter)
            x_prob, y_prob = self.x_prob, self.y_prob
            slack_vars: tuple[cp.Variable, ...] = ()
        else:
            x_prob, y_prob = self._ensure_penalty_problems()
            assert self._penalty_nu is not None
            self._penalty_nu.value = penalty_nu
            slack_vars = self._penalty_slack_vars
            for variable in self.variables():
                if variable.value is None:
                    variable.project_and_assign(np.random.standard_normal(variable.shape))

        print(f"{mode.capitalize()} alternate convex search start with solver {solver}...")
        rule_width = 85 if is_penalty else 65
        print("-" * rule_width)
        if is_penalty:
            print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<20} {'total_slack':<20}")
        else:
            print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<10}")
        print("-" * rule_width)

        slack_ids = {slack.id for slack in slack_vars}
        prox_params = [cp.Parameter(v.shape, id=v.id, **v.attributes) for v in self.variables()]
        prox_params_by_id = {parameter.id: parameter for parameter in prox_params}
        x_prox = cp.Problem(
            cp.Minimize(
                cp.multiply(
                    lbd,
                    cp.sum(
                        [
                            cp.sum_squares(prox_params_by_id[v.id] - v)
                            for v in x_prob.variables()
                            if v.id not in slack_ids
                        ]
                    ),
                )
            )
        )
        y_prox = cp.Problem(
            cp.Minimize(
                cp.multiply(
                    lbd,
                    cp.sum(
                        [
                            cp.sum_squares(prox_params_by_id[v.id] - v)
                            for v in y_prob.variables()
                            if v.id not in slack_ids
                        ]
                    ),
                )
            )
        )
        if self.objective.NAME == "minimize":
            xprox_prob = x_prob + x_prox
            yprox_prob = y_prob + y_prox
        else:
            xprox_prob = x_prob - x_prox
            yprox_prob = y_prob - y_prox
        i = 0
        total_slack = 0.0
        has_excess_slack = False
        try:
            while True:
                self._update_fixed_values(x_prob, self.fix_vars[1])
                for v in x_prob.variables():
                    if v.id not in slack_ids:
                        prox_params_by_id[v.id].project_and_assign(v.value)
                xprox_prob.solve(solver=solver, *args, **kwargs)
                xvalue = x_prob.objective.value
                self._update_fixed_values(y_prob, self.fix_vars[0])
                for v in y_prob.variables():
                    if v.id not in slack_ids:
                        prox_params_by_id[v.id].project_and_assign(v.value)
                yprox_prob.solve(solver=solver, *args, **kwargs)
                yvalue = y_prob.objective.value

                if (xprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) or (
                    yprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                ):
                    raise SolveError(f"Solver {solver} failed. Try a different solver.")
                gap = np.abs(xvalue - yvalue)
                if is_penalty:
                    total_slack = float(np.sum([np.sum(np.abs(slack.value)) for slack in slack_vars]))
                    if not np.isfinite(total_slack):
                        raise SolveError("Solver returned non-finite total slack.")
                    has_excess_slack = total_slack > penalty_slack_tol
                    print(f"{i:<7} {xvalue:<20.9f} {yvalue:<20.9f} {gap:<20.4e} {total_slack:<20.4e} ")
                else:
                    print(f"{i:<7} {xvalue:<20.9f} {yvalue:<20.9f} {gap:<10.4e}")
                if _objective_gap_within_tolerance(
                    xvalue,
                    yvalue,
                    abs_tol,
                    rel_tol,
                ):
                    if is_penalty:
                        self._status = "converge_with_slack" if has_excess_slack else "converge"
                    else:
                        self._status = "converge"
                    break
                else:
                    i += 1
                if i == max_iter:
                    if is_penalty:
                        self._status = "converge_inaccurate_with_slack" if has_excess_slack else "converge_inaccurate"
                    else:
                        self._status = "converge_inaccurate"
                    break
        except cp.SolverError as e:
            raise SolveError("Solver failed. Try with larger 'lbd' value.") from e

        print("-" * rule_width)
        print(f"Terminated with status: {self.status}.")
        print("=" * 85)
        self._value = self.objective.value
        if is_penalty and has_excess_slack:
            warnings.warn(
                f"The returned solution has total slack {total_slack}, which exceeds slack_tol={penalty_slack_tol}. "
                "Consider increasing 'nu' or trying another initial point."
            )
        return self.value

    @property
    def status(self) -> str | None:
        """The status of the last solve."""
        return self._status

    @property
    def value(self) -> float | None:
        """The objective value of the last solve."""
        return self._value

    def is_dbcp(self) -> bool:
        """Check if the problem follows DBCP rules."""
        if self.x_prob.is_dcp() and self.y_prob.is_dcp():
            return True
        return False


class BiconvexRelaxProblem:
    def __new__(cls, *_args, **_kwargs):
        raise TypeError(
            "'BiconvexRelaxProblem' was removed; use 'BiconvexProblem' and call solve(mode='penalty', ...)."
        )


BiconvexProblem.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter("biconvex_objective", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("x_var", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("y_var", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "constraints",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
    ]
)
