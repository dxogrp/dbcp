import warnings
from collections.abc import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

from dbcp.error import DBCPError, InitiationError, SolveError
from dbcp.fix import fix_prob
from dbcp.transform import relax_with_slack


class BiconvexProblem(cp.Problem):
    """A biconvex problem solved by proximal alternating convex search.

    The supplied variable partition defines the two fixed convex subproblems.
    """

    def __init__(
        self,
        biconvex_objective,
        fix_vars: tuple[Iterable[cp.Variable], Iterable[cp.Variable]],
        constraints: list[Constraint] | None = None,
    ) -> None:
        """Initialize a BiconvexProblem instance.

        Parameters
        ----------
        biconvex_objective : cp.Objective
            The biconvex objective function.
        fix_vars : tuple[Iterable[cp.Variable], Iterable[cp.Variable]]
            A tuple of two iterables of cvxpy Variables. The first iterable contains
            the variables to be optimized in the x-problem, and
            the second iterable contains the variables to be optimized in the y-problem.
        """
        super().__init__(biconvex_objective, constraints)
        self.fix_vars = fix_vars

        self._x_prob = fix_prob(self, self.fix_vars[1])
        self._y_prob = fix_prob(self, self.fix_vars[0])

        self._value: float | None = None
        self._status: str | None = None

    @property
    def x_prob(self) -> cp.Problem:
        """The x-problem with y-variables fixed."""
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        """The y-problem with x-variables fixed."""
        return self._y_prob

    @property
    def x_prob_(self) -> cp.Problem:
        """The x-problem with y-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        for p in self._x_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[1]]:
                var = [v for v in self.fix_vars[1] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        """The y-problem with x-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        for p in self._y_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[0]]:
                var = [v for v in self.fix_vars[0] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._y_prob

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
                for p in xproj_prob.parameters():
                    p.project_and_assign([v for v in self.fix_vars[1] if v.id == p.id][0].value)
                xproj_prob.solve(solver=solver)
                for p in yproj_prob.parameters():
                    p.project_and_assign([v for v in self.fix_vars[0] if v.id == p.id][0].value)
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
        self, solver: str = cp.SCS, lbd: float = 0.1, max_iter: int = 100, gap_tolerance: float = 1e-6, *args, **kwargs
    ) -> float | None:
        """Solve the biconvex problem using alternate convex search.

        Parameters
        ----------
        solver : str
            The cvxpy Solver to use for solving the convex subproblems.
        lbd : float
            The regularization parameter of the proximal term.
        max_iter : int
            The maximum number of ACS iterations.
        gap_tolerance : float
            The tolerance for the gap between x- and y-problems.
        *args, **kwargs : Additional arguments for the solver.
        """
        if not self.is_dbcp():
            raise DBCPError("Problem does not follow DBCP rules.")

        print(f"{' DBCP Summary ':=^{85}}")
        self._project(solver, kwargs.get("proj_max_iter", 10))

        print(f"Alternate convex search start with solver {solver}...")
        print("-" * 65)
        print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<10}")
        print("-" * 65)
        prox_params = [cp.Parameter(v.shape, id=v.id, **v.attributes) for v in self.variables()]
        x_prox = cp.Problem(
            cp.Minimize(
                cp.multiply(
                    lbd,
                    cp.sum(
                        [
                            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
                            for v in self.x_prob.variables()
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
                            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
                            for v in self.y_prob.variables()
                        ]
                    ),
                )
            )
        )
        if self.objective.NAME == "minimize":
            xprox_prob = self.x_prob + x_prox
            yprox_prob = self.y_prob + y_prox
        else:
            xprox_prob = self.x_prob - x_prox
            yprox_prob = self.y_prob - y_prox
        i = 0
        try:
            while True:
                for v in self.x_prob_.variables():
                    [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                xprox_prob.solve(solver=solver, *args, **kwargs)
                xvalue = self.x_prob.objective.value
                for v in self.y_prob_.variables():
                    [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                yprox_prob.solve(solver=solver, *args, **kwargs)
                yvalue = self.y_prob.objective.value

                if (xprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) or (
                    yprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                ):
                    raise SolveError(f"Solver {solver} failed. Try a different solver.")
                gap = np.abs(xvalue - yvalue)
                print(f"{i:<7} {xvalue:<20.9f} {yvalue:<20.9f} {gap:<10.4e}")
                if gap < gap_tolerance:
                    self._status = "converge"
                    break
                else:
                    i += 1
                if i == max_iter:
                    self._status = "converge_inaccurate"
                    break
        except cp.SolverError as e:
            raise SolveError("Solver failed. Try with larger 'lbd' value.") from e

        print("-" * 65)
        print(f"Terminated with status: {self.status}.")
        print("=" * 85)
        self._value = self.y_prob.objective.value
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


class BiconvexRelaxProblem(cp.Problem):
    """A slack-relaxed biconvex problem solved by alternating convex search.

    The original constraints receive penalized slack variables so the
    alternating iterations may begin from an infeasible point.
    """

    def __init__(
        self,
        biconvex_objective,
        fix_vars: tuple[Iterable[cp.Variable], Iterable[cp.Variable]],
        constraints: list[Constraint] | None = None,
    ) -> None:
        """Initialize a BiconvexRelaxProblem instance.

        Parameters
        ----------
        biconvex_objective : cp.Objective
            The biconvex objective function.
        fix_vars : tuple[Iterable[cp.Variable], Iterable[cp.Variable]]
            A tuple of two iterables of cvxpy Variables. The first iterable contains
            the variables to be optimized in the x-problem, and
            the second iterable contains the variables to be optimized in the y-problem.
        """
        super().__init__(biconvex_objective, constraints)
        self.fix_vars = fix_vars

        self.nu = cp.Parameter((), nonneg=True)
        self._rlx_prob, self.slack_vars = relax_with_slack(self, self.nu)
        self._x_prob = fix_prob(self._rlx_prob, self.fix_vars[1])
        self._y_prob = fix_prob(self._rlx_prob, self.fix_vars[0])

        self._value: float | None = None
        self._status: str | None = None

    @property
    def x_prob(self) -> cp.Problem:
        """The relaxed x-problem with y-variables fixed."""
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        """The relaxed y-problem with x-variables fixed."""
        return self._y_prob

    @property
    def rlx_prob(self) -> cp.Problem:
        """The relaxed problem with slack variables added to constraints."""
        return self._rlx_prob

    @property
    def x_prob_(self) -> cp.Problem:
        """The relaxed x-problem with y-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        for p in self._x_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[1]]:
                var = [v for v in self.fix_vars[1] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._x_prob

    @property
    def y_prob_(self) -> cp.Problem:
        """The relaxed y-problem with x-variables fixed.
        Calling this property also updates the fixed variables to current values."""
        for p in self._y_prob.parameters():
            if p.id in [v.id for v in self.fix_vars[0]]:
                var = [v for v in self.fix_vars[0] if v.id == p.id][0]
                if var.value is not None:
                    p.project_and_assign(var.value)
        return self._y_prob

    def solve(
        self,
        solver: str = cp.SCS,
        lbd: float = 0.1,
        nu: float = 1,
        max_iter: int = 100,
        gap_tolerance: float = 1e-6,
        slack_tolerance: float = 1e-6,
        *args,
        **kwargs,
    ) -> float | None:
        """Solve the biconvex problem using infeasible start alternate convex search.

        Parameters
        ----------
        solver : str
            The cvxpy Solver to use for solving the convex subproblems.
        lbd : float
            The regularization parameter of the proximal term.
        nu : float
            The penalty parameter for the total slackness.
        max_iter : int
            The maximum number of ACS iterations.
        gap_tolerance : float
            The tolerance for the gap between x- and y-problems.
        slack_tolerance : float
            The tolerance for the total slackness.
        *args, **kwargs
            Additional arguments to pass to the solver.
        """
        if not self.is_dbcp():
            raise DBCPError("Problem does not follow DBCP rules.")

        print(f"{' DBCP Summary ':=^{85}}")
        print(f"Alternate convex search start with solver {solver}...")
        print("-" * 85)
        print(f"{'iter':<7} {'xcost':<20} {'ycost':<20} {'gap':<20} {'total_slack':<20}")
        print("-" * 85)
        for v in self.variables():
            if v.value is None:
                v.project_and_assign(np.random.standard_normal(v.shape))
        slack_ids = sorted([s.id for s in self.slack_vars])
        prox_params = [cp.Parameter(v.shape, id=v.id, **v.attributes) for v in self.variables()]
        x_prox = cp.Problem(
            cp.Minimize(
                cp.multiply(
                    lbd,
                    cp.sum(
                        [
                            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
                            for v in self.x_prob.variables()
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
                            cp.sum_squares([p for p in prox_params if p.id == v.id][0] - v)
                            for v in self.y_prob.variables()
                            if v.id not in slack_ids
                        ]
                    ),
                )
            )
        )
        if self.objective.NAME == "minimize":
            xprox_prob = self.x_prob + x_prox
            yprox_prob = self.y_prob + y_prox
        else:
            xprox_prob = self.x_prob - x_prox
            yprox_prob = self.y_prob - y_prox
        self.nu.value = nu
        i = 0
        try:
            while True:
                for v in self.x_prob_.variables():
                    if v.id not in slack_ids:
                        [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                xprox_prob.solve(solver=solver, *args, **kwargs)
                xvalue = self.x_prob.objective.value
                for v in self.y_prob_.variables():
                    if v.id not in slack_ids:
                        [p for p in prox_params if p.id == v.id][0].project_and_assign(v.value)
                yprox_prob.solve(solver=solver, *args, **kwargs)
                yvalue = self.y_prob.objective.value

                if (xprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) or (
                    yprox_prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                ):
                    raise SolveError(f"Solver {solver} failed. Try a different solver.")
                gap = np.abs(xvalue - yvalue)
                total_slack = np.sum([np.sum(np.abs(s.value)) for s in self.slack_vars])
                print(f"{i:<7} {xvalue:<20.9f} {yvalue:<20.9f} {gap:<20.4e} {total_slack:<20.4e} ")
                if gap < gap_tolerance:
                    if total_slack < slack_tolerance:
                        self._status = "converge"
                    else:
                        self._status = "converge_infeasible"
                    break
                else:
                    i += 1
                if i == max_iter:
                    if total_slack < slack_tolerance:
                        self._status = "converge_inaccurate"
                    else:
                        self._status = "converge_inaccurate_infeasible"
                    break
        except cp.SolverError as e:
            raise SolveError("Solver failed. Try with larger 'lbd' value.") from e

        print("-" * 85)
        print(f"Terminated with status: {self.status}.")
        print("=" * 85)
        self._value = self.objective.value
        if "infeasible" in self.status:
            warnings.warn(
                f"The returned solution is infeasible with total constraint violation {total_slack}."
                f"Consider increasing 'nu' value or trying another initial point."
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
