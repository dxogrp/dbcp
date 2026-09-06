"""Smoke-test an installed DBCP release distribution."""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import numpy as np

import dbcp


def _solution_values(result: float | None, *variables: cp.Variable) -> np.ndarray:
    if result is None or any(variable.value is None for variable in variables):
        raise RuntimeError("Release smoke solve did not return values for every variable.")
    values = [np.asarray(variable.value, dtype=float).item() for variable in variables]
    values.append(float(result))
    solution = np.asarray(values)
    if not np.isfinite(solution).all():
        raise RuntimeError(f"Release smoke solve returned non-finite values: {solution}.")
    return solution


def _smoke_convolution() -> None:
    left = np.array([1.0, 2.0, -1.0])
    right = np.array([-0.5, 3.0])
    actual = dbcp.convolve(cp.Constant(left), cp.Constant(right)).value
    expected = np.convolve(left, right)
    if actual is None or not np.allclose(actual, expected, atol=0.0, rtol=0.0):
        raise RuntimeError(f"Release smoke convolution returned {actual}, expected {expected}.")


def _smoke_biconvex_problem() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    x.value = 1.0
    y.value = 1.0
    problem = dbcp.BiconvexProblem(cp.Minimize(cp.square(x * y - 1.0)), ([x], [y]))
    result = problem.solve(
        solver=cp.SCS,
        max_iter=10,
        canon_backend=cp.SCIPY_CANON_BACKEND,
        ignore_dpp=True,
    )
    values = _solution_values(result, x, y)
    if problem.status != "converge" or abs(values[-1]) > 1e-8:
        raise RuntimeError(f"Biconvex release smoke solve failed: status={problem.status!r}, values={values}.")


def _smoke_biconvex_relax_problem() -> None:
    x = cp.Variable(name="relaxed_x")
    y = cp.Variable(name="relaxed_y")
    x.value = 1.0
    y.value = 1.0
    problem = dbcp.BiconvexRelaxProblem(
        cp.Minimize(cp.square(x * y - 1.0)),
        ([x], [y]),
        [x * y >= 0.5],
    )
    result = problem.solve(
        solver=cp.SCS,
        nu=100.0,
        max_iter=10,
        slack_tolerance=1e-5,
        canon_backend=cp.SCIPY_CANON_BACKEND,
        ignore_dpp=True,
    )
    values = _solution_values(result, x, y)
    if any(slack.value is None for slack in problem.slack_vars):
        raise RuntimeError("Relaxed release smoke solve did not return every slack value.")
    total_slack = sum(float(np.sum(np.abs(slack.value))) for slack in problem.slack_vars)
    if (
        problem.status != "converge"
        or abs(values[-1]) > 1e-8
        or values[0] * values[1] < 0.5 - 1e-5
        or total_slack > 1e-5
    ):
        raise RuntimeError(
            f"Relaxed release smoke solve failed: status={problem.status!r}, "
            f"values={values}, total_slack={total_slack}."
        )


def main() -> int:
    expected_version = os.environ["DBCP_RELEASE_VERSION"]
    repository_root = Path(os.environ["DBCP_REPOSITORY_ROOT"]).resolve()
    installed_package = Path(dbcp.__file__).resolve()
    if installed_package.is_relative_to(repository_root / "src"):
        raise RuntimeError(f"Smoke test imported the working tree instead of the distribution: {installed_package}.")
    if dbcp.__version__ != expected_version:
        raise RuntimeError(f"Installed DBCP version is {dbcp.__version__!r}, expected {expected_version!r}.")

    _smoke_convolution()
    _smoke_biconvex_problem()
    _smoke_biconvex_relax_problem()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
