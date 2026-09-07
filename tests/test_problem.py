import inspect

import cvxpy as cp
import numpy as np
import pytest

from dbcp import BiconvexProblem, BiconvexRelaxProblem
from dbcp.problem import _objective_gap_within_tolerance


@pytest.mark.parametrize(
    ("u", "v", "abs_tol", "rel_tol", "expected"),
    [
        pytest.param(0.0, 5e-7, 1e-6, 1e-6, True, id="near-zero-absolute"),
        pytest.param(1e6, 1e6 - 0.5, 0.0, 1e-6, True, id="large-scale-relative"),
        pytest.param(-1e6, -1e6 + 0.5, 0.0, 1e-6, True, id="negative-relative"),
        pytest.param(1_000.0, 999.0, 0.1, 9e-4, True, id="boundary"),
        pytest.param(1_000.0, 998.999, 0.1, 9e-4, False, id="just-outside"),
        pytest.param(4.0, 3.5, 0.5, 0.0, True, id="absolute-only"),
        pytest.param(4.0, 3.5, 0.0, 0.125, True, id="relative-only"),
    ],
)
def test_objective_gap_within_tolerance(
    u,
    v,
    abs_tol,
    rel_tol,
    expected,
):
    assert (
        _objective_gap_within_tolerance(
            u,
            v,
            abs_tol,
            rel_tol,
        )
        is expected
    )


def _make_problem(problem_type):
    x = cp.Variable()
    y = cp.Variable()
    x.value = 0.0
    y.value = 0.0
    return problem_type(
        cp.Minimize(cp.square(x - 1) + cp.square(y - 2)),
        [[x], [y]],
        [x >= 0, y >= 0],
    )


@pytest.mark.parametrize("problem_type", [BiconvexProblem, BiconvexRelaxProblem])
@pytest.mark.parametrize("tolerance_name", ["abs_tol", "rel_tol"])
@pytest.mark.parametrize("invalid_value", [-1e-6, np.nan, np.inf, -np.inf])
def test_solve_rejects_invalid_convergence_tolerances(
    problem_type,
    tolerance_name,
    invalid_value,
):
    problem = _make_problem(problem_type)

    with pytest.raises(ValueError, match=rf"{tolerance_name} must be finite and nonnegative"):
        problem.solve(**{tolerance_name: invalid_value})


@pytest.mark.parametrize("problem_type", [BiconvexProblem, BiconvexRelaxProblem])
def test_solve_rejects_legacy_gap_tolerance(problem_type):
    problem = _make_problem(problem_type)

    with pytest.raises(TypeError, match="gap_tolerance.*abs_tol.*rel_tol"):
        problem.solve(gap_tolerance=1e-6)


@pytest.mark.parametrize("problem_type", [BiconvexProblem, BiconvexRelaxProblem])
def test_convergence_tolerance_signature(problem_type):
    parameters = inspect.signature(problem_type.solve).parameters

    assert parameters["abs_tol"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["abs_tol"].default == 1e-6
    assert parameters["rel_tol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["rel_tol"].default == 1e-6
    if problem_type is BiconvexRelaxProblem:
        parameter_names = list(parameters)
        assert parameter_names.index("slack_tolerance") == parameter_names.index("abs_tol") + 1


def test_biconvex_problem_solves_with_scale_aware_tolerances():
    problem = _make_problem(BiconvexProblem)

    value = problem.solve(
        cp.CLARABEL,
        abs_tol=1e6,
        rel_tol=1e-4,
    )

    assert value is not None
    assert problem.status == "converge"


def test_relaxed_problem_solves_with_scale_aware_tolerances():
    problem = _make_problem(BiconvexRelaxProblem)

    value = problem.solve(
        cp.CLARABEL,
        abs_tol=1e6,
        slack_tolerance=1e-4,
        rel_tol=1e-4,
    )

    assert value is not None
    assert problem.status == "converge"
