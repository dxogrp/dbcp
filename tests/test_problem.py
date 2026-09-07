import inspect
import warnings

import cvxpy as cp
import numpy as np
import pytest

import dbcp
import dbcp.problem as problem_module
from dbcp import BiconvexProblem
from dbcp.error import InitiationError
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


def _make_problem():
    x = cp.Variable()
    y = cp.Variable()
    x.value = 0.0
    y.value = 0.0
    return BiconvexProblem(
        cp.Minimize(cp.square(x - 1) + cp.square(y - 2)),
        [x],
        [y],
        [x >= 0, y >= 0],
    )


def _make_unconstrained_problem():
    x = cp.Variable()
    y = cp.Variable()
    x.value = 0.0
    y.value = 0.0
    return BiconvexProblem(
        cp.Minimize(cp.square(x - 1) + cp.square(y - 2)),
        [x],
        [y],
    )


def _make_inconsistent_problem():
    x = cp.Variable()
    y = cp.Variable()
    x.value = 0.0
    y.value = 0.0
    objective_expression = cp.square(x - 2) + cp.square(y - 3)
    problem = BiconvexProblem(
        cp.Minimize(objective_expression),
        [x],
        [y],
        [x >= x + 1],
    )
    return problem, x, y, objective_expression


@pytest.mark.parametrize("tolerance_name", ["abs_tol", "rel_tol"])
@pytest.mark.parametrize("invalid_value", [-1e-6, np.nan, np.inf, -np.inf])
def test_solve_rejects_invalid_convergence_tolerances(
    tolerance_name,
    invalid_value,
):
    problem = _make_problem()

    with pytest.raises(ValueError, match=rf"{tolerance_name} must be finite and nonnegative"):
        problem.solve(**{tolerance_name: invalid_value})


def test_solve_rejects_legacy_gap_tolerance():
    problem = _make_problem()

    with pytest.raises(TypeError, match="gap_tolerance.*abs_tol.*rel_tol"):
        problem.solve(gap_tolerance=1e-6)


def test_solve_signature():
    parameters = inspect.signature(BiconvexProblem.solve).parameters

    assert parameters["abs_tol"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["abs_tol"].default == 1e-6
    assert parameters["rel_tol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["rel_tol"].default == 1e-6
    assert parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["mode"].default == "direct"
    assert parameters["nu"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["nu"].default is None
    assert parameters["slack_tol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["slack_tol"].default is None


def test_constructor_signature():
    parameters = inspect.signature(BiconvexProblem).parameters

    assert list(parameters) == ["biconvex_objective", "x_var", "y_var", "constraints"]
    assert parameters["x_var"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["y_var"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["x_var"].default is inspect.Parameter.empty
    assert parameters["y_var"].default is inspect.Parameter.empty


def test_constructor_accepts_keyword_arguments():
    x = cp.Variable()
    y = cp.Variable()

    problem = BiconvexProblem(
        biconvex_objective=cp.Minimize(x * y),
        x_var=[x],
        y_var=[y],
        constraints=[],
    )

    assert problem.is_dbcp()


def test_constructor_materializes_groups_and_preserves_direction():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")
    problem = BiconvexProblem(cp.Minimize(x * y + cp.square(z)), iter([x]), (y,))

    assert problem.fix_vars == ((x,), (y,))
    assert {variable.id for variable in problem.x_prob.variables()} == {x.id, z.id}
    assert {variable.id for variable in problem.y_prob.variables()} == {y.id, z.id}
    assert problem.is_dbcp()


def test_constructor_accepts_empty_and_nonexhaustive_groups():
    x = cp.Variable()
    y = cp.Variable()

    problem = BiconvexProblem(cp.Minimize(cp.square(x) + cp.square(y)), [], [y])

    assert problem.fix_vars == ((), (y,))


@pytest.mark.parametrize("name", ["x_var", "y_var"])
def test_constructor_rejects_a_single_variable(name):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    arguments = {"x_var": [x], "y_var": [y]}
    arguments[name] = x if name == "x_var" else y

    with pytest.raises(TypeError, match=rf"{name} must be an iterable"):
        BiconvexProblem(cp.Minimize(cp.square(x) + cp.square(y)), **arguments)


@pytest.mark.parametrize("group_container", [list, tuple])
@pytest.mark.parametrize(
    "legacy_form",
    [
        "positional",
        "positional-with-constraints",
        "positional-with-keyword-constraints",
        "keyword",
        "keyword-with-constraints",
    ],
)
def test_constructor_rejects_every_grouped_legacy_form_with_migration_guidance(
    group_container,
    legacy_form,
):
    x = cp.Variable()
    y = cp.Variable()
    objective = cp.Minimize(x * y)
    grouped = group_container(([x], [y]))

    with pytest.raises(TypeError, match="grouped 'fix_vars'.*x_var and y_var"):
        if legacy_form == "positional":
            BiconvexProblem(objective, grouped)
        elif legacy_form == "positional-with-constraints":
            BiconvexProblem(objective, grouped, [])
        elif legacy_form == "positional-with-keyword-constraints":
            BiconvexProblem(objective, grouped, constraints=[])
        elif legacy_form == "keyword":
            BiconvexProblem(biconvex_objective=objective, fix_vars=grouped)
        else:
            BiconvexProblem(
                biconvex_objective=objective,
                fix_vars=grouped,
                constraints=[],
            )


def test_constructor_rejects_an_invalid_new_group_without_migration_guidance():
    x = cp.Variable()
    y = cp.Variable()

    with pytest.raises(TypeError, match="^x_var must contain only CVXPY Variables\\.$") as error:
        BiconvexProblem(cp.Minimize(x * y), [x, 1], [y])

    assert "fix_vars" not in str(error.value)


def test_constructor_requires_a_separate_y_group():
    x = cp.Variable()

    with pytest.raises(TypeError, match="^y_var is required; pass x_var and y_var as separate arguments\\.$"):
        BiconvexProblem(cp.Minimize(cp.square(x)), [x])


def test_constructor_rejects_variables_absent_from_problem():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    with pytest.raises(ValueError, match="Every variable in y_var must appear.*y"):
        BiconvexProblem(cp.Minimize(cp.square(x)), [x], [y])


def test_constructor_rejects_overlapping_groups():
    x = cp.Variable()
    y = cp.Variable()

    with pytest.raises(ValueError, match="x_var and y_var must be disjoint"):
        BiconvexProblem(cp.Minimize(x * y), [x], [x, y])


def test_retired_problem_class_raises_migration_error_and_is_not_public():
    assert "BiconvexRelaxProblem" not in dbcp.__all__

    with pytest.raises(
        TypeError,
        match=(
            "'BiconvexRelaxProblem' was removed; use 'BiconvexProblem' and call solve\\(mode='penalty', \\.\\.\\.\\)\\."
        ),
    ):
        dbcp.BiconvexRelaxProblem(None, None)


def test_biconvex_problem_solves_with_scale_aware_tolerances():
    problem = _make_problem()

    value = problem.solve(
        cp.CLARABEL,
        abs_tol=1e6,
        rel_tol=1e-4,
    )

    assert value is not None
    assert problem.status == "converge"


def test_direct_initialization_preserves_problem_parameters():
    lower_bound = cp.Parameter(value=1.0)
    x = cp.Variable()
    y = cp.Variable()
    x.value = 0.0
    y.value = 0.0
    problem = BiconvexProblem(
        cp.Minimize(cp.square(x) + cp.square(y)),
        [x],
        [y],
        [x >= lower_bound],
    )

    problem.solve(cp.CLARABEL, abs_tol=1e6)

    assert x.value >= 1 - 1e-6


def test_penalty_inspection_properties_are_lazy_and_stable(monkeypatch):
    calls = 0
    relax_with_slack = problem_module.relax_with_slack

    def counting_relax_with_slack(*args, **kwargs):
        nonlocal calls
        calls += 1
        return relax_with_slack(*args, **kwargs)

    monkeypatch.setattr(problem_module, "relax_with_slack", counting_relax_with_slack)
    problem = _make_problem()
    direct_x_prob = problem.x_prob
    direct_y_prob = problem.y_prob

    assert calls == 0
    penalty_prob = problem.penalty_prob
    penalty_x_prob = problem.penalty_x_prob
    penalty_y_prob = problem.penalty_y_prob
    slack_vars = problem.slack_vars

    assert calls == 1
    assert penalty_prob is problem.penalty_prob
    assert penalty_x_prob is problem.penalty_x_prob
    assert penalty_y_prob is problem.penalty_y_prob
    assert slack_vars is problem.slack_vars
    assert slack_vars
    assert problem.x_prob is direct_x_prob
    assert problem.y_prob is direct_y_prob

    value = problem.solve(
        cp.CLARABEL,
        abs_tol=1e6,
        rel_tol=1e-4,
        mode="penalty",
    )

    assert calls == 1
    assert value is not None
    assert problem.status == "converge"
    assert value == problem.objective.value
    assert len(penalty_prob.parameters()) == 1
    assert penalty_prob.parameters()[0].value == 1
    assert penalty_prob is problem.penalty_prob
    assert penalty_x_prob is problem.penalty_x_prob
    assert penalty_y_prob is problem.penalty_y_prob
    assert slack_vars is problem.slack_vars


@pytest.mark.parametrize(
    ("positive_slack", "gap_within_tolerance", "expected_status"),
    [
        pytest.param(False, True, "converge", id="zero-slack-converged"),
        pytest.param(False, False, "converge_inaccurate", id="zero-slack-max-iterations"),
        pytest.param(True, True, "converge_infeasible", id="positive-slack-converged"),
        pytest.param(
            True,
            False,
            "converge_inaccurate_infeasible",
            id="positive-slack-max-iterations",
        ),
    ],
)
def test_penalty_status_uses_inclusive_zero_slack_boundary(
    monkeypatch,
    positive_slack,
    gap_within_tolerance,
    expected_status,
):
    if positive_slack:
        problem, *_ = _make_inconsistent_problem()
    else:
        problem = _make_unconstrained_problem()
    monkeypatch.setattr(
        "dbcp.problem._objective_gap_within_tolerance",
        lambda *_args: gap_within_tolerance,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        problem.solve(
            cp.CLARABEL,
            max_iter=1,
            mode="penalty",
            slack_tol=0.0,
        )

    total_slack = sum(float(np.sum(np.abs(slack.value))) for slack in problem.slack_vars)
    infeasibility_warnings = [
        warning for warning in caught_warnings if "returned solution is infeasible" in str(warning.message)
    ]
    if positive_slack:
        assert total_slack > 0
        assert len(infeasibility_warnings) == 1
        assert ". Consider increasing 'nu'" in str(infeasibility_warnings[0].message)
    else:
        assert total_slack == 0
    assert problem.status == expected_status
    assert bool(infeasibility_warnings) is ("infeasible" in expected_status)


def test_penalty_mode_returns_original_objective_only():
    problem, _x, y, objective_expression = _make_inconsistent_problem()
    lbd = 1.0
    nu = 4.0
    initial_y = float(y.value)
    penalty_y_prob = problem.penalty_y_prob
    slack_vars = problem.slack_vars

    with pytest.warns(UserWarning, match="returned solution is infeasible"):
        returned_value = problem.solve(
            cp.CLARABEL,
            lbd=lbd,
            abs_tol=1e6,
            mode="penalty",
            nu=nu,
            slack_tol=0.0,
        )

    total_slack = float(np.sum([np.sum(np.abs(slack.value)) for slack in slack_vars]))
    original_value = float(objective_expression.value)
    penalized_value = float(penalty_y_prob.objective.value)
    final_proximal_term = lbd * float(np.square(y.value - initial_y))

    assert total_slack >= 1 - 1e-6
    assert penalized_value - original_value == pytest.approx(nu * total_slack, rel=1e-5)
    assert final_proximal_term > 0
    assert returned_value == pytest.approx(original_value)
    assert returned_value != pytest.approx(penalized_value)
    assert returned_value != pytest.approx(penalized_value + final_proximal_term)


def test_failed_direct_solve_can_retry_with_penalty_mode():
    problem, _x, _y, _objective_expression = _make_inconsistent_problem()

    with pytest.raises(InitiationError):
        problem.solve(cp.CLARABEL, proj_max_iter=1)

    assert problem.status is None
    assert problem.value is None
    with pytest.warns(UserWarning, match="returned solution is infeasible"):
        value = problem.solve(
            cp.CLARABEL,
            abs_tol=1e6,
            mode="penalty",
            slack_tol=0.0,
            proj_max_iter=1,
        )

    assert value is not None
    assert problem.status == "converge_infeasible"


@pytest.mark.parametrize("mode", ["relaxed", "DIRECT", None, 1])
def test_solve_rejects_invalid_modes(mode):
    problem = _make_problem()

    with pytest.raises(ValueError, match="mode must be either 'direct' or 'penalty'"):
        problem.solve(mode=mode)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        pytest.param("nu", 1.0, id="nu"),
        pytest.param("slack_tol", 1e-6, id="slack-tolerance"),
    ],
)
def test_direct_mode_rejects_explicit_penalty_only_options(option, value):
    problem = _make_problem()

    with pytest.raises(ValueError, match="mode='penalty'"):
        problem.solve(mode="direct", **{option: value})


@pytest.mark.parametrize("option", ["nu", "slack_tol"])
@pytest.mark.parametrize("invalid_value", [-1e-6, np.nan, np.inf, -np.inf, "invalid"])
def test_penalty_mode_rejects_invalid_numeric_options(option, invalid_value):
    problem = _make_problem()

    with pytest.raises(ValueError, match=rf"{option} must be finite and nonnegative"):
        problem.solve(mode="penalty", **{option: invalid_value})


def test_solve_rejects_legacy_slack_tolerance():
    problem = _make_problem()

    with pytest.raises(TypeError) as error:
        problem.solve(mode="penalty", slack_tolerance=1e-6)

    assert str(error.value) == "'slack_tolerance' was removed; use 'slack_tol'."


@pytest.mark.parametrize("legacy_method", ["direct", "penalty"])
def test_solve_rejects_legacy_method_selector_with_migration_guidance(legacy_method):
    problem = _make_problem()

    with pytest.raises(TypeError, match="'method' selector was renamed to 'mode'"):
        problem.solve(method=legacy_method)


def test_solve_resets_stale_results_before_a_failed_retry():
    problem = _make_problem()
    problem.solve(cp.CLARABEL, abs_tol=1e6)
    assert problem.status is not None
    assert problem.value is not None

    with pytest.raises(ValueError, match="mode"):
        problem.solve(mode="unknown")

    assert problem.status is None
    assert problem.value is None


def test_problem_can_switch_modes_and_update_penalty():
    problem = _make_problem()
    penalty_problem = problem.penalty_prob
    assert len(penalty_problem.parameters()) == 1
    penalty_parameter = penalty_problem.parameters()[0]

    problem.solve(cp.CLARABEL, abs_tol=1e6)
    problem.solve(cp.CLARABEL, abs_tol=1e6, mode="penalty", nu=2, slack_tol=1e-4)
    assert penalty_parameter.value == 2

    problem.solve(cp.CLARABEL, abs_tol=1e6, mode="penalty", nu=3, slack_tol=1e-4)
    assert problem.penalty_prob is penalty_problem
    assert penalty_parameter.value == 3

    problem.solve(cp.CLARABEL, abs_tol=1e6, mode="direct")
    assert problem.status == "converge"


def test_cvxpy_custom_solve_method_is_forwarded_to_alternating_subproblems(monkeypatch):
    method_name = "dbcp_test_custom_solve"
    calls = []

    def custom_solve(problem, *args, **kwargs):
        calls.append((problem, args, kwargs.copy()))
        return cp.Problem._solve(problem, *args, **kwargs)

    monkeypatch.setitem(cp.Problem.REGISTERED_SOLVE_METHODS, method_name, custom_solve)
    problem = _make_problem()

    problem.solve(
        solver=cp.CLARABEL,
        abs_tol=1e6,
        mode="direct",
        method=method_name,
    )

    assert len(calls) == 2
    assert all(called_problem is not problem for called_problem, _args, _kwargs in calls)
    assert all(args == () for _problem, args, _kwargs in calls)
    assert all(kwargs["solver"] == cp.CLARABEL for _problem, _args, kwargs in calls)
