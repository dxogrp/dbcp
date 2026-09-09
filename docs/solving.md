# Solving

Let $x$ and $y$ denote the two variable blocks supplied to
{class}`dbcp.BiconvexProblem`, and let $f_0(x,y)$ denote the original modeled
objective. {meth}`dbcp.BiconvexProblem.solve` provides two solution modes. The
default, `mode="direct"`, enforces the original constraints and searches for a
feasible starting point when necessary. `mode="penalty"` instead relaxes the
constraints with penalized slack variables and permits infeasible iterates.
After this mode-specific setup, both modes use proximal alternating convex
search.

## Initial values

Both solve modes retain existing variable `.value` arrays as their initial
point. When a value is missing, DBCP draws a standard-normal array and projects
it through the variable's CVXPY attributes. Assign all variable values
explicitly when reproducible initialization is important.

## Direct mode and feasible initialization

Direct mode is selected with `mode="direct"` or by omitting `mode`. It solves
the original constraints and checks whether the initial point satisfies all of
them. If it does, alternating convex search starts immediately.

Otherwise, DBCP adds temporary slacks to the constraints and alternately
minimizes their total one-norm until the original constraints are satisfied.
If this search cannot find a feasible point within its internal iteration
limit, it raises `dbcp.error.InitiationError`.

The direct-only `proj_max_iter` keyword controls the feasibility search's
iteration limit and defaults to `10`. DBCP consumes this keyword rather than
passing it to CVXPY. It has no effect in penalty mode.

## Penalty mode

Calling {meth}`dbcp.BiconvexProblem.solve` with `mode="penalty"` introduces a
slack variable for each original constraint. Slacks for equality and zero
constraints are unrestricted; slacks for the other {ref}`supported constraint
families <supported-constraints>` are nonnegative. Let $\mathcal{S}$ denote the
collection of generated slack variables. Penalty mode skips feasible
initialization, so the alternating solve may start and remain infeasible for
the original constraints. The total slack is

$$
S = \sum_{s\in\mathcal{S}} \lVert s\rVert_1.
$$

Let $\nu$ denote the `nu` argument, the weight applied to $S$. It must be finite
and strictly positive; its effective default is `1`. For minimization, penalty
mode uses the objective

$$
f_0(x,y)+\nu S,
$$

while maximization subtracts the same penalty. A larger `nu` places more
emphasis on satisfying the original constraints.

The `slack_tol` argument is the finite, nonnegative threshold used to classify
the returned total slack; its effective default is `1e-6`. Omitting `nu` or
`slack_tol`, or explicitly passing `None`, selects its effective default.
These arguments apply only to penalty mode, and direct mode rejects a
non-`None` value for either one.

The following example uses penalty mode for solving a biconvex problem:

```python
value = problem.solve(
    solver=cp.CLARABEL,
    mode="penalty",
    nu=1e3,
    lbd=0.1,
    abs_tol=1e-5,
    slack_tol=1e-7,
)
```

When the final sum of absolute slack values is above `slack_tol`, the status
ends in `_with_slack`; equality with the tolerance is accepted.
See {doc}`results` for all possible statuses.

## Proximal alternating convex search

Let $\lambda$ denote the nonnegative proximal weight supplied through `lbd`.
Given the current point $(x^{(k)},y^{(k)})$, direct mode alternates between
convex subproblems of the schematic form

$$
x^{(k+1)}\in\mathop{\rm argmin}_x
f_0(x,y^{(k)})+\lambda\|x-x^{(k)}\|_F^2
$$

and

$$
y^{(k+1)}\in\mathop{\rm argmin}_y
f_0(x^{(k+1)},y)+\lambda\|y-y^{(k)}\|_F^2,
$$

with the analogous sign adjustment for a maximization objective. The proximal
terms make each half-step prefer points near the preceding iterate. Penalty
mode uses the same alternating structure with its penalized objective and
jointly optimizes the slack variables in each half-step; the slack variables
do not receive proximal terms.

After each pair of solves, DBCP evaluates the two fixed-subproblem objectives
without their proximal terms. Let $u$ and $v$ denote these objective values.
They come from the original objective in direct mode and include the weighted
slack penalty in penalty mode. Let $\epsilon_{\mathrm{abs}}$ and
$\epsilon_{\mathrm{rel}}$ be the absolute and relative tolerances supplied
through `abs_tol` and `rel_tol`, respectively. DBCP stops when

$$
|u-v| \leq \epsilon_{\mathrm{abs}}
  + \epsilon_{\mathrm{rel}}\max\{|u|,|v|\},
$$

or after `max_iter` iterations.

### Arguments shared by both modes

```python
value = problem.solve(
    solver=cp.CLARABEL,
    lbd=0.5,
    max_iter=200,
    abs_tol=1e-7,
    rel_tol=1e-6,
)
```

The default solver is `cp.SCS`, `lbd=0.1`, `max_iter=100`, and
`abs_tol=rel_tol=1e-6`. `abs_tol` may be passed positionally, whereas
`rel_tol` is keyword-only. Setting `rel_tol` to zero gives an absolute-only
stopping criterion. Additional positional and keyword arguments are passed to
each alternating CVXPY subproblem solve, but not to the auxiliary direct-mode
feasibility solves.

Note that alternating convex search is a local heuristic. Its objective-gap
stopping test, including a `converge` status, is **not** a certificate of
global optimality.

## Inspecting generated subproblems

The `x_prob` and `y_prob` properties expose the two direct fixed subproblems.
The `penalty_prob`, `penalty_x_prob`, `penalty_y_prob`, and `slack_vars`
properties lazily construct the penalty formulation, its fixed subproblems,
and its tuple of slack variables. Each property returns the same object on
repeated access. See the {doc}`api` for the complete public interface.

## Continuing from an existing point

The original CVXPY variables retain their numerical values after a solve.
Calling `solve()` again therefore starts from the current point. This can be
useful after an inaccurate termination or after changing an ordinary CVXPY
parameter, the iteration limit, the proximal weight, or the slack penalty. A
later call may also switch between the direct and penalty modes explicitly.

DBCP prints initialization and iteration progress to standard output. Solver
verbosity and other backend-specific options can be passed through CVXPY's
keyword arguments.
