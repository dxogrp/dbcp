# Solving

## Proximal alternating convex search

Let $x$ and $y$ denote the two variable blocks. Given the current point
$(x^{(k)},y^{(k)})$, DBCP alternates between convex subproblems of the schematic form

$$
x^{(k+1)}\in\mathop{\rm argmin}_x
f_0(x,y^{(k)})+\lambda\|x-x^{(k)}\|_F^2
$$

and

$$
y^{(k+1)}\in\mathop{\rm argmin}_y
f_0(x^{(k+1)},y)+\lambda\|y-y^{(k)}\|_F^2,
$$

with the analogous sign adjustment for a maximization objective. The
regularization parameter is the `lbd` solve argument. The proximal terms make
each half-step prefer points near the preceding iterate.

After each pair of solves, DBCP evaluates the two fixed-subproblem objectives
without their proximal terms. For a direct problem, these are the original
objective; for a relaxed problem, they also contain the weighted slack penalty.
Let these objective values be $u$ and $v$. DBCP stops when

$$
|u-v| \leq \epsilon_{\mathrm{abs}}
  + \epsilon_{\mathrm{rel}}\max\{|u|,|v|\},
$$

or after `max_iter` iterations. The solve arguments `abs_tol` and
`rel_tol` specify $\epsilon_{\mathrm{abs}}$ and
$\epsilon_{\mathrm{rel}}$, respectively.

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
`abs_tol=rel_tol=1e-6`. `abs_tol` may be passed
positionally, whereas `rel_tol` is keyword-only. Setting the latter
to zero gives an absolute-only stopping criterion. Additional positional and
keyword arguments are passed to CVXPY's `Problem.solve()` calls.

For a relaxed problem, `nu=1` and `slack_tolerance=1e-6` by default.

Note that alternating convex search is a local heuristic.
Its objective-gap stopping test, including a `converge`
status, is **not** a certificate of global optimality.

## Feasible initialization

Before a direct {class}`dbcp.BiconvexProblem` solve, every unset variable is
randomly initialized and projected through its CVXPY attributes. If all
constraints are satisfied, ACS starts immediately.

Otherwise, DBCP adds temporary slacks to the constraints and alternately
minimizes their total one-norm until the original constraints are satisfied.
If this search cannot find a feasible point within its internal iteration
limit, it raises `dbcp.error.InitiationError`.

Supplying feasible `.value` arrays before solving avoids random initialization
and can improve repeatability.

## Infeasible-start relaxation

{class}`dbcp.BiconvexRelaxProblem` introduces nonnegative slacks $s$ for
inequality constraints and unrestricted slacks $t$ for equality constraints.
For minimization, it solves a penalized objective of the form

$$
f_0(x,y)+\nu\left(\mathbf{1}^T s+\lVert t\rVert_1\right),
$$

while maximization subtracts the same penalty. Set `nu` in `solve()`; a larger
value places more emphasis on satisfying the original constraints.

```python
value = problem.solve(
    solver=cp.CLARABEL,
    nu=1e3,
    lbd=0.1,
    abs_tol=1e-5,
    slack_tolerance=1e-7,
)
```

The final sum of absolute slack values is compared with `slack_tolerance` to
classify the point as feasible or infeasible. See {doc}`results` for every
status.

## Continuing from an existing point

The original CVXPY variables retain their numerical values after a solve.
Calling `solve()` again therefore starts from the current point. This can be
useful after an inaccurate termination or after changing an ordinary CVXPY
parameter, the iteration limit, the proximal weight, or the slack penalty.

DBCP prints initialization and iteration progress to standard output. Solver
verbosity and other backend-specific options can be passed through CVXPY's
keyword arguments.
