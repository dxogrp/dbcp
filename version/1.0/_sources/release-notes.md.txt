# Release notes

## 1.0

- Changed the `BiconvexProblem` constructor to accept `x_var` and `y_var` as
  separate arguments. The constructor materializes the two iterables, verifies
  that every member is a problem variable, and rejects overlapping groups. The
  groups may still be empty or non-exhaustive. Replace a grouped call such as
  `BiconvexProblem(objective, ([x], [y]), constraints)` with
  `BiconvexProblem(objective, [x], [y], constraints)`.

- Unified direct and slack-penalized solves under `BiconvexProblem.solve()`.
  The default `mode="direct"` preserves the direct algorithm, while
  `mode="penalty"` selects the slack-penalized algorithm and permits an
  infeasible starting point. Omitting the penalty-only `nu` and `slack_tol`
  options, or explicitly passing `None`, gives effective defaults of `1` and
  `1e-6`, respectively. `nu` must be finite and strictly positive; `nu=0` is
  now rejected. A non-`None` value for either penalty-only option is also
  rejected in direct mode. The `method` keyword is available for CVXPY and
  passes through to each subproblem solve; replace the former DBCP selector
  `method="penalty"` with `mode="penalty"`.
  The old `slack_tolerance` keyword is rejected with this error:
  `'slack_tolerance' was removed; use 'slack_tol'.`

  Penalty results whose total slack exceeds `slack_tol` now use the statuses
  `converge_with_slack` and `converge_inaccurate_with_slack`, replacing
  `converge_infeasible` and `converge_inaccurate_infeasible`. Equality with
  `slack_tol` is accepted.

  `BiconvexRelaxProblem` has been removed from the public API. Constructing the
  migration stub raises this error: `'BiconvexRelaxProblem' was removed; use
  'BiconvexProblem' and call solve(mode='penalty', ...).` Repeated calls keep
  the current variable values, so callers can explicitly retry either mode or
  switch modes on the same object.

- Added penalty-formulation inspection to `BiconvexProblem`. The existing
  `x_prob` and `y_prob` properties continue to expose the direct fixed
  subproblems. The new `penalty_prob`, `penalty_x_prob`, `penalty_y_prob`, and
  `slack_vars` properties lazily construct and then return stable penalty
  problem and slack-variable objects.

- Replaced the public `gap_tolerance` solve argument with `abs_tol` and the
  keyword-only `rel_tol`. Both default to $10^{-6}$, and DBCP stops when

  $$
  |u-v| \leq \epsilon_{\mathrm{abs}}
    + \epsilon_{\mathrm{rel}}\max\{|u|,|v|\}.
  $$

  The legacy `gap_tolerance` keyword is rejected with guidance to use the new
  arguments. Setting `rel_tol=0` yields an absolute-only stopping criterion.

## 0.2

The prototype series provided `BiconvexProblem` for direct alternating solves,
`BiconvexRelaxProblem` for slack-relaxed solves, and `convolve` for biconvex
convolution expressions.
