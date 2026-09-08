# Results and statuses

## Numerical result

Both solve modes return their final objective value from `solve()` and write
numerical solutions into the original CVXPY variables:

```python
value = problem.solve()

print(problem.status)
print(problem.value)
print(x.value)
print(y.value)
```

`problem.value` is the original modeled objective evaluated at the final
variables. It excludes the proximal term and, for `mode="penalty"`, the slack
penalty. When the status ends in `_with_slack`, this value is evaluated at a
point whose final total slack exceeds `slack_tol` and should not be interpreted
as the objective value of a certified feasible point.

## Direct-mode statuses

```{list-table}
:header-rows: 1
:widths: 28 72

* - Status
  - Meaning
* - `converge`
  - The gap between the final x- and y-subproblem objective values satisfies
    the combined `abs_tol` and `rel_tol` stopping test.
* - `converge_inaccurate`
  - The solve reached `max_iter` before that objective gap satisfied the
    combined stopping test.
```

## Penalty-mode statuses

```{list-table}
:header-rows: 1
:widths: 38 62

* - Status
  - Meaning
* - `converge`
  - The slack-penalized subproblem objective gap satisfies the combined
    absolute-and-relative stopping test and total slack is at or below
    `slack_tol`.
* - `converge_with_slack`
  - The slack-penalized subproblem objective gap satisfies the combined
    stopping test, but the final total slack is above `slack_tol`.
* - `converge_inaccurate`
  - The iteration limit was reached before the slack-penalized objective gap
    met its combined tolerance, but total slack is at or below `slack_tol`.
* - `converge_inaccurate_with_slack`
  - The iteration limit was reached before the slack-penalized objective gap
    met its combined tolerance, and total slack remains above `slack_tol`.
```

A penalty-mode status ending in `_with_slack` also emits a warning with the
final total slack and `slack_tol`. This status describes the returned relaxed
point; it does not establish that the original problem is infeasible. Increasing
`nu`, improving the initial values, or using another convex solver can reduce
the slack. Conversely, a penalty-mode status without `_with_slack` means only
that the final total slack is at or below `slack_tol`; it does not certify exact
feasibility for the original constraints.

See {doc}`solving` for the ACS algorithm and the interpretation of its
objective-gap stopping test.
