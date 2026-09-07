# Results and statuses

## Numerical result

Both problem classes return their final objective value from `solve()` and
write numerical solutions into the original CVXPY variables:

```python
value = problem.solve()

print(problem.status)
print(problem.value)
print(x.value)
print(y.value)
```

For {class}`dbcp.BiconvexProblem`, `problem.value` is the original objective
evaluated after the final y-subproblem; it excludes the proximal term.

For {class}`dbcp.BiconvexRelaxProblem`, `problem.value` is the original modeled
objective at the final variables. It excludes both the proximal term and the
slack penalty. The relaxed problem itself is available through `rlx_prob` for
advanced inspection.

## Direct-problem statuses

```{list-table}
:header-rows: 1
:widths: 28 72

* - Status
  - Meaning
* - `converge`
  - The absolute gap between the final x- and y-subproblem objective values is
    below `gap_tolerance`.
* - `converge_inaccurate`
  - The solve reached `max_iter` before that objective gap fell below the
    tolerance.
```

## Relaxed-problem statuses

```{list-table}
:header-rows: 1
:widths: 38 62

* - Status
  - Meaning
* - `converge`
  - The slack-penalized subproblem objective gap is within tolerance and total
    slack is below `slack_tolerance`.
* - `converge_infeasible`
  - The slack-penalized subproblem objective gap is within tolerance, but the
    final total slack is not.
* - `converge_inaccurate`
  - The iteration limit was reached before the slack-penalized objective gap
    met its tolerance, but total slack is below its tolerance.
* - `converge_inaccurate_infeasible`
  - The iteration limit was reached before the slack-penalized objective gap
    met its tolerance, and total slack remains above its tolerance.
```

A relaxed status containing `infeasible` also emits a warning with the final
constraint-violation total. Increasing `nu`, improving the initial values, or
using another convex solver can produce a more feasible point.

See {doc}`solving` for the ACS algorithm and the interpretation of its
objective-gap stopping test.
