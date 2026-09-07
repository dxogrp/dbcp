# Troubleshooting

## `is_dbcp()` returns false

Inspect `problem.x_prob` and `problem.y_prob` separately with CVXPY's
`is_dcp()` diagnostics. One of the problems still contains a non-DCP
composition after the opposite variable block has been replaced by
parameters. Check that the two supplied groups are disjoint and that every
variable omitted from both can remain active in both subproblems. The groups
need not be exhaustive. Also check that CVXPY knows any signs required by the
composition rules.

The constructor itself can raise `TypeError` for a specialized constraint
class that DBCP does not know how to copy. See {doc}`rules` for the supported
families.

## Feasible initialization fails

A direct solve can raise `dbcp.error.InitiationError` when its alternating
slack-minimization phase cannot find a point satisfying the original
constraints. Assign feasible or nearly feasible values to the model variables,
try another random seed, or use {class}`dbcp.BiconvexRelaxProblem` when a
penalized infeasible start is appropriate.

## A convex subproblem fails

`dbcp.error.SolveError` means a fixed convex subproblem failed or returned a
status other than CVXPY's `optimal` or `optimal_inaccurate`. Try another
compatible CVXPY solver, improve scaling or initial values, or increase `lbd`
to strengthen the proximal regularization.

## The solve reaches the iteration limit

`converge_inaccurate` means the objective-gap stopping test was not met within
`max_iter`. Inspect the variable values and objective, then consider another
starting point, a larger iteration budget, or a different `lbd`. If the
objective is near zero, adjust `abs_tol`; if its magnitude is large,
adjust `rel_tol`. Calling `solve()` again continues from the current
variable values.

## A relaxed solution remains infeasible

Statuses containing `infeasible` mean the total absolute slack is at least
`slack_tolerance`. Increase `nu` to penalize slack more heavily, initialize
closer to the original feasible set, or reconsider whether the constraints are
jointly feasible.

## CVXPY reports a non-DPP or canonicalization-backend warning

DBCP requires each fixed problem to be DCP, but it does not require the
parameterized problem to be DPP. CVXPY may therefore warn that a parameterized
subproblem will be canonicalized again or that it selected the SciPy
canonicalization backend. These warnings concern compilation and performance;
they do not by themselves mean that `is_dbcp()` should be false. Prefer an
equivalent DPP formulation when one exists, and select a canonicalization
backend explicitly only when that choice is intentional.

## Convolution rejects its inputs

{func}`dbcp.convolve` accepts only one-dimensional CVXPY expressions. Reshape
vectors explicitly before calling it; matrices and higher-dimensional arrays
raise `ValueError`.
