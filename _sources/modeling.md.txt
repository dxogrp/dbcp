# Problem modeling

## Variable partition

Construct an ordinary CVXPY objective and constraints, then pass exactly two
variable groups to {class}`dbcp.BiconvexProblem` or
{class}`dbcp.BiconvexRelaxProblem`:

```python
x = cp.Variable(n, name="x")
y = cp.Variable(m, name="y")

problem = dbcp.BiconvexProblem(
    cp.Minimize(cp.sum_squares(A @ x + cp.multiply(B @ y, C @ x) - d)),
    [[x], [y]],
    constraints,
)
```

The first group contains variables optimized in the x-subproblem; the second
contains variables optimized in the y-subproblem. Every optimization variable
should belong to exactly one group. DBCP replaces the inactive group's
variables with generated CVXPY parameters when it constructs each fixed
subproblem.

The grouping is part of the model. DBCP does not search for a partition or
verify that the supplied lists are disjoint and exhaustive.

(structural-requirements)=
## Structural requirements

For the supplied partition, both fixed subproblems must satisfy CVXPY's DCP
rules. In particular:

- the objective must be scalar and may be `cp.Minimize` or `cp.Maximize`;
- fixing the second block must leave a convex minimization objective or a
  concave maximization objective in the first block;
- fixing the first block must give the corresponding curvature in the second;
- every inequality must be DCP in either fixed problem; and
- every equality must be affine in either fixed problem.

Call {meth}`dbcp.BiconvexProblem.is_dbcp` after construction to check the two
generated subproblems.

DBCP's constraint transformation currently supports CVXPY equality,
inequality, zero, nonpositive, nonnegative, positive-semidefinite, and
second-order-cone constraints. A different constraint class can raise
`TypeError` while the fixed or relaxed problems are being constructed.

See {doc}`rules` for product compositions and the custom convolution helper.

## Direct and relaxed models

{class}`dbcp.BiconvexProblem` solves the original constraints. If the supplied
initial variable values are infeasible, it first performs an alternating
feasibility search over a slack-relaxed auxiliary problem.

{class}`dbcp.BiconvexRelaxProblem` instead retains slacks during the main solve
and penalizes their total magnitude. This is useful when maintaining exact
feasibility at every alternating step is difficult. Its status distinguishes
solutions that satisfy the original constraints from those with residual
slack.

## Variables, parameters, and initial values

CVXPY variable attributes such as `nonneg=True`, `nonpos=True`, symmetry, and
positive semidefiniteness are copied to the generated fixed parameters. Native
constraints remain part of each applicable subproblem.

Ordinary CVXPY parameters can be used as problem data and retain their normal
CVXPY behavior. Give them numerical values before solving.

An existing variable `.value` is used as an initial point. When a value is
missing, DBCP draws a standard-normal array and projects it through the
variable's CVXPY attributes. Assign all variable values explicitly when
reproducible initialization is important.
