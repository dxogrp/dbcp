# DBCP rules and supported expressions

DBCP inherits CVXPY's expression syntax and DCP rules within each fixed
subproblem. Its additional modeling pattern is multiplication between
expressions that depend on different variable blocks.

## Product compositions

The DBCP syntax described by the [accompanying paper](https://haozhu10015.github.io/papers/dbcp.html) permits the following
curvature and sign combinations across a product:

```{list-table}
:header-rows: 1
:widths: 42 42 16

* - First factor
  - Second factor
  - Product role
* - Affine
  - Affine
  - Biaffine
* - Nonnegative affine
  - Convex
  - Biconvex
* - Nonpositive affine
  - Concave
  - Biconvex
* - Nonnegative convex
  - Nonnegative convex
  - Biconvex
* - Nonpositive concave
  - Nonpositive concave
  - Biconvex
```

The factors may be exchanged. Their signs and curvatures must be known to
CVXPY; numerical values alone do not establish a symbolic sign.

The formal DBCP product rule requires one consistent assignment of the
relevant variables to two blocks across the objective and all constraints. In
every product whose two factors both contain variables, one factor may contain
variables only from one block and the other only from the other block. This
prevents expressions like `x * y`, `y * z`, and `z * x` appearing sumultaneously
in the same problem.

In the package, the user supplies the two disjoint variable groups fixed on
alternating ACS steps. Variables may be omitted from both groups and remain
active in both subproblems, provided both subproblems are DCP.
{meth}`dbcp.BiconvexProblem.is_dbcp` checks that operational condition; it does
not separately traverse products to verify the formal assignment rule.

## CVXPY atoms and constraints

An CVXPY expression is usable when copying it with either variable block
replaced by parameters succeeds and both resulting problems satisfy DCP.
DPP is not required, although a non-DPP parameterization can make CVXPY
canonicalize a subproblem again on each solve.

The supported constraint families are:

- equality and zero constraints;
- scalar or elementwise inequalities, including nonnegative and nonpositive
  constraints;
- positive-semidefinite constraints; and
- second-order-cone constraints.

Other specialized CVXPY constraint objects are not currently transformed by
DBCP.

## Discrete convolution

{func}`dbcp.convolve` constructs the full discrete convolution of two
one-dimensional CVXPY expressions. For lengths $m$ and $n$, it returns an
expression of length $m+n-1$ with

$$
c_k=\sum_{i+j=k}x_i y_j.
$$

When `x` and `y` belong to different variable blocks, the result is biaffine
and is useful in models such as blind deconvolution. Both inputs must be
one-dimensional; otherwise the helper raises `ValueError`.
