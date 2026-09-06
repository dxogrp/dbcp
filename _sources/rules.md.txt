# DBCP rules and supported expressions

DBCP inherits CVXPY's expression syntax and DCP rules within each fixed
subproblem. Its additional modeling pattern is multiplication between
expressions that depend on different variable blocks.

## Product compositions

The DBCP syntax described by the accompanying paper permits the following
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

The formal DBCP syntax also requires an acyclic variable-interaction graph,
where an edge joins variables that occur on opposite sides of one of these
products. In the current package, the user supplies the two-block partition
directly and {meth}`dbcp.BiconvexProblem.is_dbcp` operationally checks only
that both resulting fixed problems are DCP. It does not independently audit
the interaction graph.

## CVXPY atoms and constraints

There is no separate allowlist of scalar CVXPY atoms. An expression is usable
when copying it with either variable block replaced by parameters succeeds and
both resulting problems satisfy DCP. DPP is not required, although a
non-DPP parameterization can make CVXPY canonicalize a subproblem again on
each solve.

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
