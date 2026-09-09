# Problem modeling

## Variable groups

Construct an ordinary CVXPY objective and constraints, then pass the two
variable groups as separate arguments to {class}`dbcp.BiconvexProblem`:

```python
x = cp.Variable(n, name="x")
y = cp.Variable(m, name="y")

problem = dbcp.BiconvexProblem(
    cp.Minimize(cp.sum_squares(A @ x + cp.multiply(B @ y, C @ x) - d)),
    [x],
    [y],
    constraints,
)
```

The `x_var` argument contains variables optimized only in the x-subproblem;
`y_var` contains variables optimized only in the y-subproblem. Each argument
must be an iterable of CVXPY variables, even when it contains only one
variable. The two groups must be disjoint, but they need not contain every
optimization variable. DBCP replaces the inactive group's variables with
generated CVXPY parameters when it constructs each fixed subproblem. A
variable omitted from both groups remains active in both subproblems, which is
valid when both subproblems remain DCP.

For example, `Z` can remain outside both groups in the following model:

```python
X = cp.Variable((m, k))
Y = cp.Variable((k, n))
Z = cp.Variable((m, n))

problem = dbcp.BiconvexProblem(
    cp.Minimize(cp.norm(X @ Y + Z - A, "fro")),
    [X],
    [Y],
    [cp.norm(Z, "fro") <= 1],
)
```

Fixing either `X` or `Y` makes the product affine in the other factor, while
`Z` remains an ordinary variable in both convex subproblems.

(dbcp-rules)=
## DBCP rules

DBCP inherits CVXPY's expression syntax and DCP rules within each fixed
subproblem. Its additional modeling pattern is multiplication between
expressions that depend on different variable blocks.

(structural-requirements)=
### Structural requirements

For the supplied groups, both fixed subproblems must satisfy CVXPY's DCP
rules. In particular:

- the objective must be scalar and may be `cp.Minimize` or `cp.Maximize`;
- fixing the second block must leave a convex minimization objective or a
  concave maximization objective in the first block;
- fixing the first block must give the corresponding curvature in the second;
- every inequality must be DCP in both fixed problems; and
- every equality must be affine in both fixed problems.

Call {meth}`dbcp.BiconvexProblem.is_dbcp` after construction to check this
operational condition. The method checks the two generated subproblems; it
does not separately traverse products to enforce the formal two-block
assignment rule described below.

(product-compositions)=
### Product compositions

Products with a constant or parameter factor follow the ordinary DCP rules.
When both factors contain variables, the DBCP syntax described by the
[accompanying paper](https://haozhu10015.github.io/papers/dbcp.html) permits
the following curvature and sign combinations:

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
prevents expressions like `x * y`, `y * z`, and `z * x` from appearing
simultaneously in the same problem: the first two products require `x` and `z`
to share a block, while the last requires them to be in different blocks.

The rule does not prohibit every cycle. Compatible even cycles are allowed;
for example, the products `x * y`, `y * z`, `z * w`, and `w * x` admit the
two-block assignment `{x, z}` and `{y, w}`.

(supported-constraints)=
### Supported CVXPY expressions and constraints

DBCP must be able to copy each CVXPY expression while replacing either
variable group with generated parameters. DPP is not required, although a
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

## Variable attributes and parameters

CVXPY variable attributes such as `nonneg=True`, `nonpos=True`, symmetry, and
positive semidefiniteness are copied to the generated fixed parameters. Native
constraints remain part of each applicable subproblem.

Ordinary CVXPY parameters can be used as problem data and retain their normal
CVXPY behavior. Give them numerical values before solving.

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

See {doc}`solving` for initialization, solve modes, and generated-subproblem
inspection.
