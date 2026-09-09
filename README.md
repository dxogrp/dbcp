# DBCP: Disciplined Biconvex Programming

[![CI](https://github.com/dxogrp/dbcp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dxogrp/dbcp/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/dbcp.svg)](https://pypi.org/project/dbcp/) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dxogrp.github.io/dbcp/) [![License](https://img.shields.io/github/license/dxogrp/dbcp.svg)](https://github.com/dxogrp/dbcp/blob/main/LICENSE)

- [DBCP: Disciplined Biconvex Programming](#dbcp-disciplined-biconvex-programming)
  - [Basic idea](#basic-idea)
  - [Installation](#installation)
    - [PyPI](#pypi)
    - [Development setup](#development-setup)
  - [Quick start](#quick-start)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [License](#license)
  - [Citing](#citing)

DBCP is a [CVXPY](https://www.cvxpy.org/) extension for modeling and
approximately solving biconvex optimization problems of the form

$$
\begin{array}{ll}
    \text{minimize} & f_0(x,y) \\
    \text{subject to} & f_i(x,y) \leq 0, \quad i=1,\ldots,m \\
    & h_i(x,y)=0, \quad i=1,\ldots,p,
\end{array}
$$

where $x \in \mathcal{X}$ and $y \in \mathcal{Y}$ are two variable blocks.
With either block fixed, the objective and inequality functions are convex in
the other block, and the equality functions are affine in the other block.
The theoretical and technical details are described in the
[accompanying paper](https://haozhu10015.github.io/papers/dbcp.html).

## Basic idea

DBCP extends CVXPY's disciplined convex programming rules with structured
products between expressions from the two variable blocks. A model is accepted
when fixing either supplied block produces a DCP-compliant CVXPY problem.

DBCP solves accepted models with proximal alternating convex search.
`BiconvexProblem.solve()` uses the original constraints by default; its
`mode="penalty"` option instead introduces and penalizes constraint slacks to
permit infeasible iterates. The [user guide](https://dxogrp.github.io/dbcp/)
describes the modeling rules, solution methods, and result statuses in detail.

## Installation

### PyPI

DBCP requires Python 3.12 or newer, CVXPY 1.9 or newer, and NumPy 2.3.3 or
newer. Install it from PyPI with:

```shell
pip install dbcp
```

### Development setup

DBCP manages its development environment with [uv](https://docs.astral.sh/uv/).
After installing uv, clone the repository and install the locked development
dependencies:

```shell
git clone https://github.com/dxogrp/dbcp.git
cd dbcp
make sync
```

## Quick start

This example factors a nonnegative matrix $A\in\mathbf{R}^{m\times n}$ as
$XY$, where $X\in\mathbf{R}^{m\times k}$ and
$Y\in\mathbf{R}^{k\times n}$:

$$
\begin{array}{ll}
\text{minimize} & \|XY-A\|_F^2 \\
\text{subject to} & X_{ij} \geq 0,\quad i=1,\ldots,m,\quad j=1,\ldots,k\\
& Y_{ij} \geq 0,\quad i=1,\ldots,k,\quad j=1,\ldots,n.
\end{array}
$$

The objective is convex in $X$ for fixed $Y$ and convex in $Y$ for fixed
$X$.

```python
import cvxpy as cp
import numpy as np

import dbcp

rng = np.random.default_rng(10015)
m, n, k = 5, 10, 3
A = rng.random((m, k)) @ rng.random((k, n))

X = cp.Variable((m, k), name="X")
Y = cp.Variable((k, n), name="Y")

problem = dbcp.BiconvexProblem(
    cp.Minimize(cp.sum_squares(X @ Y - A)),
    [X],
    [Y],
    [X >= 0, Y >= 0],
)

assert problem.is_dbcp()
value = problem.solve()
```

The `[X]` and `[Y]` arguments supply the `x_var` and `y_var` variable groups,
while the last argument encodes the nonnegativity constraints. DBCP alternately
optimizes one group while holding the other fixed and writes the result into
the original CVXPY variables.

Because unset variables are initialized randomly, different starting points
can produce different factorizations. Assign `X.value` and `Y.value` before
`solve()` when a specific warm start is desired.

## Documentation

The complete user guide and API reference are available in the
[published documentation](https://dxogrp.github.io/dbcp/). To build and preview
the documentation locally, run:

```shell
make docs
```

## Examples

The [`examples`](examples) directory contains seven
[Marimo](https://marimo.io/) notebooks demonstrating DBCP. Run

```shell
make marimo
```

to install Marimo and open the notebooks in your browser. Executed,
non-interactive versions are available in the
[published example gallery](https://dxogrp.github.io/dbcp/examples.html).

## License

DBCP is licensed under the [Apache License 2.0](LICENSE).

## Citing

If you find DBCP useful in your research, please consider [citing our paper](https://dxogrp.github.io/dbcp/cite.html).
