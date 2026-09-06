# Installation

## PyPI

DBCP requires Python 3.12 or newer, CVXPY 1.9 or newer, and NumPy 2.3.3 or
newer. Install it from PyPI with:

```shell
pip install dbcp
```

Confirm the installation and its version:

```shell
python -c "import dbcp; print(dbcp.__version__)"
```

CVXPY supplies the convex solvers used by DBCP. The default is SCS; any
compatible solver installed in the same environment can be selected when
calling `solve()`.

## Development setup

DBCP manages its development environment with [uv](https://docs.astral.sh/uv/).
Install uv before setting up the repository.

1. Clone the repository:

   ```shell
   git clone https://github.com/dxogrp/dbcp.git
   cd dbcp
   ```

2. Create the virtual environment and install the locked development
   dependencies:

   ```shell
   make sync
   ```

Run the primary contributor checks with:

```shell
make test
make lint
```

The example and documentation toolchains use separate locked dependency
groups. Install them only when needed:

```shell
make sync-examples  # Marimo, Matplotlib, and scikit-learn
make sync-docs      # Sphinx and MyST
```

For dependency updates, edit `pyproject.toml`, run `uv lock`, and commit the
resulting `uv.lock` change.
