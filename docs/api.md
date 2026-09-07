# API reference

The symbols below are the complete public namespace exported by `dbcp`
{sub-ref}`release`.

## Modeling and solving

```{eval-rst}
.. autoclass:: dbcp.BiconvexProblem
   :members: x_prob, y_prob, penalty_prob, penalty_x_prob, penalty_y_prob, slack_vars, is_dbcp, solve, status, value
   :member-order: bysource
```

## Expressions

```{eval-rst}
.. autofunction:: dbcp.convolve
```

## Package version

```{eval-rst}
.. autodata:: dbcp.__version__
```
