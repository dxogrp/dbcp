# DBCP: Disciplined Biconvex Programming in Python

DBCP is a [CVXPY](https://www.cvxpy.org/) extension for modeling and
approximately solving biconvex optimization problems. A two-block biconvex
problem has the form

$$
\begin{array}{ll}
\text{minimize} & f_0(x,y) \\
\text{subject to} & f_i(x,y) \leq 0, \quad i=1,\ldots,m, \\
& h_i(x,y)=0, \quad i=1,\ldots,p,
\end{array}
$$

where $x\in X$ and $y\in Y$ are the two variable blocks. With $y$ fixed,
the objective and inequality functions are convex in $x$ and the equality
functions are affine in $x$. The same conditions hold in $y$ when $x$ is
fixed.

## Disciplined biconvex programming

DBCP extends CVXPY's disciplined convex programming rules with structured
products between expressions from the two variable blocks. A model is accepted
when fixing either supplied block produces a DCP-compliant CVXPY problem.

DBCP solves accepted models using proximal alternating convex search. Each
iteration solves one convex subproblem with the other block fixed. This is a
local heuristic for a generally difficult nonconvex problem: convergence does
not certify a global optimum.

Use {class}`dbcp.BiconvexProblem` when the original constraints should be
satisfied throughout the alternating solve. Use
{class}`dbcp.BiconvexRelaxProblem` to introduce and penalize constraint slacks,
which permits infeasible iterates and reports whether the final point is
feasible for the original model.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
modeling
rules
solving
results
examples
troubleshooting
api
release-notes
cite
```
