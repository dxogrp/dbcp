# DBCP: Disciplined Biconvex Programming in Python

DBCP is a [CVXPY](https://www.cvxpy.org/) extension for modeling and
approximately solving biconvex optimization problems. A biconvex
problem has the form

$$
\begin{array}{ll}
\text{minimize} & f_0(x,y) \\
\text{subject to} & f_i(x,y) \leq 0, \quad i=1,\ldots,m\\
& h_i(x,y)=0, \quad i=1,\ldots,p,
\end{array}
$$

where $x\in \mathcal{X}$ and $y\in \mathcal{Y}$ are the two variable blocks. With $y$ fixed,
the objective and inequality constraint functions are convex in $x$ and
the equality constraint functions are affine in $x$. The same conditions
hold in $y$ when $x$ is fixed.

DBCP extends CVXPY's disciplined convex programming rules with structured
products between expressions from the two variable blocks. A model is accepted
when fixing either supplied block produces a [DCP](https://www.cvxpy.org/tutorial/dcp/index.html)-compliant CVXPY problem.
See the {ref}`DBCP modeling rules <dbcp-rules>` for the complete requirements.

DBCP solves accepted models using proximal alternating convex search. Each
iteration solves one convex subproblem with the other block fixed. See
{doc}`solving` for the algorithm and the interpretation of its stopping test.

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
modeling
solving
results
examples
troubleshooting
api
release-notes
cite
```
