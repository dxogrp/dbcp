# Release notes

## 1.0

### 1.0.0

- Replaced the public `gap_tolerance` solve argument with `abs_tol` and the
  keyword-only `rel_tol`. Both default to $10^{-6}$, and DBCP stops when

  $$
  |u-v| \leq \epsilon_{\mathrm{abs}}
    + \epsilon_{\mathrm{rel}}\max\{|u|,|v|\}.
  $$

  The legacy `gap_tolerance` keyword is rejected with guidance to use the new
  arguments. Setting `rel_tol=0` yields an absolute-only stopping criterion.

## 0.2

The prototype series provided {class}`dbcp.BiconvexProblem` for direct
alternating solves, {class}`dbcp.BiconvexRelaxProblem` for slack-relaxed
solves, and {func}`dbcp.convolve` for biconvex convolution expressions.
