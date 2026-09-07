# Examples

The gallery consists of standalone [Marimo](https://marimo.io/) notebooks.
Each link opens an executed, non-interactive HTML snapshot containing the
notebook code and outputs.

- {example}`Nonnegative matrix factorization <nonneg_matrix_fact>`
  introduces the two-block model and proximal alternating solve.
- {example}`Bilinear logistic regression <bilin_logi_reg>`
  fits a low-rank bilinear classifier with a maximization objective.
- {example}`Blind deconvolution <blind_deconv>`
  recovers a sparse signal and smooth kernel using {func}`dbcp.convolve`.
- {example}`Sparse dictionary learning <dict_learning>`
  explores the tradeoff between reconstruction error and code sparsity.
- {example}`Input-output hidden Markov model <iohmm>`
  uses {class}`dbcp.BiconvexRelaxProblem` for a constrained latent-state fit.
- {example}`k-means clustering <kmeans>`
  models cluster centers and soft assignment weights as separate blocks.
- {example}`Constrained k-means clustering <kmeans_constr>`
  compares unconstrained centers with centers restricted to prescribed balls.

For live interaction, install and open the gallery from a repository checkout:

```shell
make sync-examples
make marimo
```
