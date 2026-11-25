# MELBA 0.1.0 algorithm

## Model form

For each training feature column, MELBA subtracts its training mean and divides by its training standard deviation. A zero or near-zero scale is replaced by 1. By default, the feature map contains the standardized terms `z_i`, their squares `z_i²`, and every pairwise product `z_i z_j` for `i < j`. Setting `quadratic_terms=False` keeps only `z_i`.

MELBA then centers and scales each expanded column using training statistics. The prediction in standardized target units is

`ŷ = b + Σ_j w_j φ_j(x)`,

where `φ_j` denotes one mapped feature term. MELBA standardizes the target as `(y - μ_y) / s_y` during training and reverses that scaling for predictions. A constant target uses `s_y = 1`.

This feature map is ordinary second-degree polynomial regression. The implementation does not claim scientific novelty or a new regression family.

## Objective and loss

For standardized target `t_i` and residual `r_i = ŷ_i - t_i`, the Huber loss with threshold `δ > 0` is

`Lδ(r) = 0.5 r²` when `|r| <= δ`, and `δ (|r| - 0.5 δ)` otherwise.

The optimized objective is

`J(b, w) = (1/n) Σ_i Lδ(r_i) + (λ/2) Σ_j w_j²`,

where `λ` is the non-negative L2 penalty. The intercept `b` is not penalized. The quadratic loss near zero gives smooth gradients for ordinary errors; the linear tails reduce the influence of large residuals compared with squared loss.

## Optimization and stopping

MELBA initializes the intercept at zero in standardized target space and the coefficients from a small zero-centered normal distribution controlled by `random_state`. For the augmented design matrix `A` with an intercept column, it uses the full-batch gradient

`∇J = Aᵀ clip(ŷ - t, -δ, δ) / n + λ [0, w]`.

Each update subtracts the gradient multiplied by `learning_rate / B`, where `B = ||A||_F² / n + λ` and `||A||_F` is the Frobenius norm. Since this bounds the largest eigenvalue of `AᵀA/n`, `learning_rate` is restricted to `(0, 1]`. Training stops when the absolute objective change between updates is no greater than `tolerance`, or when `epochs` updates have run. `loss_history_`, `loss_`, `n_iter_`, and `converged_` record the outcome.

The Huber objective with L2 regularization is convex in the mapped coefficients. The polynomial feature map itself does not make predictions linear in the original input columns.

## Prediction and metrics

Prediction applies training-time feature and target scales. `evaluate` reports mean absolute error, mean squared error, root mean squared error, and

`R² = 1 - Σ_i (y_i - ŷ_i)² / Σ_i (y_i - mean(y))²`.

For a constant target, MELBA returns `R² = 1` when predictions exactly match it and `R² = 0` otherwise. This avoids division by zero and follows a common finite-score convention.

The evaluation metrics scale inputs before subtraction when a direct difference would overflow, and scale residuals before squaring. If the final MAE, MSE, RMSE, or R² value is outside the finite `float64` range, the metric raises `OverflowError` instead of returning infinity. RMSE is computed directly so it can remain available when MSE itself is too large to represent.

## Assumptions and limitations

Features and targets must be finite real numeric values. MELBA does not impute missing observations, encode categories, or select features. Quadratic expansion uses O(p²) terms, so it is best suited to datasets with a modest number of columns. Standardization improves numerical conditioning but does not remove the need to inspect units, outliers, data leakage, and train/test separation. Performance must be measured on data that was not used to fit the model.
