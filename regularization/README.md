# Regularization and Prior Models

This directory contains priors for complex-valued phase retrieval. The new
functional API separates loss evaluation, gradient evaluation, proximal
operators, and physical constraint projections.

## Differentiable priors

```matlab
[loss, cache] = regularization_forward(object, "tv", weight=1e-3);
gradient = regularization_backward(cache);
```

Available models are:

| Model | Prior |
| --- | --- |
| `l2` | Object-domain Tikhonov regularization |
| `l1_smooth` | Smooth object-domain sparsity |
| `gradient_l2` | Quadratic first-order smoothness |
| `tv` | Isotropic total variation |
| `gradient_hyperlaplacian` | Nonconvex gradient Lp sparsity |
| `hessian_l2` | Quadratic curvature regularization |
| `hessian_tv` | Edge-aware second-order regularization |

All gradients use the complex Euclidean convention

```text
d loss = real(sum(conj(gradient) .* d object))
```

and periodic boundary conditions.

## Proximal operators

```matlab
sparseObject = regularization_prox(object, "l1", 0.01);
```

Both complex soft thresholding (`l1`) and hard thresholding (`l0`) are
provided.

## Physical constraints

```matlab
object = regularization_project(object, "support", mask=supportMask);
object = regularization_project(object, "magnitude_bounds", ...
    minimum=0, maximum=1);
```

The available projections are `support`, `magnitude_bounds`, `phase_only`,
and `real_nonnegative`.

## Verification

Run `test_regularization_gradients` to compare every analytic gradient with
a centered finite-difference directional derivative. The older TV and
gradient-L0 demonstration scripts are retained in their original folders
for comparison with HQS and ADMM formulations.
