# Likelihood losses

This folder implements data-fidelity terms for complex-valued phase
retrieval. The measured input is always an intensity image, while the
prediction is a complex detector field.

## Basic use

```matlab
[loss, cache] = likelihood_forward(detectorField, intensity, ...
    "gaussian_intensity");
detectorGradient = likelihood_backward(cache);
```

For a linear forward model `A`, map the detector gradient back to the object
with the adjoint operator:

```matlab
[loss, objectGradient] = likelihood_loss(object, intensity, ...
    @(x) A * x, @(y) A' * y, "poisson");
```

The gradient convention is

```text
d loss = real(sum(conj(gradient) .* d field))
```

so the real and imaginary parts are ordinary derivatives with respect to
the real and imaginary parts of the field. Constant factors may differ from
the scaled Wirtinger descent directions in Table 1 of the review because
this implementation returns exact derivatives of the documented losses.

## Available models

| Model | Intended measurement model |
| --- | --- |
| `gaussian_intensity` | Additive Gaussian noise on intensity |
| `gaussian_amplitude` | Additive Gaussian noise on amplitude |
| `gaussian_gamma` | Gaussian noise after a power transform |
| `poisson` | Photon-counting noise |
| `anscombe` | Variance-stabilized Poisson surrogate |
| `poisson_gaussian` | Shot and read-noise surrogate |
| `laplace_amplitude` | Heavy-tailed amplitude residuals |
| `hyper_laplacian` | Sparse, heavy-tailed amplitude residuals |
| `huber_intensity` | Gaussian inliers with robust outlier handling |

Run `test_likelihood_gradients` to compare every backward pass with a
centered finite-difference directional derivative.
