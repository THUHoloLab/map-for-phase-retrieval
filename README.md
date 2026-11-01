# MAP for Phase Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/MATLAB-R2024b-E16737?logo=mathworks"
       alt="MATLAB R2024b">
  <img src="https://img.shields.io/badge/License-MIT-2F855A"
       alt="MIT License">
  <img src="https://img.shields.io/badge/status-active%20development-3B82F6"
       alt="Active development">
</p>

> MATLAB implementations of phase-retrieval models and algorithms from a
> maximum a posteriori inference perspective.

Phase retrieval estimates a complex optical field from intensity-only
measurements. Although reconstruction algorithms can look very different,
many can be organized around the same components:

```text
measurement model + likelihood + prior/constraint + optimization algorithm
```

This repository accompanies the review manuscript *Recovering phase from
intensity: an inference view*. It provides compact, commented MATLAB
implementations for Fourier phase retrieval, inline holography,
ptychography, Fourier ptychography, likelihood functions, and regularizers.

The repository is under active development. The emphasis is on readable
reference implementations that expose the connection between physical
forward models and statistical inference.

## MAP Framework

For measurements `I`, an unknown complex field `U`, and a forward operator
`A`, MAP reconstruction can be written as

```text
U_MAP = arg min_U F(A(U), I) + lambda R(U)
```

where:

- `F` is a negative log-likelihood or practical data-fidelity loss;
- `R` is a prior, regularizer, or physical constraint;
- `lambda` balances measurement consistency and prior information.

The code is organized so likelihood gradients, regularization gradients,
forward models, and optimization schemes can be studied independently or
combined into a reconstruction pipeline.

## Implemented Methods

| Area | Implementations | Entry points |
| --- | --- | --- |
| Likelihood models | Gaussian intensity/amplitude/gamma, Poisson, Anscombe, Poisson-Gaussian, Laplace, Hyper-Laplacian, Huber | [`likelihood_forward.m`](likelihood/likelihood_forward.m), [`likelihood_backward.m`](likelihood/likelihood_backward.m) |
| Priors and constraints | L2, smooth L1, gradient L2, TV, gradient Hyper-Laplacian, Hessian L2, Hessian-TV, L0/L1 proximal operators, support and magnitude constraints | [`regularization_forward.m`](regularization/regularization_forward.m), [`regularization_project.m`](regularization/regularization_project.m) |
| Fourier phase retrieval | Hybrid Input-Output and Difference Map | [`hio_phase_retrieval.m`](Fourier%20spectrum%20phase%20retrieval/hio_phase_retrieval.m), [`difference_map_phase_retrieval.m`](Fourier%20spectrum%20phase%20retrieval/difference_map_phase_retrieval.m) |
| Multi-distance inline holography | Sequential layer-wise projection, global backpropagation, random-layer backpropagation | [`reconstruct_sequential_layers.m`](Multi-distance%20inline%20holography/reconstruct_sequential_layers.m), [`reconstruct_global_backprop.m`](Multi-distance%20inline%20holography/reconstruct_global_backprop.m) |
| Scanning ptychography | rPIE and automatic differentiation with complex Adam | [`ptycho_reconstruct_rpie.m`](Ptychography/ptycho_reconstruct_rpie.m), [`ptycho_reconstruct_autodiff.m`](Ptychography/ptycho_reconstruct_autodiff.m) |
| Fourier ptychography | m-FPM, AS-FPM, ADMM-FPM, FD-FPM, VEM-FPM | [`Fourier ptychography`](Fourier%20ptychography/README.md) |
| Splitting methods | TV and gradient-L0 examples using gradient descent, HQS, and ADMM/Split Bregman | [`ADMM family`](ADMM%20family) |

## Likelihood Models

The [`likelihood`](likelihood/README.md) module accepts a predicted complex
detector field and measured intensity. A typical forward/backward call is:

```matlab
[loss, cache] = likelihood_forward(detectorField, intensity, ...
    "poisson");
detectorGradient = likelihood_backward(cache);
```

For a linear optical operator and its adjoint:

```matlab
[loss, objectGradient] = likelihood_loss( ...
    object, intensity, forwardOperator, adjointOperator, ...
    "gaussian_amplitude");
```

The returned gradients are exact complex Euclidean gradients. Numerical
finite-difference tests are included for every implemented loss.

## Priors and Constraints

The [`regularization`](regularization/README.md) module provides
differentiable priors:

```matlab
[priorLoss, cache] = regularization_forward(object, "tv", ...
    weight=1e-3);
priorGradient = regularization_backward(cache);
```

It also provides proximal and projection operations:

```matlab
object = regularization_prox(object, "l1", 0.01);
object = regularization_project(object, "support", mask=support);
object = regularization_project(object, "magnitude_bounds", ...
    minimum=0, maximum=1);
```

## Fourier Spectrum Phase Retrieval

The [`Fourier spectrum phase retrieval`](Fourier%20spectrum%20phase%20retrieval/README.md)
module contains deliberately short implementations of two projection
algorithms:

1. **Hybrid Input-Output (HIO)** uses feedback outside the known object
   support to reduce stagnation.
2. **Difference Map (DM)** combines reflected support and Fourier-magnitude
   projections.

```matlab
[objectHIO, errorHIO] = hio_phase_retrieval( ...
    measuredFourierAmplitude, support);
[objectDM, errorDM] = difference_map_phase_retrieval( ...
    measuredFourierAmplitude, support);
```

## Multi-distance Inline Holography

The [`Multi-distance inline holography`](Multi-distance%20inline%20holography/README.md)
module shares one angular-spectrum forward model across three update
strategies:

1. **Sequential layer-wise:** impose each distance constraint and
   immediately propagate the corrected field back.
2. **Global backpropagation:** accumulate residuals from all distances
   before updating the object.
3. **Random-layer backpropagation:** update from one randomly selected
   distance at each step.

<p align="center">
  <img src="Multi-distance%20inline%20holography/method_schematic.png"
       width="900"
       alt="Comparison of multi-distance inline holography reconstruction strategies">
</p>

## Ptychography

The [`Ptychography`](Ptychography/README.md) module uses a common simulated
dataset for two reconstruction styles:

- **rPIE:** explicit sequential object/probe updates;
- **automatic differentiation:** a differentiable ptychographic operator,
  `dlgradient`, mini-batches, and complex Adam updates.

```matlab
[object, probe, history] = ptycho_reconstruct_rpie( ...
    data, initialObject, initialProbe);

[object, probe, history] = ptycho_reconstruct_autodiff( ...
    data, initialObject, initialProbe, batchSize=8);
```

The automatic-differentiation implementation uses a custom
`deep.DifferentiableFunction` for the complex Fourier operator and its
adjoint.

## Fourier Ptychography

The [`Fourier ptychography`](Fourier%20ptychography/README.md) benchmark
contains synthetic and experimental examples for:

1. momentum FPM;
2. adaptive-step FPM;
3. ADMM-FPM;
4. feature-domain FPM;
5. variational EM FPM.

APIC is referenced as an external library and is not distributed here:
[APIC analytical reconstruction](https://github.com/rzcao/APIC-analytical-complex-field-reconstruction).

## Quick Start

Clone the repository and open MATLAB in its root directory:

```matlab
addpath(genpath(pwd));
```

Run the compact numerical tests:

```matlab
test_likelihood_gradients
test_regularization_gradients
test_fourier_spectrum_phase_retrieval
test_multidistance_inline_holography
test_ptychography
```

Run the visual demonstrations:

```matlab
demo_likelihood
demo_regularization
demo_fourier_spectrum_phase_retrieval
demo_multidistance_inline_holography
demo_ptychography
```

## Requirements

- MATLAB R2024b or a compatible recent release
- Image Processing Toolbox for several imaging examples
- Deep Learning Toolbox for automatic-differentiation ptychography
- Parallel Computing Toolbox and an NVIDIA GPU are optional accelerators
- Statistics and Machine Learning Toolbox is required only when Poisson
  simulation through `poissrnd` is enabled

The core likelihood, regularization, Fourier phase-retrieval, and
multi-distance algorithms run on CPU.

## Repository Structure

```text
map-for-phase-retrieval/
|-- likelihood/
|-- regularization/
|-- Fourier spectrum phase retrieval/
|-- Multi-distance inline holography/
|-- Ptychography/
|-- Fourier ptychography/
|-- ADMM family/
|-- MAP_LPR_20261101_clean.pdf
`-- LICENSE
```

Each major module contains its own README, demonstration, and numerical
test where applicable.

## Verification

The repository includes:

- centered finite-difference checks for likelihood gradients;
- centered finite-difference checks for regularization gradients;
- a finite-difference check for the differentiable ptychography operator;
- angular-spectrum forward/backward round-trip tests;
- reconstruction smoke tests for HIO, Difference Map, rPIE, automatic
  differentiation, and multi-distance holography.

## Citation

This repository accompanies:

> Shuhe Zhang and Liangcai Cao, *Recovering phase from intensity: an
> inference view*.

The current manuscript is included as
[`MAP_LPR_20261101_clean.pdf`](MAP_LPR_20261101_clean.pdf). Publication
metadata and BibTeX will be added after formal publication.

When using an algorithm adapted from or compared with an external project,
please also cite the original paper and repository listed in the
corresponding module README.

## License and Third-Party Code

Original code in this repository is released under the
[MIT License](LICENSE).

Some benchmark folders contain imported or adapted research code with
additional attribution or licensing requirements. Review the
`THIRD_PARTY_NOTICES.md` and local license files inside those modules before
redistribution. In particular, APIC source code is not included.

## Questions and Feedback

Please use the repository
[Issues](https://github.com/THUHoloLab/map-for-phase-retrieval/issues)
page for bug reports, questions, and feature requests.
