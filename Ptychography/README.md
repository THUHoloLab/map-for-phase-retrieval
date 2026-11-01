# Ptychography

This module provides two MATLAB implementations of far-field scanning
ptychography:

- `ptycho_reconstruct_rpie`: a conventional sequential rPIE solver with
  explicit object and probe updates.
- `ptycho_reconstruct_autodiff`: an end-to-end differentiable solver using
  `dlarray`, `dlfeval`, `dlgradient`, and complex Adam updates.

Both solvers are independent implementations released under this
repository's MIT license. No source code from SheffieldPtycho is copied
because that project uses an academic, non-commercial license.

## Data convention

```matlab
data.amplitude  % probeHeight-by-probeWidth-by-numberOfFrames
data.positions  % numberOfFrames-by-2, [row, column], one-based
```

Each position identifies the top-left corner of an object patch. The
forward model is

```text
exit wave = object patch .* probe
detector amplitude = abs(fftshift(fft2(exit wave)))
```

## Simulation

```matlab
data = ptycho_simulate(object, probe, positions, ...
    photonCount=1e5, readNoise=1e-3);
```

Poisson noise is optional. `poissrnd` requires Statistics and Machine
Learning Toolbox; noise-free simulation has no such dependency.

## Conventional reconstruction

```matlab
[object, probe, history] = ptycho_reconstruct_rpie( ...
    data, initialObject, initialProbe, epochs=100);
```

The rPIE denominator blends local illumination power with its maximum.
This prevents unstable updates in weakly illuminated pixels. Use
`updateProbe=false` when the probe is calibrated.

## Automatic differentiation

```matlab
[object, probe, history] = ptycho_reconstruct_autodiff( ...
    data, initialObject, initialProbe, epochs=100, batchSize=8);
```

The differentiable version requires MATLAB Deep Learning Toolbox. Set
`useGPU=true` to use a supported GPU. `PtychoAmplitudeOperator` implements
the centered FFT forward model and its exact adjoint as a custom
`deep.DifferentiableFunction`. Unlike PtychoX, this educational
implementation does not use custom CUDA kernels, favoring readability and
portability over maximum throughput.

## Examples and tests

- Run `demo_ptychography` for a visual comparison.
- Run `test_ptychography` for a compact CPU smoke test.

## References

- A. M. Maiden and J. M. Rodenburg, "An improved ptychographical phase
  retrieval algorithm for diffractive imaging," *Ultramicroscopy* 109,
  1256-1262 (2009).
- A. M. Maiden, D. Johnson, and P. Li, "Further improvements to the
  ptychographical iterative engine," *Optica* 4, 736-745 (2017).
- [SheffieldPtycho](https://github.com/andyMaiden/SheffieldPtycho), used as
  an algorithmic reference for conventional PIE-family organization.
- [PtychoX](https://github.com/THUHoloLab/PtychoX), used as an algorithmic
  reference for differentiable ptychographic reconstruction.
