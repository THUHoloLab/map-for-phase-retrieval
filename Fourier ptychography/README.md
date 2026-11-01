# Fourier Ptychographic Microscopy Benchmarks

This directory contains MATLAB implementations of five Fourier
ptychographic microscopy (FPM) reconstruction methods. The code was
imported from the
[FPM_benchmarks](https://github.com/ShuheZhang-MUMC/FPM_benchmarks)
repository at commit `5c0f327a091784ee0b30a0b93840d272153f3fe3`.

## Implemented Methods

| Method | Description |
| --- | --- |
| m-FPM | Extended ptychographic iterative engine with momentum |
| AS-FPM | FPM with adaptive step-size control |
| ADMM-FPM | Alternating-direction reconstruction |
| FD-FPM | FPM using a feature-domain data-fidelity term |
| VEM-FPM | Variational expectation-maximization reconstruction |

APIC is not copied into this repository. Use the authors' external
[APIC analytical reconstruction library](https://github.com/rzcao/APIC-analytical-complex-field-reconstruction)
when APIC results are required.

## Directory Structure

- `simulations_benchmark`: Synthetic data generation, reconstruction, and
  quantitative comparison of all six methods.
- `Experiment_1`: Reconstructions for published experimental datasets.
- `Experiment_2`: Reconstructions for USAF target datasets.
- `sources`: Figures used by this documentation.

## Simulation Workflow

1. Open MATLAB in `simulations_benchmark`.
2. Run `gen_img_batch.m` to generate noisy synthetic measurements.
3. Open `simulations_benchmark/recover`.
4. Run `run_benchmarks.m` to reconstruct the generated datasets.

The scripts use relative paths, so run them from the directories stated
above. Adjust the selected algorithm, noise levels, LED shifts, and output
paths directly in the two driver scripts.

## Main Data Structure

Each reconstruction function accepts a `loaded_data` structure containing:

| Field | Meaning |
| --- | --- |
| `I_low` | Low-resolution intensity stack, indexed by LED |
| `dpix_c` | Camera pixel pitch |
| `mag` | Objective magnification |
| `na_cal` | Objective numerical aperture |
| `freqXY_calib` | Calibrated LED centers in Fourier-space pixels |
| `na_calib` | Calibrated illumination numerical apertures |
| `na_rp_cal` | Objective pupil radius in Fourier-space pixels |

Most iterative solvers require a CUDA-capable GPU through MATLAB's
Parallel Computing Toolbox. Image Processing Toolbox is also required.

## Default Simulation Parameters

| Parameter | Value |
| --- | --- |
| Wavelength | 0.62863 um |
| Objective numerical aperture | 0.25 |
| Magnification | 10x |
| Camera pixel pitch | 4.8 um |
| LED-to-sample distance | 50 mm |
| LED radial spacing | 6 mm |
| LED ring counts | `[1, 8, 12, 16, 24, 36, 48, 54]` |
| Low-resolution image size | 128 x 128 |
| Reconstructed image size | 512 x 512 |
| Upsampling ratio | 4 |

## References

- m-FPM is adapted from the UConn Fourier ptychography implementation.
- AS-FPM follows the adaptive step-size method in *Optics Express* 24,
  20724-20735.
- APIC is available from the
  [authors' repository](https://github.com/rzcao/APIC-analytical-complex-field-reconstruction)
  and is described in *Nature Communications* 15, 4544 (2024).
- FD-FPM follows the feature-domain method in *Optica* 11, 634-646 (2024).
