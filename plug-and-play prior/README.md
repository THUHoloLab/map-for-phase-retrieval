# Plug-and-Play Prior

This folder contains compact denoiser-based algorithms for
single-distance inline holography:

- `pnp_admm_inline`: amplitude projection and an ADMM denoiser update;
- `pnp_fista_inline`: amplitude-loss gradient, TV denoising, and momentum;
- `red_inline`: amplitude-loss gradient and the RED residual `x - D(x)`.

```matlab
[object, error] = pnp_admm_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.01);

[object, error] = pnp_fista_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.01);

[object, error] = red_inline( ...
    intensity, distance, wavelength, pixelSize, 100, 0.1);
```

Run:

```matlab
addpath(genpath(pwd))
demo_pnp_inline
test_pnp_inline
```

The propagation function is shared with the `Multi-distance inline
holography` folder. The code uses the repository's MIT License.
