# Multi-distance Inline Holography

This module reconstructs a complex object field from inline holograms
recorded at several propagation distances. It uses a common angular
spectrum propagator and provides three update strategies:

1. `reconstruct_sequential_layers`: visit each distance in sequence,
   replace the propagated amplitude, and backpropagate immediately.
2. `reconstruct_global_backprop`: accumulate amplitude residuals from all
   distances and apply one global object-plane update.
3. `reconstruct_random_layer_backprop`: randomly select one distance per
   update and backpropagate only that layer's residual.

## Data format

```matlab
data.intensity      % height-by-width-by-numberOfDistances
data.distances      % signed propagation distances
data.wavelength     % same unit as distances and pixelSize
data.pixelSize      % detector/object-plane pixel pitch
```

## Example

```matlab
data = simulate_multidistance_holography( ...
    objectField, distances, wavelength, pixelSize);
initialField = sqrt(data.intensity(:, :, 1));

[field, history] = reconstruct_sequential_layers(data, initialField);
[field, history] = reconstruct_global_backprop(data, initialField);
[field, history] = reconstruct_random_layer_backprop(data, initialField);
```

Run `demo_multidistance_inline_holography` for a visual comparison or
`test_multidistance_inline_holography` for a compact CPU test.

Run `plot_method_schematic` to generate `method_schematic.png`, a visual
comparison of the information flow in the three reconstruction strategies.

The propagation operator uses the angular-spectrum transfer function. The
three reconstruction schemes are intentionally kept explicit so they can
be connected to the likelihood and regularization APIs elsewhere in this
repository.
