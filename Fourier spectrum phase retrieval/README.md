# Fourier Spectrum Phase Retrieval

Two short object-domain algorithms are provided for recovering a complex
object from a measured Fourier amplitude:

- `hio_phase_retrieval.m`: Hybrid Input-Output (HIO)
- `difference_map_phase_retrieval.m`: Difference Map

Both methods use the same two projections:

1. Fourier-magnitude projection:

```matlab
F <- measuredAmplitude .* F ./ abs(F)
```

2. Object-support projection:

```matlab
object(~support) = 0
```

## Example

```matlab
[object, errorHistory] = hio_phase_retrieval( ...
    measuredAmplitude, support);

[object, errorHistory] = difference_map_phase_retrieval( ...
    measuredAmplitude, support);
```

Run `demo_fourier_spectrum_phase_retrieval` for a visual comparison and
`test_fourier_spectrum_phase_retrieval` for a compact numerical test.
