function test_multidistance_inline_holography
%TEST_MULTIDISTANCE_INLINE_HOLOGRAPHY Run compact numerical tests.

rng(8);
pixelSize = 1e-6;
wavelength = 500e-9;
imageSize = 32;
distances = [0.2; 0.5; 0.8] * 1e-3;
[x, y] = meshgrid(linspace(-1, 1, imageSize));
objectField = (0.8 + 0.1 * cos(4 * pi * x)) .* exp(1i * 0.4 * y);
data = simulate_multidistance_holography( ...
    objectField, distances, wavelength, pixelSize);

% Propagation followed by its adjoint should preserve field energy.
testField = randn(imageSize) + 1i * randn(imageSize);
forward = propagate_angular_spectrum( ...
    testField, distances(2), wavelength, pixelSize);
backward = propagate_angular_spectrum( ...
    forward, -distances(2), wavelength, pixelSize);
relativeRoundTripError = norm(testField(:) - backward(:)) / norm(testField(:));
assert(relativeRoundTripError < 1e-10, ...
    "Angular-spectrum round-trip error is %.3e.", relativeRoundTripError);

initialField = sqrt(data.intensity(:, :, 1));
[~, sequentialHistory] = reconstruct_sequential_layers( ...
    data, initialField, iterations=4, verbose=false);
[~, globalHistory] = reconstruct_global_backprop( ...
    data, initialField, iterations=8, stepSize=0.2, verbose=false);
[~, randomHistory] = reconstruct_random_layer_backprop( ...
    data, initialField, iterations=24, stepSize=0.15, verbose=false);

assert(all(isfinite(sequentialHistory.loss)));
assert(all(isfinite(globalHistory.loss)));
assert(all(isfinite(randomHistory.loss)));
fprintf("ASM round-trip relative error: %.3e\n", relativeRoundTripError);
fprintf("Multi-distance inline holography tests passed.\n");
end
