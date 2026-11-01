function test_fourier_spectrum_phase_retrieval
%TEST_FOURIER_SPECTRUM_PHASE_RETRIEVAL Run a small reconstruction test.

rng(6);
imageSize = 48;
[x, y] = meshgrid(1:imageSize, 1:imageSize);
support = (x - imageSize / 2).^2 + (y - imageSize / 2).^2 < ...
    (0.35 * imageSize)^2;
trueObject = support .* exp(1i * 0.4 * sin(x / 7));
fourierAmplitude = abs(fft2(trueObject));

options.object = support .* exp(1i * 2 * pi * rand(imageSize));
options.iterations = 200;

[hioObject, hioError] = hio_phase_retrieval( ...
    fourierAmplitude, support, options);
[dmObject, dmError] = difference_map_phase_retrieval( ...
    fourierAmplitude, support, options);

assert(all(isfinite(hioObject), "all"));
assert(all(isfinite(dmObject), "all"));
assert(hioError(end) < hioError(1));
assert(dmError(end) < dmError(1));
fprintf("HIO error: %.3e -> %.3e\n", hioError(1), hioError(end));
fprintf("Difference Map error: %.3e -> %.3e\n", dmError(1), dmError(end));
fprintf("Fourier spectrum phase retrieval tests passed.\n");
end
