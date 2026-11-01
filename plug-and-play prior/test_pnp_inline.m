function test_pnp_inline
%TEST_PNP_INLINE Run a small numerical test.

n = 32;
pixelSize = 1.12e-6;
wavelength = 532e-9;
distance = 0.8e-3;

[x, y] = meshgrid(linspace(-1, 1, n));
truth = (1 - 0.2 * exp(-10 * (x.^2 + y.^2))) .* ...
    exp(0.5i * exp(-15 * ((x - 0.2).^2 + y.^2)));
detector = propagate_angular_spectrum( ...
    truth, distance, wavelength, pixelSize);

[admm, errorAdmm] = pnp_admm_inline( ...
    abs(detector).^2, distance, wavelength, pixelSize, 10, 0.002);
[fista, errorFista] = pnp_fista_inline( ...
    abs(detector).^2, distance, wavelength, pixelSize, 10, 0.002);
[red, errorRed] = red_inline( ...
    abs(detector).^2, distance, wavelength, pixelSize, 10, 0.1);

assert(all(isfinite([admm(:); fista(:); red(:)])))
assert(all(isfinite([errorAdmm; errorFista; errorRed])))
disp("PnP inline-holography test passed.")
end
