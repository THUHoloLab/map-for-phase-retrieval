function [object, error] = pnp_fista_inline( ...
    intensity, distance, wavelength, pixelSize, iterations, tvWeight)
%PNP_FISTA_INLINE PnP-FISTA for single-distance inline holography.

if nargin < 5, iterations = 100; end
if nargin < 6, tvWeight = 0.01; end

amplitude = sqrt(max(intensity, 0));
object = propagate_angular_spectrum( ...
    amplitude, -distance, wavelength, pixelSize);
y = object;
t = 1;
error = zeros(iterations, 1);

for k = 1:iterations
    [gradient, error(k)] = inline_amplitude_gradient( ...
        y, amplitude, distance, wavelength, pixelSize);

    oldObject = object;
    object = tv_admm_denoiser(y - gradient, tvWeight, 10);

    oldT = t;
    t = (1 + sqrt(1 + 4 * t^2)) / 2;
    y = object + (oldT - 1) / t * (object - oldObject);
end
end
