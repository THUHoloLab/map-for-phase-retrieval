function [object, error] = pnp_admm_inline( ...
    intensity, distance, wavelength, pixelSize, iterations, tvWeight)
%PNP_ADMM_INLINE PnP-ADMM for single-distance inline holography.
%   The data step replaces the detector amplitude and propagates the field
%   back to the object plane. The prior step uses a TV denoiser.

if nargin < 5, iterations = 100; end
if nargin < 6, tvWeight = 0.01; end

amplitude = sqrt(max(intensity, 0));
object = propagate_angular_spectrum( ...
    amplitude, -distance, wavelength, pixelSize);

v = object;
u = zeros(size(object));
error = zeros(iterations, 1);

for k = 1:iterations
    % Inline-holography data constraint.
    detector = propagate_angular_spectrum( ...
        v - u, distance, wavelength, pixelSize);
    detector = amplitude .* exp(1i * angle(detector));
    x = propagate_angular_spectrum( ...
        detector, -distance, wavelength, pixelSize);

    % Plug-and-play prior and ADMM multiplier update.
    v = tv_admm_denoiser(x + u, tvWeight, 10);
    u = u + x - v;

    predicted = propagate_angular_spectrum( ...
        v, distance, wavelength, pixelSize);
    error(k) = norm(abs(predicted) - amplitude, "fro") / ...
        norm(amplitude, "fro");
end

object = v;
end
