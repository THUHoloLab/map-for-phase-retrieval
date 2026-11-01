function [object, error] = red_inline( ...
    intensity, distance, wavelength, pixelSize, iterations, redWeight)
%RED_INLINE Regularization by Denoising for inline holography.

if nargin < 5, iterations = 100; end
if nargin < 6, redWeight = 0.1; end

amplitude = sqrt(max(intensity, 0));
object = propagate_angular_spectrum( ...
    amplitude, -distance, wavelength, pixelSize);
error = zeros(iterations, 1);
step = 0.8;

for k = 1:iterations
    [dataGradient, error(k)] = inline_amplitude_gradient( ...
        object, amplitude, distance, wavelength, pixelSize);
    denoised = tv_admm_denoiser(object, 0.01, 10);
    redGradient = object - denoised;
    object = object - step * ...
        (dataGradient + redWeight * redGradient);
end
end
