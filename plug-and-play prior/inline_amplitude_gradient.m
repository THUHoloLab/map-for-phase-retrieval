function [gradient, error] = inline_amplitude_gradient( ...
    object, amplitude, distance, wavelength, pixelSize)
%INLINE_AMPLITUDE_GRADIENT Gradient of the detector-amplitude loss.

detector = propagate_angular_spectrum( ...
    object, distance, wavelength, pixelSize);
residual = detector - amplitude .* exp(1i * angle(detector));
gradient = propagate_angular_spectrum( ...
    residual, -distance, wavelength, pixelSize);
error = norm(residual, "fro") / norm(amplitude, "fro");
end
