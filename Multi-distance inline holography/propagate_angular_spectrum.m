function output = propagate_angular_spectrum(input, distance, wavelength, pixelSize)
%PROPAGATE_ANGULAR_SPECTRUM Propagate a complex field with the ASM.
%   OUTPUT = PROPAGATE_ANGULAR_SPECTRUM(INPUT, DISTANCE, WAVELENGTH,
%   PIXELSIZE) uses the angular-spectrum transfer function. DISTANCE is
%   signed, so negative values propagate toward the object plane.
%
%   INPUT is an M-by-N complex field. WAVELENGTH and PIXELSIZE use the
%   same length unit. Evanescent frequencies are suppressed for stability.

arguments
    input {mustBeNumeric}
    distance (1,1) double {mustBeReal}
    wavelength (1,1) double {mustBePositive}
    pixelSize (1,1) double {mustBePositive}
end

[rows, columns] = size(input);
frequencyX = ifftshift((-floor(columns / 2):ceil(columns / 2)-1) / ...
    (columns * pixelSize));
frequencyY = ifftshift((-floor(rows / 2):ceil(rows / 2)-1) / ...
    (rows * pixelSize));
[fx, fy] = meshgrid(frequencyX, frequencyY);

propagating = wavelength^2 * (fx.^2 + fy.^2) <= 1;
longitudinalFrequency = sqrt(max(0, 1 - ...
    wavelength^2 * (fx.^2 + fy.^2)));
transferFunction = propagating .* exp( ...
    1i * 2 * pi * distance / wavelength .* longitudinalFrequency);

output = ifft2(fft2(input) .* transferFunction);
end
