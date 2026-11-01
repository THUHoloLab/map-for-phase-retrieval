function [object, errorHistory] = difference_map_phase_retrieval( ...
        fourierAmplitude, support, options)
%DIFFERENCE_MAP_PHASE_RETRIEVAL Difference Map phase retrieval.
%   OBJECT = DIFFERENCE_MAP_PHASE_RETRIEVAL(FOURIERAMPLITUDE, SUPPORT,
%   OPTIONS) uses Fourier-magnitude and support projections.
%
%   OPTIONS can contain:
%     iterations  Number of iterations, default 500
%     beta        Difference Map step parameter, default 1
%     object      Initial complex object, default random phase

if nargin < 3
    options = struct;
end
if ~isfield(options, "iterations"), options.iterations = 500; end
if ~isfield(options, "beta"), options.beta = 0.6; end

if isfield(options, "object")
    object = options.object;
else
    object = exp(1i * 2 * pi * rand(size(fourierAmplitude)));
end

fourierAmplitude = double(fourierAmplitude);
support = logical(support);
errorHistory = zeros(options.iterations, 1);
epsilon = 1e-8;

for iter = 1:options.iterations
    % Projection onto the Fourier-magnitude constraint.
    spectrum = fft2(object);
    projectedSpectrum = fourierAmplitude .* spectrum ./ ...
        max(abs(spectrum), epsilon);
    magnitudeProjection = ifft2(projectedSpectrum);

    % Difference Map update:
    % x <- x + beta * (P_support(2 P_magnitude(x) - x) - P_magnitude(x)).
    reflected = 2 * magnitudeProjection - object;
    reflected(~support) = 0;
    object = object + options.beta * (reflected - magnitudeProjection);

    errorHistory(iter) = mean((abs(spectrum) - fourierAmplitude).^2, "all");
end
end
