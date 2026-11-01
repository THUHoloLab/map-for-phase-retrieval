function [object, errorHistory] = hio_phase_retrieval(fourierAmplitude, support, options)
%HIO_PHASE_RETRIEVAL Hybrid input-output phase retrieval.
%   OBJECT = HIO_PHASE_RETRIEVAL(FOURIERAMPLITUDE, SUPPORT, OPTIONS)
%   reconstructs an object from its Fourier magnitude.
%
%   SUPPORT is a logical mask in the object plane. OPTIONS can contain:
%     iterations  Number of iterations, default 500
%     beta        HIO feedback parameter, default 0.9
%     object      Initial complex object, default random phase

if nargin < 3
    options = struct;
end
if ~isfield(options, "iterations"), options.iterations = 500; end
if ~isfield(options, "beta"), options.beta = 0.5; end

if isfield(options, "object")
    object = options.object;
else
    object = exp(1i * 2 * pi * rand(size(fourierAmplitude)));
end

errorHistory = zeros(options.iterations, 1);
fourierAmplitude = double(fourierAmplitude);
support = logical(support);
epsilon = 1e-8;

for iter = 1:options.iterations
    % Enforce the measured Fourier magnitude.
    spectrum = fft2(object);
    projectedSpectrum = fourierAmplitude .* spectrum ./ ...
        max(abs(spectrum), epsilon);
    projectedObject = ifft2(projectedSpectrum);

    % HIO feedback outside the known support.
    oldObject = object;
    object = projectedObject;
    object(~support) = oldObject(~support) - ...
        options.beta * projectedObject(~support);

    errorHistory(iter) = mean((abs(spectrum) - fourierAmplitude).^2, "all");
end
end
