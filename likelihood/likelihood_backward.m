function gradient = likelihood_backward(cache, upstream)
%LIKELIHOOD_BACKWARD Differentiate a likelihood with respect to FIELD.
%   GRADIENT = LIKELIHOOD_BACKWARD(CACHE) returns the complex Euclidean
%   gradient of the loss evaluated by LIKELIHOOD_FORWARD.
%
%   The convention is
%       dLOSS = real(sum(conj(GRADIENT) .* dFIELD, "all")).
%   Therefore real(GRADIENT) and imag(GRADIENT) equal the derivatives with
%   respect to the real and imaginary parts of FIELD, respectively.
%
%   GRADIENT = LIKELIHOOD_BACKWARD(CACHE, UPSTREAM) additionally applies a
%   real scalar upstream derivative (default: 1).

arguments
    cache (1,1) struct
    upstream (1,1) double {mustBeReal} = 1
end

field = cache.field;
measurement = cache.measurement;
intensity = cache.intensity;
amplitude = cache.amplitude;
epsilon = cache.epsilon;
residual = cache.residual;

switch cache.model
    case "gaussian_intensity"
        radialDerivative = residual;

    case "gaussian_amplitude"
        radialDerivative = residual ./ (2 * amplitude);

    case "gaussian_gamma"
        radialDerivative = residual .* cache.gamma .* ...
            (intensity + epsilon).^(cache.gamma - 1);

    case "poisson"
        radialDerivative = 1 - measurement ./ (intensity + epsilon);

    case "anscombe"
        radialDerivative = residual ./ amplitude;

    case "poisson_gaussian"
        denominator = intensity + epsilon;
        radialDerivative = residual .* ...
            (intensity + measurement + 2 * epsilon) ./ denominator.^2;

    case "laplace_amplitude"
        radialDerivative = sign(residual) ./ (2 * amplitude);

    case "hyper_laplacian"
        radialDerivative = cache.p * residual .* ...
            (residual.^2 + epsilon^2).^(cache.p / 2 - 1) ./ ...
            (2 * amplitude);

    case "huber_intensity"
        radialDerivative = residual;
        outsideQuadraticRegion = abs(residual) > cache.delta;
        radialDerivative(outsideQuadraticRegion) = cache.delta * ...
            sign(residual(outsideQuadraticRegion));

    otherwise
        error("likelihood_backward:UnknownModel", ...
            "Unknown likelihood model in CACHE: %s.", cache.model);
end

% For q = |field|^2, the complex Euclidean gradient is
% 2 * (d loss / d q) .* field.
gradient = upstream * 2 .* cache.weight .* radialDerivative .* field;
end
