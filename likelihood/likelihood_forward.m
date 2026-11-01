function [loss, cache] = likelihood_forward(field, measurement, model, options)
%LIKELIHOOD_FORWARD Evaluate a phase-retrieval data-fidelity term.
%   [LOSS, CACHE] = LIKELIHOOD_FORWARD(FIELD, MEASUREMENT, MODEL, OPTIONS)
%   evaluates a loss between the predicted detector FIELD and a measured
%   intensity image. CACHE contains the intermediate variables required by
%   LIKELIHOOD_BACKWARD.
%
%   Supported MODEL values:
%     "gaussian_intensity"      Gaussian noise on intensity
%     "gaussian_amplitude"      Gaussian noise on amplitude
%     "gaussian_gamma"          Gaussian noise after a power transform
%     "poisson"                 Poisson photon-counting likelihood
%     "anscombe"                Variance-stabilized Poisson surrogate
%     "poisson_gaussian"        Poisson-Gaussian rational surrogate
%     "laplace_amplitude"       Robust L1 amplitude loss
%     "hyper_laplacian"         Robust amplitude loss with 0 < p < 1
%     "huber_intensity"         Huber loss on intensity residuals
%
%   OPTIONS is optional and can contain:
%     epsilon   Positive numerical stabilizer (default: 1e-8)
%     weight    Scalar or array of nonnegative data weights (default: 1)
%     gamma     Power for gaussian_gamma (default: 0.5)
%     p         Exponent for hyper_laplacian (default: 0.8)
%     delta     Huber transition point (default: 1)
%
%   The returned loss is a sum over all detector samples. MEASUREMENT is
%   interpreted as intensity for every model.

arguments
    field {mustBeNumeric}
    measurement {mustBeNumeric, mustBeReal, mustBeNonnegative}
    model {mustBeTextScalar}
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.weight {mustBeNumeric, mustBeReal, mustBeNonnegative} = 1
    options.gamma (1,1) double {mustBePositive} = 0.5
    options.p (1,1) double {mustBePositive} = 0.8
    options.delta (1,1) double {mustBePositive} = 1
end

if ~isequal(size(field), size(measurement))
    error("likelihood_forward:SizeMismatch", ...
        "FIELD and MEASUREMENT must have identical sizes.");
end

weight = options.weight;
if ~isscalar(weight) && ~isequal(size(weight), size(field))
    error("likelihood_forward:WeightSizeMismatch", ...
        "OPTIONS.weight must be scalar or have the same size as FIELD.");
end

model = lower(string(model));
epsilon = options.epsilon;
intensity = abs(field).^2;
amplitude = sqrt(intensity + epsilon);
measuredAmplitude = sqrt(measurement);

cache = struct( ...
    "field", field, ...
    "measurement", measurement, ...
    "model", model, ...
    "epsilon", epsilon, ...
    "weight", weight, ...
    "intensity", intensity, ...
    "amplitude", amplitude, ...
    "gamma", options.gamma, ...
    "p", options.p, ...
    "delta", options.delta);

switch model
    case "gaussian_intensity"
        residual = intensity - measurement;
        pointLoss = 0.5 * residual.^2;

    case "gaussian_amplitude"
        residual = amplitude - measuredAmplitude;
        pointLoss = 0.5 * residual.^2;

    case "gaussian_gamma"
        predicted = (intensity + epsilon).^options.gamma;
        observed = (measurement + epsilon).^options.gamma;
        residual = predicted - observed;
        pointLoss = 0.5 * residual.^2;
        cache.predicted = predicted;

    case "poisson"
        predicted = intensity + epsilon;
        residual = intensity - measurement;
        pointLoss = predicted - measurement .* log(predicted);

    case "anscombe"
        predicted = 2 * amplitude;
        observed = 2 * sqrt(measurement + 3 / 8);
        residual = predicted - observed;
        pointLoss = 0.5 * residual.^2;

    case "poisson_gaussian"
        residual = intensity - measurement;
        denominator = intensity + epsilon;
        pointLoss = residual.^2 ./ denominator;

    case "laplace_amplitude"
        residual = amplitude - measuredAmplitude;
        pointLoss = abs(residual);

    case "hyper_laplacian"
        if options.p >= 1
            error("likelihood_forward:InvalidExponent", ...
                "The hyper-Laplacian exponent p must satisfy 0 < p < 1.");
        end
        residual = amplitude - measuredAmplitude;
        pointLoss = (residual.^2 + epsilon^2).^(options.p / 2);

    case "huber_intensity"
        residual = intensity - measurement;
        absoluteResidual = abs(residual);
        pointLoss = 0.5 * residual.^2;
        linearRegion = absoluteResidual > options.delta;
        pointLoss(linearRegion) = options.delta * ...
            (absoluteResidual(linearRegion) - 0.5 * options.delta);

    otherwise
        error("likelihood_forward:UnknownModel", ...
            "Unknown likelihood model: %s.", model);
end

cache.residual = residual;
loss = sum(weight .* pointLoss, "all");
end
