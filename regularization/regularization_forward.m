function [loss, cache] = regularization_forward(object, model, options)
%REGULARIZATION_FORWARD Evaluate a prior for a complex-valued object.
%   [LOSS, CACHE] = REGULARIZATION_FORWARD(OBJECT, MODEL, OPTIONS) returns
%   the scalar regularization loss and the intermediates required by
%   REGULARIZATION_BACKWARD.
%
%   Supported MODEL values:
%     "l2"                     Object-domain Tikhonov penalty
%     "l1_smooth"              Smooth approximation of object L1 sparsity
%     "gradient_l2"            First-order quadratic smoothness
%     "tv"                     Isotropic total variation
%     "gradient_hyperlaplacian"  Gradient Lp prior, 0 < p < 1
%     "hessian_l2"             Second-order quadratic smoothness
%     "hessian_tv"             Isotropic second-order total variation
%
%   OPTIONS can contain:
%     epsilon   Positive smoothing constant (default: 1e-8)
%     p         Hyper-Laplacian exponent (default: 0.8)
%     weight    Scalar nonnegative regularization weight (default: 1)
%
%   Periodic boundary conditions are used so the finite differences and
%   their adjoints are exact and compatible with FFT-based solvers.

arguments
    object {mustBeNumeric}
    model {mustBeTextScalar}
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.p (1,1) double {mustBePositive} = 0.8
    options.weight (1,1) double {mustBeNonnegative} = 1
end

model = lower(string(model));
epsilon = options.epsilon;
weight = options.weight;

cache = struct( ...
    "object", object, ...
    "model", model, ...
    "epsilon", epsilon, ...
    "p", options.p, ...
    "weight", weight);

switch model
    case "l2"
        pointLoss = 0.5 * abs(object).^2;

    case "l1_smooth"
        magnitude = sqrt(abs(object).^2 + epsilon^2);
        pointLoss = magnitude;
        cache.magnitude = magnitude;

    case {"gradient_l2", "tv", "gradient_hyperlaplacian"}
        [dx, dy] = forward_gradient(object);
        gradientEnergy = abs(dx).^2 + abs(dy).^2 + epsilon^2;
        cache.dx = dx;
        cache.dy = dy;
        cache.gradientEnergy = gradientEnergy;

        switch model
            case "gradient_l2"
                pointLoss = 0.5 * (abs(dx).^2 + abs(dy).^2);
            case "tv"
                pointLoss = sqrt(gradientEnergy);
            case "gradient_hyperlaplacian"
                if options.p >= 1
                    error("regularization_forward:InvalidExponent", ...
                        "The hyper-Laplacian exponent must satisfy 0 < p < 1.");
                end
                pointLoss = gradientEnergy.^(options.p / 2);
        end

    case {"hessian_l2", "hessian_tv"}
        [dxx, dyy, dxy] = forward_hessian(object);
        hessianEnergy = abs(dxx).^2 + abs(dyy).^2 + ...
            2 * abs(dxy).^2 + epsilon^2;
        cache.dxx = dxx;
        cache.dyy = dyy;
        cache.dxy = dxy;
        cache.hessianEnergy = hessianEnergy;

        if model == "hessian_l2"
            pointLoss = 0.5 * (hessianEnergy - epsilon^2);
        else
            pointLoss = sqrt(hessianEnergy);
        end

    otherwise
        error("regularization_forward:UnknownModel", ...
            "Unknown regularization model: %s.", model);
end

loss = weight * sum(pointLoss, "all");
end

function [dx, dy] = forward_gradient(object)
dx = circshift(object, [0, -1]) - object;
dy = circshift(object, [-1, 0]) - object;
end

function [dxx, dyy, dxy] = forward_hessian(object)
dxx = circshift(object, [0, -1]) - 2 * object + ...
    circshift(object, [0, 1]);
dyy = circshift(object, [-1, 0]) - 2 * object + ...
    circshift(object, [1, 0]);
dxy = circshift(object, [-1, -1]) - circshift(object, [-1, 0]) - ...
    circshift(object, [0, -1]) + object;
end
