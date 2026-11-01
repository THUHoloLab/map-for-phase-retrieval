function gradient = regularization_backward(cache, upstream)
%REGULARIZATION_BACKWARD Differentiate a regularization loss.
%   GRADIENT = REGULARIZATION_BACKWARD(CACHE) returns the complex Euclidean
%   gradient under the convention
%
%       dLOSS = real(sum(conj(GRADIENT) .* dOBJECT, "all")).
%
%   GRADIENT = REGULARIZATION_BACKWARD(CACHE, UPSTREAM) applies an
%   additional real scalar upstream derivative.

arguments
    cache (1,1) struct
    upstream (1,1) double {mustBeReal} = 1
end

switch cache.model
    case "l2"
        gradient = cache.object;

    case "l1_smooth"
        gradient = cache.object ./ cache.magnitude;

    case "gradient_l2"
        gradient = gradient_adjoint(cache.dx, cache.dy);

    case "tv"
        scale = 1 ./ sqrt(cache.gradientEnergy);
        gradient = gradient_adjoint(scale .* cache.dx, scale .* cache.dy);

    case "gradient_hyperlaplacian"
        scale = cache.p * ...
            cache.gradientEnergy.^(cache.p / 2 - 1);
        gradient = gradient_adjoint(scale .* cache.dx, scale .* cache.dy);

    case "hessian_l2"
        gradient = hessian_adjoint(cache.dxx, cache.dyy, cache.dxy);

    case "hessian_tv"
        scale = 1 ./ sqrt(cache.hessianEnergy);
        gradient = hessian_adjoint( ...
            scale .* cache.dxx, scale .* cache.dyy, scale .* cache.dxy);

    otherwise
        error("regularization_backward:UnknownModel", ...
            "Unknown regularization model in CACHE: %s.", cache.model);
end

gradient = upstream * cache.weight .* gradient;
end

function output = gradient_adjoint(dx, dy)
output = circshift(dx, [0, 1]) - dx + ...
    circshift(dy, [1, 0]) - dy;
end

function output = hessian_adjoint(dxx, dyy, dxy)
adjointXX = circshift(dxx, [0, -1]) - 2 * dxx + ...
    circshift(dxx, [0, 1]);
adjointYY = circshift(dyy, [-1, 0]) - 2 * dyy + ...
    circshift(dyy, [1, 0]);
adjointXY = circshift(dxy, [1, 1]) - circshift(dxy, [1, 0]) - ...
    circshift(dxy, [0, 1]) + dxy;
output = adjointXX + adjointYY + 2 * adjointXY;
end
