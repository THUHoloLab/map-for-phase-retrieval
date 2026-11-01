function output = regularization_prox(input, model, threshold)
%REGULARIZATION_PROX Apply a sparsity-promoting proximal operator.
%   OUTPUT = REGULARIZATION_PROX(INPUT, MODEL, THRESHOLD) supports:
%
%     "l1"   Complex soft thresholding, prox of THRESHOLD * ||x||_1
%     "l0"   Complex hard thresholding for an L0-style penalty
%
%   The complex phase is preserved for coefficients that remain nonzero.

arguments
    input {mustBeNumeric}
    model {mustBeTextScalar}
    threshold (1,1) double {mustBeNonnegative}
end

magnitude = abs(input);
phase = input ./ max(magnitude, eps(class(input)));

switch lower(string(model))
    case "l1"
        output = max(magnitude - threshold, 0) .* phase;
    case "l0"
        output = input .* (magnitude > sqrt(2 * threshold));
    otherwise
        error("regularization_prox:UnknownModel", ...
            "Unknown proximal model: %s.", model);
end
end
