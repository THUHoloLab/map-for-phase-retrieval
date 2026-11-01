function output = regularization_project(input, constraint, options)
%REGULARIZATION_PROJECT Project a wavefield onto a physical constraint.
%   OUTPUT = REGULARIZATION_PROJECT(INPUT, CONSTRAINT, OPTIONS) supports:
%
%     "support"          Set samples outside OPTIONS.mask to zero
%     "magnitude_bounds" Clip magnitude to [OPTIONS.minimum, OPTIONS.maximum]
%     "phase_only"       Set magnitude to OPTIONS.amplitude
%     "real_nonnegative" Project onto real, nonnegative values

arguments
    input {mustBeNumeric}
    constraint {mustBeTextScalar}
    options.mask = true(size(input))
    options.minimum {mustBeNumeric, mustBeReal, mustBeNonnegative} = 0
    options.maximum {mustBeNumeric, mustBeReal, mustBeNonnegative} = inf
    options.amplitude {mustBeNumeric, mustBeReal, mustBeNonnegative} = 1
end

magnitude = abs(input);
phase = input ./ max(magnitude, eps(class(input)));

switch lower(string(constraint))
    case "support"
        if ~isequal(size(options.mask), size(input))
            error("regularization_project:MaskSizeMismatch", ...
                "OPTIONS.mask must have the same size as INPUT.");
        end
        output = input .* options.mask;

    case "magnitude_bounds"
        if any(options.minimum > options.maximum, "all")
            error("regularization_project:InvalidBounds", ...
                "OPTIONS.minimum cannot exceed OPTIONS.maximum.");
        end
        clippedMagnitude = min(max(magnitude, options.minimum), ...
            options.maximum);
        output = clippedMagnitude .* phase;

    case "phase_only"
        output = options.amplitude .* phase;

    case "real_nonnegative"
        output = max(real(input), 0);

    otherwise
        error("regularization_project:UnknownConstraint", ...
            "Unknown constraint: %s.", constraint);
end
end
