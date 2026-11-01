function [objectField, history] = reconstruct_global_backprop( ...
        data, initialField, options)
%RECONSTRUCT_GLOBAL_BACKPROP Optimize all distances with global feedback.
%   At every iteration, all predicted holograms are formed. Their amplitude
%   residuals are backpropagated to the object plane and averaged before
%   one global field update. This uses every distance simultaneously.
%
%   OPTIONS:
%     iterations  Number of gradient steps (default 200)
%     stepSize    Complex-field update step (default 0.5)
%     epsilon     Amplitude stabilizer (default 1e-8)
%     verbose     Print progress (default true)

arguments
    data (1,1) struct
    initialField {mustBeNumeric}
    options.iterations (1,1) double {mustBeInteger, mustBePositive} = 200
    options.stepSize (1,1) double {mustBePositive} = 0.5
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.verbose (1,1) logical = true
end

validate_data(data, initialField);
objectField = initialField;
numberOfLayers = numel(data.distances);
history.loss = zeros(options.iterations, 1);

for iteration = 1:options.iterations
    correction = zeros(size(objectField), "like", objectField);
    totalLoss = 0;
    for layer = 1:numberOfLayers
        predicted = propagate_angular_spectrum(objectField, ...
            data.distances(layer), data.wavelength, data.pixelSize);
        measuredAmplitude = sqrt(max(data.intensity(:, :, layer), 0));
        residual = abs(predicted) - measuredAmplitude;
        totalLoss = totalLoss + mean(residual.^2, "all");

        detectorGradient = residual .* predicted ./ ...
            max(abs(predicted), options.epsilon);
        correction = correction + propagate_angular_spectrum( ...
            detectorGradient, -data.distances(layer), ...
            data.wavelength, data.pixelSize);
    end
    correction = correction / numberOfLayers;
    objectField = objectField - options.stepSize * correction;
    history.loss(iteration) = totalLoss / numberOfLayers;

    if options.verbose
        fprintf("Global backprop %3d/%3d, loss %.6e\n", ...
            iteration, options.iterations, history.loss(iteration));
    end
end
end

function validate_data(data, initialField)
if ~isfield(data, "intensity") || ~isfield(data, "distances")
    error("reconstruct_global_backprop:InvalidData", ...
        "DATA must contain intensity and distances.");
end
if ~isequal(size(data.intensity, 1), size(initialField, 1)) || ...
        ~isequal(size(data.intensity, 2), size(initialField, 2)) || ...
        size(data.intensity, 3) ~= numel(data.distances)
    error("reconstruct_global_backprop:SizeMismatch", ...
        "Field and multi-distance data dimensions are inconsistent.");
end
end
