function [objectField, history] = reconstruct_random_layer_backprop( ...
        data, initialField, options)
%RECONSTRUCT_RANDOM_LAYER_BACKPROP Use stochastic single-layer feedback.
%   Each iteration randomly selects one measured distance, computes its
%   amplitude residual, backpropagates that residual, and updates the object
%   field. This stochastic strategy reduces memory and per-step cost.
%
%   OPTIONS:
%     iterations  Number of random layer updates (default 500)
%     stepSize    Field update step (default 0.5)
%     randomSeed  Reproducible random seed (default 1)
%     epsilon     Amplitude stabilizer (default 1e-8)
%     verbose     Print progress every 50 updates (default true)

arguments
    data (1,1) struct
    initialField {mustBeNumeric}
    options.iterations (1,1) double {mustBeInteger, mustBePositive} = 500
    options.stepSize (1,1) double {mustBePositive} = 0.5
    options.randomSeed (1,1) double {mustBeInteger} = 1
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.verbose (1,1) logical = true
end

if ~isfield(data, "intensity") || ~isfield(data, "distances")
    error("reconstruct_random_layer_backprop:InvalidData", ...
        "DATA must contain intensity and distances.");
end
rng(options.randomSeed);
objectField = initialField;
numberOfLayers = numel(data.distances);
history.loss = zeros(options.iterations, 1);

for iteration = 1:options.iterations
    layer = randi(numberOfLayers);
    predicted = propagate_angular_spectrum(objectField, ...
        data.distances(layer), data.wavelength, data.pixelSize);
    measuredAmplitude = sqrt(max(data.intensity(:, :, layer), 0));
    residual = abs(predicted) - measuredAmplitude;
    detectorGradient = residual .* predicted ./ ...
        max(abs(predicted), options.epsilon);
    correction = propagate_angular_spectrum(detectorGradient, ...
        -data.distances(layer), data.wavelength, data.pixelSize);
    objectField = objectField - options.stepSize * correction;

    history.loss(iteration) = mean(residual.^2, "all");
    if options.verbose && (iteration == 1 || mod(iteration, 50) == 0)
        fprintf("Random layer %3d/%3d, layer %d, loss %.6e\n", ...
            iteration, options.iterations, layer, history.loss(iteration));
    end
end
end
