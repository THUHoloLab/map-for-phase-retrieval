function [objectField, history] = reconstruct_sequential_layers( ...
        data, initialField, options)
%RECONSTRUCT_SEQUENTIAL_LAYERS Multi-distance GS with layer-wise updates.
%   Each measured distance is visited in sequence. The current field is
%   propagated to that plane, its amplitude is replaced by the measured
%   amplitude, and the corrected wave is propagated back to the object
%   plane. This is the standard multi-height projection strategy.
%
%   OPTIONS:
%     iterations  Number of complete distance sweeps (default 100)
%     relaxation  Amplitude-projection relaxation in (0,1] (default 1)
%     shuffle     Randomize layer order each sweep (default false)
%     epsilon     Amplitude stabilizer (default 1e-8)
%     verbose     Print progress (default true)

arguments
    data (1,1) struct
    initialField {mustBeNumeric}
    options.iterations (1,1) double {mustBeInteger, mustBePositive} = 100
    options.relaxation (1,1) double {mustBeInRange(options.relaxation,0,1)} = 1
    options.shuffle (1,1) logical = false
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.verbose (1,1) logical = true
end

validate_data(data, initialField);
objectField = initialField;
numberOfLayers = numel(data.distances);
history.loss = zeros(options.iterations, 1);

for iteration = 1:options.iterations
    order = 1:numberOfLayers;
    if options.shuffle
        order = order(randperm(numberOfLayers));
    end
    for layer = order
        predicted = propagate_angular_spectrum(objectField, ...
            data.distances(layer), data.wavelength, data.pixelSize);
        measuredAmplitude = sqrt(max(data.intensity(:, :, layer), 0));
        projected = measuredAmplitude .* predicted ./ ...
            max(abs(predicted), options.epsilon);
        corrected = predicted + options.relaxation * (projected - predicted);
        objectField = propagate_angular_spectrum(corrected, ...
            -data.distances(layer), data.wavelength, data.pixelSize);
    end
    history.loss(iteration) = amplitude_loss(objectField, data);
    if options.verbose
        fprintf("Sequential layer %3d/%3d, loss %.6e\n", ...
            iteration, options.iterations, history.loss(iteration));
    end
end
end

function loss = amplitude_loss(objectField, data)
loss = 0;
for layer = 1:numel(data.distances)
    predicted = propagate_angular_spectrum(objectField, ...
        data.distances(layer), data.wavelength, data.pixelSize);
    loss = loss + mean((abs(predicted) - ...
        sqrt(max(data.intensity(:, :, layer), 0))).^2, "all");
end
loss = loss / numel(data.distances);
end

function validate_data(data, initialField)
required = ["intensity", "distances", "wavelength", "pixelSize"];
if ~all(isfield(data, required))
    error("reconstruct_sequential_layers:InvalidData", ...
        "DATA must contain intensity, distances, wavelength, and pixelSize.");
end
if ~isequal(size(data.intensity, 1), size(initialField, 1)) || ...
        ~isequal(size(data.intensity, 2), size(initialField, 2))
    error("reconstruct_sequential_layers:SizeMismatch", ...
        "Initial field and measured intensity frames must match in size.");
end
if size(data.intensity, 3) ~= numel(data.distances)
    error("reconstruct_sequential_layers:LayerMismatch", ...
        "The number of intensity frames must equal the number of distances.");
end
end
