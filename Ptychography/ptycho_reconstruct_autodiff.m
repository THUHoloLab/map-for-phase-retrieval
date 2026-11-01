function [object, probe, history] = ptycho_reconstruct_autodiff( ...
        data, initialObject, initialProbe, options)
%PTYCHO_RECONSTRUCT_AUTODIFF End-to-end differentiable reconstruction.
%   This solver builds the ptychographic forward model with native MATLAB
%   dlarray operations and obtains complex object/probe gradients using
%   dlgradient. Adam updates are applied to both unknown wavefields.
%
%   OPTIONS:
%     epochs              Number of complete scan passes (default: 100)
%     batchSize           Frames per gradient update (default: 8)
%     objectLearningRate  Adam learning rate for object (default: 0.03)
%     probeLearningRate   Adam learning rate for probe (default: 0.01)
%     beta1, beta2        Adam moment factors (default: 0.9, 0.999)
%     epsilon             Adam/amplitude stabilizer (default: 1e-8)
%     updateProbe         Whether to optimize the probe (default: true)
%     useGPU              Move arrays to a supported GPU (default: false)
%     shuffle             Randomize batches each epoch (default: true)
%     tvWeight            Optional smooth TV weight on object (default: 0)
%     verbose             Print epoch losses (default: true)

arguments
    data (1,1) struct
    initialObject {mustBeNumeric}
    initialProbe {mustBeNumeric}
    options.epochs (1,1) double {mustBeInteger, mustBePositive} = 100
    options.batchSize (1,1) double {mustBeInteger, mustBePositive} = 8
    options.objectLearningRate (1,1) double {mustBePositive} = 0.03
    options.probeLearningRate (1,1) double {mustBePositive} = 0.01
    options.beta1 (1,1) double {mustBeInRange(options.beta1,0,1)} = 0.9
    options.beta2 (1,1) double {mustBeInRange(options.beta2,0,1)} = 0.999
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.updateProbe (1,1) logical = true
    options.useGPU (1,1) logical = false
    options.shuffle (1,1) logical = true
    options.tvWeight (1,1) double {mustBeNonnegative} = 0
    options.verbose (1,1) logical = true
end

validate_autodiff_inputs(data, initialObject, initialProbe);
if options.useGPU
    initialObject = gpuArray(single(initialObject));
    initialProbe = gpuArray(single(initialProbe));
    measuredAmplitude = gpuArray(single(data.amplitude));
else
    initialObject = single(initialObject);
    initialProbe = single(initialProbe);
    measuredAmplitude = single(data.amplitude);
end

object = dlarray(complex(initialObject));
probe = dlarray(complex(initialProbe));
objectMoment1 = [];
objectMoment2 = [];
probeMoment1 = [];
probeMoment2 = [];
iteration = 0;
numberOfFrames = size(data.positions, 1);
history.loss = zeros(options.epochs, 1);
forwardOperator = PtychoAmplitudeOperator;

for epoch = 1:options.epochs
    if options.shuffle
        frameOrder = randperm(numberOfFrames);
    else
        frameOrder = 1:numberOfFrames;
    end

    epochLoss = 0;
    processedFrames = 0;
    for firstFrame = 1:options.batchSize:numberOfFrames
        batchIndices = frameOrder(firstFrame:min( ...
            firstFrame + options.batchSize - 1, numberOfFrames));
        batchPositions = data.positions(batchIndices, :);
        batchAmplitude = measuredAmplitude(:, :, batchIndices);

        [loss, objectGradient, probeGradient] = dlfeval( ...
            @model_loss, forwardOperator, object, probe, ...
            batchPositions, batchAmplitude, ...
            options.tvWeight, options.epsilon, options.updateProbe);

        iteration = iteration + 1;
        [object, objectMoment1, objectMoment2] = complex_adam_update( ...
            object, objectGradient, objectMoment1, objectMoment2, ...
            iteration, options.objectLearningRate, options);

        if options.updateProbe
            [probe, probeMoment1, probeMoment2] = complex_adam_update( ...
                probe, probeGradient, probeMoment1, probeMoment2, ...
                iteration, options.probeLearningRate, options);
        end

        batchCount = numel(batchIndices);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * batchCount;
        processedFrames = processedFrames + batchCount;
    end

    history.loss(epoch) = epochLoss / processedFrames;
    if options.verbose
        fprintf("AD epoch   %3d/%3d, objective %.6e\n", ...
            epoch, options.epochs, history.loss(epoch));
    end
end

object = gather(extractdata(object));
probe = gather(extractdata(probe));
end

function [loss, objectGradient, probeGradient] = model_loss( ...
        forwardOperator, object, probe, positions, measuredAmplitude, ...
        tvWeight, epsilon, updateProbe)
predictedAmplitude = forwardOperator(object, probe, positions);
residual = predictedAmplitude - measuredAmplitude;
loss = mean(residual.^2, "all");

if tvWeight > 0
    dx = object(:, [2:end, 1]) - object;
    dy = object([2:end, 1], :) - object;
    loss = loss + tvWeight * mean( ...
        sqrt(abs(dx).^2 + abs(dy).^2 + epsilon^2), "all");
end

if updateProbe
    [objectGradient, probeGradient] = dlgradient(loss, object, probe);
else
    objectGradient = dlgradient(loss, object);
    probeGradient = [];
end
end

function [parameter, moment1, moment2] = complex_adam_update( ...
        parameter, gradient, moment1, moment2, iteration, learningRate, options)
if isempty(moment1)
    moment1 = zeros(size(gradient), "like", gradient);
    moment2 = zeros(size(real(extractdata(gradient))), ...
        "like", real(extractdata(gradient)));
end

moment1 = options.beta1 * moment1 + (1 - options.beta1) * gradient;
gradientPower = abs(gradient).^2;
moment2 = options.beta2 * moment2 + ...
    (1 - options.beta2) * gradientPower;

correctedMoment1 = moment1 / (1 - options.beta1^iteration);
correctedMoment2 = moment2 / (1 - options.beta2^iteration);
parameter = parameter - learningRate * correctedMoment1 ./ ...
    (sqrt(correctedMoment2) + options.epsilon);
end

function validate_autodiff_inputs(data, object, probe)
if ~isfield(data, "amplitude") || ~isfield(data, "positions")
    error("ptycho_reconstruct_autodiff:InvalidData", ...
        "DATA must contain amplitude and positions.");
end
if size(data.amplitude, 1) ~= size(probe, 1) || ...
        size(data.amplitude, 2) ~= size(probe, 2)
    error("ptycho_reconstruct_autodiff:ProbeSizeMismatch", ...
        "The probe size must match each diffraction frame.");
end
lastRows = data.positions(:, 1) + size(probe, 1) - 1;
lastColumns = data.positions(:, 2) + size(probe, 2) - 1;
if any(lastRows > size(object, 1)) || any(lastColumns > size(object, 2))
    error("ptycho_reconstruct_autodiff:PositionOutsideObject", ...
        "Every scan window must lie inside INITIALOBJECT.");
end
end
