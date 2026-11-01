function [object, probe, history] = ptycho_reconstruct_rpie( ...
        data, initialObject, initialProbe, options)
%PTYCHO_RECONSTRUCT_RPIE Reconstruct object and probe with rPIE updates.
%   [OBJECT, PROBE, HISTORY] = PTYCHO_RECONSTRUCT_RPIE(DATA,
%   INITIALOBJECT, INITIALPROBE, OPTIONS) applies sequential amplitude
%   projection and regularized PIE updates.
%
%   The implementation follows the standard rPIE structure: each exit wave
%   is projected onto the measured Fourier amplitude, then the resulting
%   correction is divided by a blend of local and maximum illumination
%   power. Setting alphaObject=alphaProbe=1 gives ePIE-like normalization.
%
%   OPTIONS:
%     epochs          Number of complete scan passes (default: 100)
%     objectStep      Object update step (default: 0.8)
%     probeStep       Probe update step (default: 0.2)
%     alphaObject     rPIE object regularizer in [0,1] (default: 0.1)
%     alphaProbe      rPIE probe regularizer in [0,1] (default: 0.1)
%     updateProbe     Whether to refine the probe (default: true)
%     shuffle         Randomize frame order each epoch (default: true)
%     objectMagnitude Maximum allowed object magnitude (default: Inf)
%     epsilon         Numerical stabilizer (default: 1e-8)
%     verbose         Print epoch losses (default: true)

arguments
    data (1,1) struct
    initialObject {mustBeNumeric}
    initialProbe {mustBeNumeric}
    options.epochs (1,1) double {mustBeInteger, mustBePositive} = 100
    options.objectStep (1,1) double {mustBePositive} = 0.8
    options.probeStep (1,1) double {mustBePositive} = 0.2
    options.alphaObject (1,1) double {mustBeInRange(options.alphaObject,0,1)} = 0.1
    options.alphaProbe (1,1) double {mustBeInRange(options.alphaProbe,0,1)} = 0.1
    options.updateProbe (1,1) logical = true
    options.shuffle (1,1) logical = true
    options.objectMagnitude (1,1) double {mustBePositive} = inf
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.verbose (1,1) logical = true
end

validate_data(data, initialObject, initialProbe);
object = initialObject;
probe = initialProbe;
numberOfFrames = size(data.positions, 1);
history.loss = zeros(options.epochs, 1);

for epoch = 1:options.epochs
    if options.shuffle
        frameOrder = randperm(numberOfFrames);
    else
        frameOrder = 1:numberOfFrames;
    end

    epochLoss = 0;
    for frame = frameOrder
        rows = data.positions(frame, 1) + (0:size(probe, 1)-1);
        columns = data.positions(frame, 2) + (0:size(probe, 2)-1);
        objectPatch = object(rows, columns);

        exitWave = objectPatch .* probe;
        detectorWave = fftshift(fft2(exitWave));
        measuredAmplitude = data.amplitude(:, :, frame);
        epochLoss = epochLoss + mean( ...
            (abs(detectorWave) - measuredAmplitude).^2, "all");

        projectedDetectorWave = measuredAmplitude .* ...
            detectorWave ./ max(abs(detectorWave), options.epsilon);
        projectedExitWave = ifft2(ifftshift(projectedDetectorWave));
        exitCorrection = projectedExitWave - exitWave;

        probePower = abs(probe).^2;
        objectDenominator = (1 - options.alphaObject) * probePower + ...
            options.alphaObject * max(probePower, [], "all") + ...
            options.epsilon;
        object(rows, columns) = objectPatch + options.objectStep * ...
            conj(probe) .* exitCorrection ./ objectDenominator;

        if options.updateProbe
            objectPower = abs(objectPatch).^2;
            probeDenominator = (1 - options.alphaProbe) * objectPower + ...
                options.alphaProbe * max(objectPower, [], "all") + ...
                options.epsilon;
            probe = probe + options.probeStep * conj(objectPatch) .* ...
                exitCorrection ./ probeDenominator;
        end
    end

    if isfinite(options.objectMagnitude)
        magnitude = abs(object);
        object = min(magnitude, options.objectMagnitude) .* ...
            object ./ max(magnitude, options.epsilon);
    end

    history.loss(epoch) = epochLoss / numberOfFrames;
    if options.verbose
        fprintf("rPIE epoch %3d/%3d, amplitude MSE %.6e\n", ...
            epoch, options.epochs, history.loss(epoch));
    end
end
end

function validate_data(data, object, probe)
requiredFields = ["amplitude", "positions"];
if ~all(isfield(data, requiredFields))
    error("ptycho_reconstruct_rpie:InvalidData", ...
        "DATA must contain amplitude and positions.");
end
if ~isequal(size(data.amplitude, 1), size(probe, 1)) || ...
        ~isequal(size(data.amplitude, 2), size(probe, 2))
    error("ptycho_reconstruct_rpie:ProbeSizeMismatch", ...
        "The probe size must match each diffraction frame.");
end
lastRows = data.positions(:, 1) + size(probe, 1) - 1;
lastColumns = data.positions(:, 2) + size(probe, 2) - 1;
if any(lastRows > size(object, 1)) || any(lastColumns > size(object, 2))
    error("ptycho_reconstruct_rpie:PositionOutsideObject", ...
        "Every scan window must lie inside INITIALOBJECT.");
end
end
