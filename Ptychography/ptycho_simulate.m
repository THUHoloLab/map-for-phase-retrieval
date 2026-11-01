function data = ptycho_simulate(object, probe, positions, options)
%PTYCHO_SIMULATE Generate far-field ptychographic measurements.
%   DATA = PTYCHO_SIMULATE(OBJECT, PROBE, POSITIONS, OPTIONS) crops the
%   object at each scan position, multiplies it by the complex probe, and
%   propagates the exit wave to the detector with a centered 2-D FFT.
%
%   POSITIONS is an N-by-2 matrix of one-based [row, column] coordinates
%   for the top-left corner of each illuminated object patch.
%
%   OPTIONS can contain:
%     photonCount  Mean photon scale. Inf disables Poisson noise.
%                  (default: Inf)
%     readNoise    Gaussian detector-noise standard deviation in intensity
%                  units after normalization (default: 0)
%     randomSeed   Random seed used for noise generation (default: 1)
%
%   DATA contains measured amplitude, intensity, positions, and dimensions.

arguments
    object {mustBeNumeric}
    probe {mustBeNumeric}
    positions (:,2) double {mustBeInteger, mustBePositive}
    options.photonCount (1,1) double {mustBePositive} = inf
    options.readNoise (1,1) double {mustBeNonnegative} = 0
    options.randomSeed (1,1) double {mustBeInteger} = 1
end

validate_positions(size(object), size(probe), positions);
rng(options.randomSeed);

numberOfFrames = size(positions, 1);
intensity = zeros([size(probe), numberOfFrames], "like", real(object));

for frame = 1:numberOfFrames
    rows = positions(frame, 1) + (0:size(probe, 1)-1);
    columns = positions(frame, 2) + (0:size(probe, 2)-1);
    exitWave = object(rows, columns) .* probe;
    detectorWave = fftshift(fft2(exitWave));
    intensity(:, :, frame) = abs(detectorWave).^2;
end

normalization = max(intensity, [], "all");
if normalization == 0
    error("ptycho_simulate:ZeroSignal", ...
        "The simulated detector intensity is identically zero.");
end

if isfinite(options.photonCount)
    normalizedIntensity = intensity / normalization;
    noisyCounts = poissrnd(options.photonCount * normalizedIntensity);
    intensity = normalization * noisyCounts / options.photonCount;
end

if options.readNoise > 0
    intensity = intensity + normalization * options.readNoise * ...
        randn(size(intensity), "like", intensity);
    intensity = max(intensity, 0);
end

data = struct( ...
    "amplitude", sqrt(intensity), ...
    "intensity", intensity, ...
    "positions", positions, ...
    "objectSize", size(object), ...
    "probeSize", size(probe));
end

function validate_positions(objectSize, probeSize, positions)
lastRows = positions(:, 1) + probeSize(1) - 1;
lastColumns = positions(:, 2) + probeSize(2) - 1;
if any(lastRows > objectSize(1)) || any(lastColumns > objectSize(2))
    error("ptycho_simulate:PositionOutsideObject", ...
        "Every scan window must lie inside the object array.");
end
end
