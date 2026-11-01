function data = simulate_multidistance_holography(objectField, distances, ...
        wavelength, pixelSize, options)
%SIMULATE_MULTIDISTANCE_HOLOGRAPHY Generate inline holograms at many distances.
%   DATA = SIMULATE_MULTIDISTANCE_HOLOGRAPHY(OBJECTFIELD, DISTANCES,
%   WAVELENGTH, PIXELSIZE) propagates OBJECTFIELD to every distance and
%   records the resulting intensity.
%
%   OPTIONS:
%     photonCount  Poisson count scale; Inf disables shot noise (default Inf)
%     readNoise    Additive Gaussian intensity noise (default 0)
%     randomSeed   Random seed for noise generation (default 1)

arguments
    objectField {mustBeNumeric}
    distances (:,1) double {mustBeReal}
    wavelength (1,1) double {mustBePositive}
    pixelSize (1,1) double {mustBePositive}
    options.photonCount (1,1) double {mustBePositive} = inf
    options.readNoise (1,1) double {mustBeNonnegative} = 0
    options.randomSeed (1,1) double {mustBeInteger} = 1
end

rng(options.randomSeed);
numberOfDistances = numel(distances);
intensity = zeros([size(objectField), numberOfDistances], ...
    "like", real(objectField));

for layer = 1:numberOfDistances
    propagatedField = propagate_angular_spectrum( ...
        objectField, distances(layer), wavelength, pixelSize);
    intensity(:, :, layer) = abs(propagatedField).^2;
end

scale = max(intensity, [], "all");
if scale <= 0
    error("simulate_multidistance_holography:ZeroSignal", ...
        "The simulated intensity stack is identically zero.");
end

if isfinite(options.photonCount)
    normalized = intensity / scale;
    counts = poissrnd(options.photonCount * normalized);
    intensity = scale * counts / options.photonCount;
end
if options.readNoise > 0
    intensity = max(intensity + scale * options.readNoise * ...
        randn(size(intensity), "like", intensity), 0);
end

data = struct("intensity", intensity, ...
    "distances", distances, ...
    "wavelength", wavelength, ...
    "pixelSize", pixelSize, ...
    "objectSize", size(objectField));
end
