function test_ptychography
%TEST_PTYCHOGRAPHY Run a compact reconstruction smoke test.

rng(9);
objectSize = 32;
probeSize = 12;
[x, y] = meshgrid(linspace(-1, 1, objectSize));
trueObject = (0.5 + 0.4 * exp(-4 * (x.^2 + y.^2))) .* ...
    exp(1i * 0.5 * x);

[xp, yp] = meshgrid(linspace(-1, 1, probeSize));
trueProbe = exp(-5 * (xp.^2 + yp.^2));
coordinates = [1, 7, 13, 19, 21];
[columns, rows] = meshgrid(coordinates);
positions = [rows(:), columns(:)];
data = ptycho_simulate(trueObject, trueProbe, positions);

initialObject = ones(objectSize, "single");
initialProbe = exp(-4 * (xp.^2 + yp.^2));

check_autodiff_gradient(data, initialObject, initialProbe);

[~, ~, rpieHistory] = ptycho_reconstruct_rpie( ...
    data, initialObject, initialProbe, epochs=3, verbose=false, ...
    shuffle=false);
assert(all(isfinite(rpieHistory.loss)));
assert(rpieHistory.loss(end) < rpieHistory.loss(1));

[objectAd, probeAd, adHistory] = ptycho_reconstruct_autodiff( ...
    data, initialObject, initialProbe, epochs=3, batchSize=5, ...
    objectLearningRate=0.01, probeLearningRate=0.005, ...
    verbose=false, shuffle=false);
assert(all(isfinite(adHistory.loss)));
assert(all(isfinite(objectAd), "all"));
assert(all(isfinite(probeAd), "all"));

fprintf("rPIE loss: %.3e -> %.3e\n", ...
    rpieHistory.loss(1), rpieHistory.loss(end));
fprintf("AD loss:   %.3e -> %.3e\n", ...
    adHistory.loss(1), adHistory.loss(end));
fprintf("Ptychography smoke tests passed.\n");
end

function check_autodiff_gradient(data, object, probe)
operator = PtychoAmplitudeOperator;
object = complex(single(object));
probe = complex(single(probe));
objectDirection = complex(randn(size(object), "single"), ...
    randn(size(object), "single"));
probeDirection = complex(randn(size(probe), "single"), ...
    randn(size(probe), "single"));

directionNorm = sqrt(sum(abs(objectDirection).^2, "all") + ...
    sum(abs(probeDirection).^2, "all"));
objectDirection = objectDirection / directionNorm;
probeDirection = probeDirection / directionNorm;

[~, objectGradient, probeGradient] = dlfeval( ...
    @autodiff_loss, operator, dlarray(object), dlarray(probe), ...
    data.positions, single(data.amplitude));
objectGradient = extractdata(objectGradient);
probeGradient = extractdata(probeGradient);
analyticDerivative = real(sum(conj(objectGradient) .* ...
    objectDirection, "all") + sum(conj(probeGradient) .* ...
    probeDirection, "all"));

step = 1e-3;
lossPlus = plain_amplitude_loss( ...
    object + step * objectDirection, probe + step * probeDirection, data);
lossMinus = plain_amplitude_loss( ...
    object - step * objectDirection, probe - step * probeDirection, data);
numericDerivative = (lossPlus - lossMinus) / (2 * step);
relativeError = abs(analyticDerivative - numericDerivative) / ...
    max([1, abs(analyticDerivative), abs(numericDerivative)]);
assert(relativeError < 2e-3, ...
    "Automatic-differentiation gradient error is %.3e.", relativeError);
fprintf("AD gradient relative error: %.3e\n", relativeError);
end

function [loss, objectGradient, probeGradient] = autodiff_loss( ...
        operator, object, probe, positions, measuredAmplitude)
prediction = operator(object, probe, positions);
loss = mean((prediction - measuredAmplitude).^2, "all");
[objectGradient, probeGradient] = dlgradient(loss, object, probe);
end

function loss = plain_amplitude_loss(object, probe, data)
loss = 0;
for frame = 1:size(data.positions, 1)
    rows = data.positions(frame, 1) + (0:size(probe, 1)-1);
    columns = data.positions(frame, 2) + (0:size(probe, 2)-1);
    prediction = abs(fftshift(fft2(object(rows, columns) .* probe)));
    loss = loss + mean( ...
        (prediction - data.amplitude(:, :, frame)).^2, "all");
end
loss = loss / size(data.positions, 1);
end
