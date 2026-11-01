%DEMO_PTYCHOGRAPHY Compare rPIE and automatic differentiation.

clear;
clc;
rng(4);

objectSize = 96;
probeSize = 32;
scanStep = 12;

% Construct a smooth complex object without external image files.
[xObject, yObject] = meshgrid(linspace(-1, 1, objectSize));
amplitude = 0.35 + 0.55 * exp(-3 * (xObject.^2 + yObject.^2));
amplitude = amplitude + 0.18 * (abs(xObject) < 0.45 & abs(yObject) < 0.12);
phase = 1.2 * exp(-8 * ((xObject + 0.25).^2 + (yObject - 0.2).^2));
phase = phase - 0.8 * exp(-10 * ((xObject - 0.3).^2 + (yObject + 0.25).^2));
trueObject = amplitude .* exp(1i * phase);

% Use a compact Gaussian probe with mild quadratic phase curvature.
[xProbe, yProbe] = meshgrid(linspace(-1, 1, probeSize));
trueProbe = exp(-4 * (xProbe.^2 + yProbe.^2)) .* ...
    exp(1i * 0.5 * (xProbe.^2 + yProbe.^2));

scanCoordinates = 1:scanStep:(objectSize - probeSize + 1);
[scanColumns, scanRows] = meshgrid(scanCoordinates);
positions = [scanRows(:), scanColumns(:)];
data = ptycho_simulate(trueObject, trueProbe, positions);

initialObject = ones(objectSize, "single");
initialProbe = exp(-3 * (xProbe.^2 + yProbe.^2));

[objectRpie, probeRpie, historyRpie] = ptycho_reconstruct_rpie( ...
    data, initialObject, initialProbe, epochs=40, verbose=false);

[objectAd, probeAd, historyAd] = ptycho_reconstruct_autodiff( ...
    data, initialObject, initialProbe, epochs=40, batchSize=8, ...
    objectLearningRate=0.03, probeLearningRate=0.01, verbose=false);

figure("Name", "Ptychography reconstruction");
tiledlayout(2, 3);
nexttile; imagesc(abs(trueObject)); axis image off; title("True amplitude");
nexttile; imagesc(abs(objectRpie)); axis image off; title("rPIE amplitude");
nexttile; imagesc(abs(objectAd)); axis image off; title("AD amplitude");
nexttile; imagesc(abs(trueProbe)); axis image off; title("True probe");
nexttile; imagesc(abs(probeRpie)); axis image off; title("rPIE probe");
nexttile; imagesc(abs(probeAd)); axis image off; title("AD probe");
colormap gray;

figure("Name", "Ptychography loss");
semilogy(historyRpie.loss, "LineWidth", 1.5);
hold on;
semilogy(historyAd.loss, "LineWidth", 1.5);
xlabel("Epoch");
ylabel("Loss");
legend("rPIE amplitude MSE", "AD objective");
grid on;
