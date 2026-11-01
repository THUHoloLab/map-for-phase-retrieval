%DEMO_MULTIDISTANCE_INLINE_HOLOGRAPHY Compare three reconstruction schemes.

clear;
clc;
rng(5);

pixelSize = 1.12e-6;
wavelength = 532e-9;
imageSize = 96;
distances = [0.4, 0.7, 1.0, 1.3] * 1e-3;

[x, y] = meshgrid(linspace(-1, 1, imageSize));
amplitude = 0.75 + 0.2 * exp(-5 * (x.^2 + y.^2));
amplitude = amplitude + 0.1 * (abs(x) < 0.25 & abs(y) < 0.6);
phase = 1.2 * exp(-10 * ((x + 0.25).^2 + (y - 0.15).^2)) - ...
    0.9 * exp(-12 * ((x - 0.25).^2 + (y + 0.2).^2));
trueField = amplitude .* exp(1i * phase);

data = simulate_multidistance_holography( ...
    trueField, distances, wavelength, pixelSize);
initialField = sqrt(data.intensity(:, :, 1));

[fieldSequential, historySequential] = reconstruct_sequential_layers( ...
    data, initialField, iterations=40, verbose=false);
[fieldGlobal, historyGlobal] = reconstruct_global_backprop( ...
    data, initialField, iterations=80, stepSize=0.3, verbose=false);
[fieldRandom, historyRandom] = reconstruct_random_layer_backprop( ...
    data, initialField, iterations=320, stepSize=0.25, verbose=false);

figure("Name", "Multi-distance inline holography");
tiledlayout(2, 4);
nexttile; imagesc(abs(trueField)); axis image off; title("True amplitude");
nexttile; imagesc(abs(fieldSequential)); axis image off; title("Sequential layers");
nexttile; imagesc(abs(fieldGlobal)); axis image off; title("Global backprop");
nexttile; imagesc(abs(fieldRandom)); axis image off; title("Random layer");
nexttile; imagesc(angle(trueField)); axis image off; title("True phase");
nexttile; imagesc(angle(fieldSequential)); axis image off; title("Sequential phase");
nexttile; imagesc(angle(fieldGlobal)); axis image off; title("Global phase");
nexttile; imagesc(angle(fieldRandom)); axis image off; title("Random phase");
colormap gray;

figure("Name", "Multi-distance losses");
semilogy(historySequential.loss, "LineWidth", 1.4);
hold on;
semilogy(historyGlobal.loss, "LineWidth", 1.4);
semilogy(historyRandom.loss, "LineWidth", 1.4);
grid on;
xlabel("Iteration or sweep");
ylabel("Mean amplitude loss");
legend("Sequential", "Global backprop", "Random layer");
