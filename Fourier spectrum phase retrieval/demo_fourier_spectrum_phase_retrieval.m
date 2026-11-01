%DEMO_FOURIER_SPECTRUM_PHASE_RETRIEVAL Compare HIO and Difference Map.

clear;
clc;
rng(2);

imageSize = 256; padsize = 64;
img1 = imresize(single(imread('cameraman.tif'))/255,[imageSize,imageSize]);
img2 = imresize(mean(single(imread('peppers.png'))/255,3),[imageSize,imageSize]);

[x, y] = meshgrid(1:imageSize, 1:imageSize);
support = (x - imageSize / 2).^2 + (y - imageSize / 2).^2 < ...
    (0.38 * imageSize)^2;

support = padarray(support,[padsize,padsize],'both');
img1 = padarray(img1,[padsize,padsize],'both');
img2 = padarray(img2,[padsize,padsize],'both');
% Simple compact object with amplitude and phase structure.
trueObject = support .*img1 .* ...
    exp(1i * 0.8 * img2);
fourierAmplitude = abs(fft2(trueObject));

options.object = support .* exp(1i * 2 * pi * rand(size(img1)));
options.iterations = 600;

[hioObject, hioError] = hio_phase_retrieval( ...
    fourierAmplitude, support, options);
[dmObject, dmError] = difference_map_phase_retrieval( ...
    fourierAmplitude, support, options);

figure("Name", "Fourier spectrum phase retrieval");
tiledlayout(2, 4);
nexttile; imagesc(abs(trueObject)); axis image off; title("True amplitude");
nexttile; imagesc(abs(hioObject)); axis image off; title("HIO amplitude");
nexttile; imagesc(abs(dmObject)); axis image off; title("DM amplitude");
nexttile; imagesc(log(1 + fourierAmplitude)); axis image off;
title("Measured Fourier amplitude");
nexttile; imagesc(angle(trueObject)); axis image off; title("True phase");
nexttile; imagesc(angle(hioObject)); axis image off; title("HIO phase");
nexttile; imagesc(angle(dmObject)); axis image off; title("DM phase");
nexttile;
semilogy(hioError, "LineWidth", 1.2);
hold on;
semilogy(dmError, "LineWidth", 1.2);
grid on;
xlabel("Iteration");
ylabel("Fourier amplitude error");
legend("HIO", "Difference Map");
colormap gray;
