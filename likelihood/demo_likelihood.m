%DEMO_LIKELIHOOD Minimal forward and backward phase-retrieval example.

clear;
clc;
rng(1);

objectSize = 32;
object = randn(objectSize) + 1i * randn(objectSize);

% Orthonormal FFT propagation. These two functions are an adjoint pair.
forwardOperator = @(x) fft2(x) / sqrt(numel(x));
adjointOperator = @(y) ifft2(y) * sqrt(numel(y));

cleanDetectorField = forwardOperator(object);
measurement = abs(cleanDetectorField).^2;
initialObject = object + 0.1 * ...
    (randn(size(object)) + 1i * randn(size(object)));

[loss, objectGradient] = likelihood_loss( ...
    initialObject, measurement, forwardOperator, adjointOperator, ...
    "poisson");

fprintf("Poisson loss: %.6e\n", loss);
fprintf("Object-gradient norm: %.6e\n", norm(objectGradient(:)));
