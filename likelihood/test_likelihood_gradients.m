function test_likelihood_gradients
%TEST_LIKELIHOOD_GRADIENTS Check all analytic gradients numerically.

rng(7);
field = randn(5, 4) + 1i * randn(5, 4);
measurement = 0.2 + 3 * rand(5, 4);
direction = randn(size(field)) + 1i * randn(size(field));
direction = direction / norm(direction(:));

models = [ ...
    "gaussian_intensity", "gaussian_amplitude", "gaussian_gamma", ...
    "poisson", "anscombe", "poisson_gaussian", ...
    "laplace_amplitude", "hyper_laplacian", "huber_intensity"];

step = 1e-6;
tolerance = 2e-4;

for model = models
    [~, cache] = likelihood_forward(field, measurement, model, ...
        gamma=0.7, p=0.8, delta=0.6);
    gradient = likelihood_backward(cache);
    analyticDerivative = real(sum(conj(gradient) .* direction, "all"));

    lossPlus = likelihood_forward( ...
        field + step * direction, measurement, model, ...
        gamma=0.7, p=0.8, delta=0.6);
    lossMinus = likelihood_forward( ...
        field - step * direction, measurement, model, ...
        gamma=0.7, p=0.8, delta=0.6);
    numericDerivative = (lossPlus - lossMinus) / (2 * step);

    relativeError = abs(analyticDerivative - numericDerivative) / ...
        max([1, abs(analyticDerivative), abs(numericDerivative)]);
    assert(relativeError < tolerance, ...
        "Gradient check failed for %s (relative error %.3e).", ...
        model, relativeError);
    fprintf("%-24s relative error: %.3e\n", model, relativeError);
end

fprintf("All likelihood gradient checks passed.\n");
end
