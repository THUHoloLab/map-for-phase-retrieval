function test_regularization_gradients
%TEST_REGULARIZATION_GRADIENTS Verify analytic gradients numerically.

rng(11);
object = randn(7, 6) + 1i * randn(7, 6);
direction = randn(size(object)) + 1i * randn(size(object));
direction = direction / norm(direction(:));

models = [ ...
    "l2", "l1_smooth", "gradient_l2", "tv", ...
    "gradient_hyperlaplacian", "hessian_l2", "hessian_tv"];

step = 1e-6;
tolerance = 2e-5;

for model = models
    [~, cache] = regularization_forward( ...
        object, model, epsilon=1e-3, p=0.8, weight=0.7);
    gradient = regularization_backward(cache);
    analyticDerivative = real(sum(conj(gradient) .* direction, "all"));

    lossPlus = regularization_forward( ...
        object + step * direction, model, ...
        epsilon=1e-3, p=0.8, weight=0.7);
    lossMinus = regularization_forward( ...
        object - step * direction, model, ...
        epsilon=1e-3, p=0.8, weight=0.7);
    numericDerivative = (lossPlus - lossMinus) / (2 * step);

    relativeError = abs(analyticDerivative - numericDerivative) / ...
        max([1, abs(analyticDerivative), abs(numericDerivative)]);
    assert(relativeError < tolerance, ...
        "Gradient check failed for %s (relative error %.3e).", ...
        model, relativeError);
    fprintf("%-28s relative error: %.3e\n", model, relativeError);
end

fprintf("All regularization gradient checks passed.\n");
end
