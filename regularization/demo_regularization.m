%DEMO_REGULARIZATION Combine a likelihood gradient with a TV prior.

clear;
clc;
rng(3);

object = randn(64) + 1i * randn(64);
regularizationWeight = 1e-2;
learningRate = 5e-2;

for iteration = 1:20
    [regularizationLoss, cache] = regularization_forward( ...
        object, "tv", weight=regularizationWeight);
    regularizationGradient = regularization_backward(cache);

    % Replace this placeholder data gradient with the gradient returned by
    % the selected phase-retrieval likelihood and forward model.
    dataGradient = object;
    object = object - learningRate * ...
        (dataGradient + regularizationGradient);

    fprintf("Iteration %2d, TV loss %.6e\n", ...
        iteration, regularizationLoss);
end

% Constraint projections can be applied after each gradient update.
support = false(size(object));
support(9:end-8, 9:end-8) = true;
object = regularization_project(object, "support", mask=support);
