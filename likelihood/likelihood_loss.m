function [loss, objectGradient, detectorGradient] = likelihood_loss( ...
        object, measurement, forwardOperator, adjointOperator, model, options)
%LIKELIHOOD_LOSS Evaluate a loss and backpropagate through a linear model.
%   [LOSS, OBJECTGRADIENT, DETECTORGRADIENT] = LIKELIHOOD_LOSS(OBJECT,
%   MEASUREMENT, FORWARDOPERATOR, ADJOINTOPERATOR, MODEL, OPTIONS) computes
%   the detector field as FORWARDOPERATOR(OBJECT), evaluates the selected
%   likelihood, and maps its gradient back with ADJOINTOPERATOR.
%
%   FORWARDOPERATOR and ADJOINTOPERATOR must be function handles that form
%   an adjoint pair. For a matrix A, use @(x) A*x and @(y) A'*y.

arguments
    object {mustBeNumeric}
    measurement {mustBeNumeric, mustBeReal, mustBeNonnegative}
    forwardOperator (1,1) function_handle
    adjointOperator (1,1) function_handle
    model {mustBeTextScalar}
    options.epsilon (1,1) double {mustBePositive} = 1e-8
    options.weight {mustBeNumeric, mustBeReal, mustBeNonnegative} = 1
    options.gamma (1,1) double {mustBePositive} = 0.5
    options.p (1,1) double {mustBePositive} = 0.8
    options.delta (1,1) double {mustBePositive} = 1
end

detectorField = forwardOperator(object);
[loss, cache] = likelihood_forward(detectorField, measurement, model, ...
    epsilon=options.epsilon, weight=options.weight, ...
    gamma=options.gamma, p=options.p, delta=options.delta);
detectorGradient = likelihood_backward(cache);
objectGradient = adjointOperator(detectorGradient);
end
