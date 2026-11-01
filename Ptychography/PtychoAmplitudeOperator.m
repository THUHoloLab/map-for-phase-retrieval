classdef PtychoAmplitudeOperator < deep.DifferentiableFunction
    %PTYCHOAMPLITUDEOPERATOR Differentiable far-field ptychography model.
    %   The operator extracts object patches, multiplies them by a shared
    %   probe, and returns centered Fourier amplitudes. Its backward method
    %   applies the exact adjoint FFT and overlap-adds object gradients.
    %
    %   This custom operation is necessary because native complex fft2 on
    %   dlarray is not supported by all MATLAB releases.

    methods
        function self = PtychoAmplitudeOperator
            self@deep.DifferentiableFunction(1, ...
                SaveInputsForBackward=true, ...
                SaveOutputsForBackward=false, ...
                NumMemoryValues=0);
        end

        function amplitude = forward(~, object, probe, positions)
            numberOfFrames = size(positions, 1);
            patchStack = repmat(probe .* 0, 1, 1, numberOfFrames);

            for frame = 1:numberOfFrames
                rows = positions(frame, 1) + (0:size(probe, 1)-1);
                columns = positions(frame, 2) + (0:size(probe, 2)-1);
                patchStack(:, :, frame) = object(rows, columns);
            end

            exitWaves = patchStack .* probe;
            detectorWaves = fftshift(fftshift(fft2(exitWaves), 1), 2);
            amplitude = abs(detectorWaves);
        end

        function [objectGradient, probeGradient, positionGradient] = ...
                backward(~, amplitudeGradient, ~, object, probe, positions)
            numberOfFrames = size(positions, 1);
            patchStack = repmat(probe .* 0, 1, 1, numberOfFrames);

            for frame = 1:numberOfFrames
                rows = positions(frame, 1) + (0:size(probe, 1)-1);
                columns = positions(frame, 2) + (0:size(probe, 2)-1);
                patchStack(:, :, frame) = object(rows, columns);
            end

            exitWaves = patchStack .* probe;
            detectorWaves = fftshift(fftshift(fft2(exitWaves), 1), 2);
            detectorGradient = amplitudeGradient .* detectorWaves ./ ...
                max(abs(detectorWaves), eps("single"));

            % fft2 is unnormalized, so its Euclidean adjoint is
            % numel(probe) * ifft2.
            exitGradient = ifft2(ifftshift(ifftshift( ...
                detectorGradient, 1), 2)) * numel(probe);

            objectGradient = object .* 0;
            probeGradient = sum(conj(patchStack) .* exitGradient, 3);
            for frame = 1:numberOfFrames
                rows = positions(frame, 1) + (0:size(probe, 1)-1);
                columns = positions(frame, 2) + (0:size(probe, 2)-1);
                objectGradient(rows, columns) = ...
                    objectGradient(rows, columns) + ...
                    conj(probe) .* exitGradient(:, :, frame);
            end

            positionGradient = [];
        end
    end
end
