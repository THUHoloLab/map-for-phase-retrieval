classdef optimizer_sgd
    properties
        name = 'sgd'
        lr = 1e-2;
    end
    
    methods
        function obj = optimizer_sgd(lr)
            obj.lr = lr;
        end
        
        function para_out = step(obj,para_in,grad)
            para_out = para_in - obj.lr .* grad ;
        end
    end
end

