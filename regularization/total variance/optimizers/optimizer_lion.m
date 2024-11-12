classdef optimizer_lion   
    properties
        name = 'lion'
        mom1 = 0;
        mom2 = 0;
        r1 = 0.9;
        r2 = 0.999;
        lr = 1e-2;
    end
    
    methods
        function obj = optimizer_lion(mom1,mom2,beta1,beta2,lr)
            obj.mom1 = mom1;
            obj.mom2 = mom2;
            obj.r1 = beta1;
            obj.r2 = beta2;
            obj.lr = lr;
        end
        
        function para_out = step(obj,para_in,grad)
            obj.mom1 = sign(obj.r1 .*  obj.mom2 + ...
                                           (1 - obj.r1) .* grad);

            obj.mom2 = obj.r2 .* obj.mom2 + (1 - obj.r2) .* grad;

            para_out = para_in - obj.lr .* obj.mom1 ;

        end
    end
end

