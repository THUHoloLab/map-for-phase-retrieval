classdef optimizer_adamax
  
    properties
        name = 'adam'
        mom1 = 0;
        mom2 = 0;
        r1 = 0.9;
        r2 = 0.999;
        lr = 1e-2;
    end
    
    methods
        function obj = optimizer_adamax(mom1,mom2,beta1,beta2,lr)
            obj.mom1 = mom1;
            obj.mom2 = mom2;
            obj.r1 = beta1;
            obj.r2 = beta2;
            obj.lr = lr;
        end
        
        function para_out = step(obj,para_in,grad,iter)
            obj.mom1 = obj.r1 .* obj.mom1 + ...
                                           (1 - obj.r1) .* grad;
            obj.mom2 = max(obj.r2 .* obj.mom2, abs(grad));

            bias = 1./(1-obj.r1.^iter);

            update = bias .* obj.mom1 ./ (obj.mom2 + 1e-5);
            
            para_out = para_in - obj.lr .* update ;
        end
    end
end

