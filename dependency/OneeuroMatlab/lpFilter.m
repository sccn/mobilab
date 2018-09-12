% See Gery Casiez, Nicolas Roussel, Daniel Vogel.  1 ? Filter:  A Simple Speed-based Low-pass Filter for
% Noisy Input in Interactive Systems.  CHI?12, the 30th Conference on Human Factors in Computing
% Systems, May 2012, Austin, United States. ACM, pp.2527-2530, 2012, <10.1145/2207676.2208639>.
% <hal-00670496>

classdef lpFilter < handle
    properties
        firstTime;
        hatxprev;
    end
    
    methods
        function obj = lpFilter()
            obj.firstTime = true;
        end
        
        function y = last(obj)
            y = obj.hatxprev;
        end
        
        function y = filter(obj, x, alphaval)
            if(obj.firstTime)
                obj.firstTime = false;
                hatx = x;
            else
                hatx = alphaval*x+(1-alphaval)*obj.hatxprev;
            end
            
            obj.hatxprev = hatx;
            
            y = hatx;
        end
    end
end