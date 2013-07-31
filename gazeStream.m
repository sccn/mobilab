% Defines a gazeStream class for working with a custom made eye tracker developed
% at SCCN by Matthew Grivich.
% 
% For more details visit: https://code.google.com/p/mobilab/ 
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Oct-2012

classdef gazeStream < dataStream
    properties(Dependent)
        eyeRadius          % Returns the radius of the pupil.
        
        eyePosition        % Returns the xyz position of the eye.
        
        gazePosition       % Returns the xyz position of the point the subject is looking at.
    end
    properties(Hidden = true, Constant)
        convert2phaseSpace = 1e-5;
    end
    methods
        function obj = gazeStream(header)
            if nargin < 1, error('Not enough input parameters.');end
            obj@dataStream(header);
        end
        %%
        function eyeRadius    = get.eyeRadius(obj),    eyeRadius    = obj.mmfObj.Data.x(:,9);end
        function eyePosition  = get.eyePosition(obj),  eyePosition  = obj.mmfObj.Data.x(:,[10 12 11])*obj.convert2phaseSpace;end
        function gazePosition = get.gazePosition(obj), gazePosition = obj.mmfObj.Data.x(:,[13 15 14])*obj.convert2phaseSpace;end
        function browserObj   = plot(obj)
            % Overwrites the plot method defined in its base class to display
            % gaze position as a heat map. Internally it calls the method 
            % gazeStreamBrowser.
            
            browserObj   = gazeStreamBrowser(obj);
        end
        %%
        function browserObj = gazeStreamBrowser(obj,defaults)
            % Displays gaze position data in a browser as a heat map.
            
            if nargin < 2, defaults.browser = @gazeStreamBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
    end
    methods(Static)
        function [methodsInCell,callbacks] = methods2bePublished
            methodsInCell = {'Plot'};
            callbacks = {'gazeStreamBrowser'};
        end
    end
end