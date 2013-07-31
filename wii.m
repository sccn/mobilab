classdef wii < dataStream %mocap
    methods
        %%
        function obj = wii(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        %%
        function cobj = lowpass(obj, varargin)
            if length(varargin) == 1 && iscell(varargin{1}), varargin = varargin{1};end
            if ~isempty(varargin) && iscell(varargin) && isnumeric(varargin{1}) && varargin{1} == -1
                prompt = {'Enter the cutoff frequency:'};
                dlg_title = 'Filter input parameters';
                num_lines = 1;
                def = {'6'};
                varargin = inputdlg2(prompt,dlg_title,num_lines,def);
                if isempty(varargin), return;end
                varargin{1} = str2double(varargin{1});
                if isnan(varargin{1}), return;end
            end
           
            % Cutoff Frequency
            if nargin < 2, fc = 6; else fc = varargin{1};end     
            if nargin < 3, channels = obj.getAccChannel;else channels = varargin{2};
            end
            if ~isnumeric(channels) || ~any(intersect(channels,obj.getAccChannel))
                error('prog:input','Third argument must be the channels to filter.');
            end
            cobj = lowpass@mocap(obj, fc,channels);
        end
        %%
%         function cobj = smoothDerivative(obj,order,fc,channels)
%             if nargin < 2, order = 3;end
%             if nargin < 3; fc = 18;end 
%             if nargin < 4, channels = obj.getAccChannel;end
%             
%             if ~isnumeric(order)
%                 error('prog:input','First argument must be the order of the derivative.');
%             end
%             if ~isnumeric(fc)
%                 error('prog:input','Fourth argument must be the cutoff frequency that defines the smoothing kernel.');
%             end
%             if ~isnumeric(channels) || ~any(intersect(channels,obj.getAccChannel))
%                 error('prog:input','Second argument must be the channels you going to differentiate.');
%             end
%             cobj = smoothDerivative@mocap(obj, order, fc,channels);
%         end
%         %%
%         function browserObj = mocapBrowser(obj,defaults)
%             if nargin < 2,
%                 defaults.browser = @mocapBrowserHandle;
%                 defaults.channels = 1:length(obj.getAccChannel)/3;
%             end
%             if ~isfield(defaults,'browser'), defaults.browser = @mocapBrowserHandle;end
%             if ~isfield(defaults,'channels'),defaults.channels = 1:length(obj.getAccChannel)/3;
%             end
%             browserObj = defaults.browser(obj,defaults);
%         end
%         function browserObj = plot(obj), browserObj = plot@dataStream(obj);end
        %%
        function ind = getAccChannel(obj)
            ind = false(obj.numberOfChannels,1);
            labels = lower(obj.label);
            for it=1:obj.numberOfChannels
                if strfind(labels{it},'x') || strfind(labels{it},'y') || strfind(labels{it},'z'), ind(it) = true;end
            end
            ind = find(ind);
        end
    end
end