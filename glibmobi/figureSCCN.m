function h = figureSCCN(varargin)
if nargin < 1, 
    h = figure;
    set(h,'Color',[0.66 0.76 1]);
elseif ishandle(varargin)
    figure(varargin);
end
    