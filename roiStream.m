classdef roiStream < dataStream
    properties
        reducedStateSpace
    end
    methods
        function obj = roiStream(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        %%
        function reducedStateSpace = get.reducedStateSpace(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'roiStream.set.reducedStateSpace')), reducedStateSpace = obj.reducedStateSpace;return;end
            if isempty(obj.reducedStateSpace), obj.reducedStateSpace = retrieveProperty(obj,'reducedStateSpace');end
            reducedStateSpace = obj.reducedStateSpace;
        end
        function set.reducedStateSpace(obj,reducedStateSpace)
            stack = dbstack;
            if any(strcmp({stack.name},'roiStream.get.reducedStateSpace')), obj.reducedStateSpace = reducedStateSpace;return;end
            obj.reducedStateSpace = reducedStateSpace;
            saveProperty(obj,'reducedStateSpace',reducedStateSpace);
        end
        %%
        function [connectivity, pval, hFigure] = computeConnectivityMatrix(obj,latency,plotFlag)
            if nargin < 2, latency = 1:size(obj,1);end
            if nargin < 3, plotFlag = true;end
            [connectivity,pval] = corr(obj.mmfObj.Data.x(latency,:));
            I = strfind(obj.label,'_L');
            I = cellfun(@isempty,I);
            ind = [find(~I); find(I)];
            connectivity = connectivity(ind,ind);
            pval         = pval(ind,ind);
            if plotFlag
                hFigure = figure;
                h = imagesc(connectivity);
                colormap jet;
                hAxes = get(h,'parent');
                set(hAxes,'YTick',1:obj.numberOfChannels,'YTickLabel',obj.label(ind));
                set(hAxes,'XTick',1:obj.numberOfChannels,'XTickLabel',obj.label(ind));                
                set(hAxes,'userData',obj.label(ind),'ButtonDownFcn',@onClickDownOnConn);
                rotateticklabel(hAxes);
                title('Correlation between brain regions');
                colorbar;
            else hFigure = [];
            end
        end
        %%
        function stateSpaceGeometricAnalysis(obj,stateSpaceReducedDimension,delta)
            if nargin < 2, stateSpaceReducedDimension = 3;end
            if nargin < 3, delta = 32;end
            stateSpaceReducedDimension(stateSpaceReducedDimension<2) = 2;
            stateSpaceReducedDimension(stateSpaceReducedDimension>3) = 3;
            if size(obj,1) < 3*delta, delta=1;end
            Nt = size(obj,1);
            data = obj.mmfObj.Data.x;
            % I = find(triu(ones(obj.numberOfChannels),1));
            % ss = zeros(Nt,length(I));
            ss = zeros(Nt,obj.numberOfChannels);
            
            % ss = zeros(Nt,stateSpaceReducedDimension);
            % Ones = ones(delta*2,1);
            obj.container.container.initStatusbar(1,Nt,'Computing reduced state space trajectory...');
            for it=delta+1:delta:Nt
                if it+delta <= Nt
                    % [connectivity, pval, hFigure] = computeConnectivityMatrix(obj,it-delta:it+delta-1,true);
                    % connectivity(isnan(connectivity)) = 0;
                    % ss(it-delta:it+delta-1,:) = connectivity(I);
                    Y = data(it-delta:it+delta-1,:);
                    ss(it-delta:it+delta-1,:) = Y.^2;
                    %Y = data(it-delta:it+delta-1,:);
                    %[~,~,latent] = princomp(Y);
                    %ss(it-delta:it+delta-1,:) = Ones*latent(1:stateSpaceReducedDimension)';
                else
                    Y = data(it-delta:end,:);
                    ss(it-delta:end,:) = Y.^2;
                    % [~,~,latent] = princomp(Y);
                    % ss(it-delta:end,:) = ones(size(Y,1),1)*latent(1:stateSpaceReducedDimension)';
                end
                obj.container.container.statusbar(it);
            end
            obj.container.container.statusbar(Nt);
            obj.reducedStateSpace = ss;
        end
        %%
        function hFigure = plotStateSpaceTrajectory(obj)
            if isempty(obj.reducedStateSpace), return;end
            if nargin < 2, defaults.browser = @streamBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser = @streamBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
    end
end


function onClickDownOnConn(obj,~,~)
hAxes = get(obj, 'parent');   % Get handle of current axis
userData = get(hAxes,'userData');

pos = get(hAxes, 'CurrentPoint');  % Get position of current mouse location
pos = round(pos);
C = get(obj,'CData');
if C(pos(1),pos(1,2)) && C(pos(1,2),pos(1)), disp([userData{pos(1)} ' - ' userData{pos(1,2)}]);
else disp('no connection ');
end
end