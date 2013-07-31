classdef pcaMocap < mocap
    properties
        projectionMatrix
    end
    properties(Dependent)
        angle
        dataInXY
        curvature
    end
    methods
        function obj = pcaMocap(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@mocap(header);
        end
        %%
        function projectionMatrix = get.projectionMatrix(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'pcaMocap.set.projectionMatrix')), projectionMatrix = obj.projectionMatrix;return;end
            if isempty(obj.projectionMatrix), obj.projectionMatrix = retrieveProperty(obj,'projectionMatrix');end
            projectionMatrix = obj.projectionMatrix;
        end
        function set.projectionMatrix(obj,projectionMatrix)
            stack = dbstack;
            if any(strcmp({stack.name},'pcaMocap.get.projectionMatrix')), obj.projectionMatrix = projectionMatrix;return;end
            obj.projectionMatrix = projectionMatrix;
            saveProperty(obj,'projectionMatrix',projectionMatrix);
        end
        %%
        function data = get.dataInXY(obj)
            if obj.numberOfChannels/2 > 1
                dim = obj.size;
                obj.reshape([dim(1) 2 dim(2)/2]);
                data = obj.mmfObj.Data.x;
                obj.reshape(dim);
            else data = obj.mmfObj.Data.x;
            end
        end
        function set.dataInXY(obj,data)
            if obj.numberOfChannels/2 > 1
                dim = obj.size;
                obj.reshape([dim(1) 2 dim(2)/2]);
                obj.mmfObj.Data.x = data;
                obj.reshape(dim);
            else obj.mmfObj.Data.x = data;
            end
        end
        %%
        function angle = get.angle(obj), angle = unwrap(squeeze(atan2(obj.dataInXY(:,2,:),obj.dataInXY(:,1,:))));end
        function curvature = get.curvature(obj)
            index = obj.container.findItem(obj.uuid);
            descendants = obj.container.getDescendants(index);
            indexVel = [];
            indexAcc = [];
            for it=1:length(descendants)
                if ~isempty(strfind(obj.container.item{descendants(it)}.name,'vel'))
                    indexVel = descendants(it);
                elseif ~isempty(strfind(obj.container.item{descendants(it)}.name,'acc'))
                    indexAcc = descendants(it);
                end
            end
            if isempty(indexVel) || isempty(indexAcc)
                accObj = obj.smoothDerivative(2);
                index = obj.container.findItem(accObj.uuid);
                velObj = obj.container.item{index-1}; 
            end
            a = permute(abs(velObj.dataInXY(:,1,:).*accObj.dataInXY(:,2,:) - velObj.dataInXY(:,2,:).*accObj.dataInXY(:,1,:)),[1 3 2]);
            b = permute(sum(velObj.dataInXY.^2,2).^(1.5),[1 3 2]);
            t = [-b;b];
            t = bsxfun(@rdivide,t,std(t));
            n = size(t,1);
            th = tinv(0.65,n-1);
            ind = t(n/2+1:end,:) < th;
            curvature = a./(b+eps);
            th = prctile(nonzeros(curvature(:)),95);            
            curvature(curvature>th) = th;
            ind = ind | isnan(curvature) | curvature > th;
            I = (1:n)';
            for it=1:size(curvature,2)
                Ii = find(ind(:,it));
                interp1(I(~ind(:,it)),curvature(~ind(:,it),it),Ii,'spline');
            end
            curvature = reshape(smooth(curvature,16,'moving'),size(a));
        end
        %%
        function c = projectInlineItem(obj,itemIndex)
            % project item a onto item b, both items need to be from the MoBILAB vectorMeasureInSegments class
            commandHistory.commandName = 'projectInlineItem';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = itemIndex;
            c = obj.copyobj(commandHistory);
            
            b = obj.container.item{itemIndex};
            
            bm = b.magnitude+eps;
            bu(:,1,:) = squeeze(b.dataInXY(:,1,:))./bm; 
            bu(:,2,:) = squeeze(b.dataInXY(:,2,:))./bm;
            
            dot_ab = dot(obj.dataInXY,bu,2);
            c.dataInXY(:,1,:) = dot_ab.*bu(:,1,:);
            c.dataInXY(:,2,:) = dot_ab.*bu(:,2,:);
        end
        %%
        function c = projectOutlineItem(obj,itemIndex)
            commandHistory.commandName = 'projectOutlineItem';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = itemIndex;
            c = obj.copyobj(commandHistory);
            
            b = obj.container.item{itemIndex};
            
            bm = b.magnitude+eps;
            bu(:,1,:) = squeeze(b.dataInXY(:,1,:))./bm;
            bu(:,2,:) = squeeze(b.dataInXY(:,2,:))./bm;
            
            dot_ab = dot(obj.dataInXY,bu,2);
            c.dataInXY(:,1,:) = obj.dataInXY(:,1,:) - dot_ab.*bu(:,1,:);
            c.dataInXY(:,2,:) = obj.dataInXY(:,2,:) - dot_ab.*bu(:,2,:);
        end
        %%
        function c = scaleByItem(obj,itemIndex)
            % project item a onto item b, both items need to be from the MoBILAB vectorMeasureInSegments class
            commandHistory.commandName = 'scaleByItem';
            commandHistory.uuid        = obj.uuid;
            commandHistory.varargin{1} = itemIndex;
            c = obj.copyobj(commandHistory);
                
            b = obj.container.item{itemIndex};
                    
            bm = b.magnitude;
            sigB = std(bm)+eps;
            
            sigC = std(c.magnitude)+eps;
            sig = sigB/sigC;
            c.dataInXY(:,1,:) = sig*c.dataInXY(:,1,:)./bm;
            c.dataInXY(:,2,:) = sig*c.dataInXY(:,2,:)./bm;
        end
        %%
        function browserObj = plot(obj), browserObj = projectionBrowser(obj);end
        function browserObj = cometBrowser(obj), browserObj = cometBrowserHandle2(obj);end
        function browserObj = projectionBrowser(obj,defaults)
            if nargin < 2, browserObj = projectionBrowserHandle(obj);
            else browserObj = projectionBrowserHandle(obj,defaults);
            end
        end
        function browserObj = vectorBrowser(obj,defaults)
            if nargin < 2, browserObj = vectorBrowserHandle(obj);
            else browserObj = vectorBrowserHandle(obj,defaults);
            end
        end
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Compute time derivatives');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'smoothDerivative',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot traces');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'projectionBrowser',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Comet plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'cometBrowser',-1});
            jmenu.add(menuItem);
            %---------
            menuItem = javax.swing.JMenuItem('Time frequency analysis (CWT)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'continuousWaveletTransform',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Annotation');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@annotation_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('Generate batch script');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@generateBatch_Callback,obj});
            %--
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="maroon">Delete object</HTML>');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});
        end
    end
    methods(Hidden = true)
        function newHeader = createHeader(obj,commandHistory)
            if nargin < 2
                commandHistory.commandName = 'copyobj';
                commandHistory.uuid = obj.uuid;
            end
            newHeader = createHeader@mocap(obj,commandHistory);
            if ~isempty(newHeader), return;end
            
            metadata = obj.saveobj;
            metadata.writable = true;
            metadata.parentCommand = commandHistory;
            metadata.uuid = char(java.util.UUID.randomUUID);
            path = fileparts(obj.binFile);
            
            switch commandHistory.commandName
                case 'projectInlineItem'
                    prename = 'inLine_';
                case 'projectOutlineItem'
                    prename = 'outLine_';
                case 'scaleByItem'
                    prename = 'scale_';        
            end
            metadata.name = [prename metadata.name];
            metadata.binFile = fullfile(path,[metadata.name '_' char(metadata.uuid) '_' metadata.sessionUUID '.bin']);
            allocateFile(metadata.binFile,obj.precision,[length(metadata.timeStamp) metadata.numberOfChannels]);
        end
    end
end