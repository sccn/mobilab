classdef pcdStream < coreStreamObject & headModel
    properties
        roi
    end
    methods
        function obj = pcdStream(header)
            if nargin < 1, error('Not enough input arguments.');end
            metadata = load(header,'-mat');
            obj_properties = fieldnames(metadata);
            obj_values     = struct2cell(metadata);
            varargIn = cat(1,obj_properties,obj_values);
            Np = length(obj_properties);
            index = [1:Np; Np+1:2*Np];
            varargIn = varargIn(index(:));
            obj@coreStreamObject(header);
            obj@headModel(varargIn);
            
            if ~isempty(obj.surfaces) && ~exist(obj.surfaces,'file')
                [~,name] = fileparts(obj.surfaces);
                obj.surfaces = fullfile(obj.container.mobiDataDirectory,[name '.mat']);
                if ~exist(obj.surfaces,'file'), obj.surfaces = [];end
                saveProperty(obj,'surfaces',obj.surfaces)
            end
        end
        %%
        function roi = get.roi(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'pcdStream.set.roi')), roi = obj.roi;return;end
            if isempty(obj.roi), obj.roi = retrieveProperty(obj,'roi');end
            roi = obj.roi;
        end
        function set.roi(obj,roi)
            stack = dbstack;
            if any(strcmp({stack.name},'pcdStream.get.roi')), obj.roi = roi;return;end
            obj.roi = roi;
            saveProperty(obj,'roi',roi);
        end
        %%
        function delete(obj)
            delete@headModel(obj);
            delete@coreStreamObject(obj);
        end
        %%
        function connect(obj)
            try descriptor = dir(obj.binFile);
                dim = [length(obj.timeStamp) obj.numberOfChannels];
                if ~isempty(obj.binFile) && ~isempty(obj.precision) && dim(2) > 0 && ~isempty(descriptor) && descriptor.bytes
                    obj.mmfObj = memmapfile(obj.binFile,'Format',{obj.precision fliplr(dim) 'x'},'Writable',obj.writable);
                end
            catch obj.mmfObj = [];%#ok
            end
        end
        %%
        function disp(obj)
            string = sprintf('  channelSpace:         <%ix3 double>',size(obj,2));
            if ~isempty(obj.surfaces)
                string = sprintf('%s\n  surfaces:             %s',string, obj.surfaces);
                string = sprintf('%s\n  atlas.colorTable:     <%ix1 %s>',string, length(obj.atlas.colorTable), class(length(obj.atlas.colorTable)));
                string = sprintf('%s\n  atlas.label:          <%ix1 cell>',string, length(obj.atlas.label));
            else
                string = sprintf('%s\n  surfaces:            ''''',string);
                string = sprintf('%s\n  atlas.color:         []',string);
                string = sprintf('%s\n  atlas.label:         {[]}',string);
            end
            disp@coreStreamObject(obj)
            disp(string);
        end
        %%
        function browserObj = plot(obj), browserObj = plotPCD(obj);end
        function browserObj = plotPCD(obj,defaults)
            if nargin < 2, defaults.browser = @pcdBrowserHandle;end
            if ~isstruct(defaults), defaults = struct;end
            if ~isfield(defaults,'browser'), defaults.browser = @pcdBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
        function browserObj = plotROI(obj,defaults)
            if nargin < 2, defaults.browser = @streamBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser = @streamBrowserHandle;end
            defaults.gain          = 0.008;
            defaults.normalizeFlag = true;
            dsObj.name             = obj.name;
            dsObj.timeStamp        = obj.timeStamp;
            dsObj.numberOfChannels = length(obj.roi.label);
            dsObj.event            = event;%obj.event;
            dsObj.samplingRate     = obj.samplingRate;
            dsObj.label            = obj.roi.label;
            dsObj.container        = obj.container;
            dsObj.mmfObj.Data.x    = obj.roi.data;
            dsObj.uuid             = obj.uuid;
            browserObj             = defaults.browser(dsObj,defaults);
        end
    end
    methods(Hidden = true)
        function newHeader = createHeader(obj,commandHistory), disp('Nothing to create.');end %#ok
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'plotPCD',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot ROI signal');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'plotROI'});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot spectrum');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrum',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Show head model');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'plotHeadModel',-1});
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
end