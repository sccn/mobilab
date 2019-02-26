classdef videoStream < coreStreamObject
    properties
        videoFile;
    end
    methods
        function obj = videoStream(header) 
            if nargin < 1, error('Not enough input arguments.');end
            obj = obj@coreStreamObject(header);
            warning off
            load(header,'-mat','videoFile');
            warning on
            if exist('videoFile','var')
                if ~isempty(videoFile)
                    obj.videoFile = videoFile;
                end
            else
                obj.videoFile = '';
            end
        end
        function set.videoFile(obj,videoFile)
            if ~ischar(videoFile), return;end
%             stack = dbstack;
%             if any(strcmp({stack.name},'videoStream.videoStream')), return;end
            obj.videoFile = videoFile;
            saveProperty(obj,'videoFile',videoFile);
        end
        %%
        function browserObj = plot(obj), browserObj = videoStreamBrowser(obj);end
        function browserObj = videoStreamBrowser(obj,defaults)
            if nargin < 2, defaults.browser  = @videoStreamBrowser;end
            browserObj = videoStreamBrowserHandle1(obj,defaults); 
        end
    end
    methods(Hidden=true)
        function newHeader = createHeader(obj,commandHistory), end %#ok
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Show');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'videoStreamBrowser',-1});
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
            menuItem = javax.swing.JMenuItem('<HTML><FONT color="red">Delete object</HTML>');
            jmenu.add(menuItem);
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj.container,'deleteItem',obj.container.findItem(obj.uuid)});   
        end
    end
end