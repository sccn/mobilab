% Creates a dataStream object
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 05-Apr-2011

%%
classdef sceneStream < videoStream
    methods
        %%
        function obj = sceneStream(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@videoStream(header);
        end
        function browserObj = plot(obj), browserObj = dataStreamBrowser(obj);end
        function browserObj = sceneBrowser(obj,defaults)
            if ~obj.isMemoryMappingActive, return;end
            if nargin < 2, defaults.browser = @sceneBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser = @sceneBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
        function browserObj = sceneVideoBrowser(obj,defaults)
            if nargin < 2, defaults.browser = @sceneVideoBrowserHandle;end
            if ~isfield(defaults,'browser'), defaults.browser = @sceneVideoBrowserHandle;end
            browserObj = defaults.browser(obj,defaults);
        end
    end
    methods(Hidden = true)
        %%
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'sceneBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Scene and Video Browser');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'sceneVideoBrowser',-1});
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