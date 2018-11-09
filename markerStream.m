classdef markerStream < coreStreamObject
    methods
        function obj = markerStream(header) 
            if nargin < 1, error('Not enough input arguments.');end
            obj@coreStreamObject(header);
        end
        
        function browserObj = plot(obj), browserObj = dataStreamBrowser(obj);end
        
        function hFigure = dispHedTags(obj,~)
            if nargin < 2, dispCommand = false; else dispCommand = true;end
            if dispCommand
                index = obj.container.findItem(obj.uuid); 
                disp('Running:')
                disp(['  mobilab.allStreams.item{' num2str(index) '}.dispHedTags;']);
            end
            hedTreeObj = hedTree(obj.event.hedTag);
            hFigure = plot(hedTreeObj);
        end
    end
    methods(Hidden=true)
        function newHeader = createHeader(obj,commandHistory), error('MoBILAB:sealedClass','Cannot create a copy of this object.');end %#ok
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dataStreamBrowser',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Disp Hed Tree');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'dispHedTags',-1});
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