classdef audioStream < dataStream
    methods
        function obj = audioStream(header)
            % Creates a dataStream object.
            % 
            % Input arguments:
            %       header:     header file (string)
            % 
            % Output arguments:
            %       obj:        audioStream object (handle)
            %
            % Usage:
            %       obj = audioStream(header);
            
            if nargin < 1, error('Not enough input arguments.');end
            obj@dataStream(header);
        end
        function browserObj = plot(obj,~)
            % Plots the audio signal in a browser window.
            browserObj = audioStreamBrowse(obj);
        end
        function browserObj = audioStreamBrowse(obj,defaults)
            % Plots the audio signal in a browser window.
            
            if nargin < 2, defaults.browser = @audioStreamBrowserHandle;end
            if ~isfield(defaults,'browser'), 
                if isstruct(defaults)
                    defaults.browser = @audioStreamBrowserHandle;
                else
                    defaults = struct('browser',@audioStreamBrowserHandle);
                end
            end
            browserObj = defaults.browser(obj,defaults);
        end
    end
    methods(Hidden)
        function jmenu = contextMenu(obj)
            jmenu = javax.swing.JPopupMenu;
            %--
            menuItem = javax.swing.JMenuItem('Savitzky-Golay data denoising');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'sgolayFilter',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Filter');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'filter',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('ICA');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'ica',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Time frequency analysis (CWT)');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'continuousWaveletTransform',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Plot');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'audioStreamBrowse',-1});
            jmenu.add(menuItem);
            %--
            menuItem = javax.swing.JMenuItem('Plot spectrum');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'spectrum',-1});
            jmenu.add(menuItem);
            %---------
            jmenu.addSeparator;
            %---------
            menuItem = javax.swing.JMenuItem('Inspect');
            set(handle(menuItem,'CallbackProperties'), 'ActionPerformedCallback', {@myDispatch,obj,'inspect',-1});
            jmenu.add(menuItem);
            %--
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