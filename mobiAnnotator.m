classdef mobiAnnotator < handle
    properties, container;end
    properties(SetObservable), text;end
    properties(Hidden=true),   listenerHandle;end
    methods 
        function obj = mobiAnnotator(arg,txt)
            if nargin < 1, arg = [];end
            if nargin < 2, txt = {date};end
            obj.container = arg;
            obj.text = txt;
            obj.listenerHandle = addlistener(obj,'text','PostSet',@mobiAnnotator.saveTextHandle);
        end
        %%
        function h = editor(obj), h = editNotes(obj);end
        function disp(obj),       disp(char(obj.text));end
        %%
        function set.text(obj,arg)
            if iscellstr(arg)
                obj.text = arg;
            elseif ischar(arg)
                obj.text{1} = arg;
            end
        end
        %%
        function text = saveobj(obj)
            text = obj.text;
            if strcmp(text(end),char(13)), text(end) = [];end    
            if length(text) == 1, text = [];end
        end
        %%
        function equal(obj,arg)
            if nargin < 1, error('The expression to the left of the equals sign is not a valid target for an assignment.');end
            if iscellstr(arg), obj.text = arg(:);
            else error('Property ''text'' must be a cell array of ''char''.');
            end
        end
    end
     methods(Static)
        function saveTextHandle(src,evnt) %#ok
            notes = evnt.AffectedObject.text;
            if isa(evnt.AffectedObject.container,'coreStreamObject')
                disp(['Saving notes in: ' evnt.AffectedObject.container.header]);
                save(evnt.AffectedObject.container.header,'-mat','-append','notes');
            else
                file = [evnt.AffectedObject.container.mobiDataDirectory filesep 'notes_' evnt.AffectedObject.container.sessionUUID '.txt'];
                disp(['Saving notes in: ' file]);
                cell2textfile(file,notes);
            end
        end
     end
end