classdef auxSegment
    properties(GetAccess = protected, SetAccess = protected,Hidden)
        segStreamObj
        dim;
        cmd1
        cmd2
        cmd3
    end
    methods
        function obj = auxSegment(segStreamObj)
            if nargin < 1, error('Not enough input arguments.');end
            if ~strcmp(segStreamObj.isMemoryMappingActive,'active'), disp('Cannot initialize the segments in an empty mm-file.');return;end
            obj.segStreamObj = segStreamObj;
            obj.dim = segStreamObj.size;
            if isa(segStreamObj,'projectedMocap') || isa(segStreamObj,'vectorMeasureInSegments')
                obj.cmd1 = 'obj.segStreamObj.reshape([obj.dim(1) 2 obj.dim(2)/2]);';
                obj.cmd2 = ':,:';
                obj.cmd3 = 'obj.segStreamObj.reshape(obj.dim);';
            elseif isa(segStreamObj,'segmentedMocap')
                obj.cmd1 = 'obj.segStreamObj.reshape([obj.dim(1) 3 obj.dim(2)/3]);';
                obj.cmd2 = ':,:';
                obj.cmd3 = 'obj.segStreamObj.reshape(obj.dim);';
            else
                obj.cmd1 = '';
                obj.cmd2 = ':';
                obj.cmd3 = '';
            end
        end
        %%
        function obj = subsasgn(obj,S,B)
            if ~all(ismember(S.subs{1},1:obj.segStreamObj.numberOfSegments)), error(' Index exceeds segment dimensions.');end
            if ~isnumeric(B(:)), error('The input argument must be a matrix.');end
            eval(obj.cmd1);
            [t1,t2] = obj.segStreamObj.getTimeIndex([obj.segStreamObj.segmentObj.startLatency(S.subs{1})...
                obj.segStreamObj.segmentObj.endLatency(S.subs{1})]);%#ok
            try
                eval(['obj.segStreamObj.data(t1:t1+size(B,1)-1,' obj.cmd2 ') = B;']);
            catch ME
                ME.rethrow
            end 
            eval(obj.cmd3);
        end
        %% 
        function segment = subsref(obj,S)
            [t1,t2] = obj.segStreamObj.getTimeIndex([obj.segStreamObj.segmentObj.startLatency(S.subs{1})...
                obj.segStreamObj.segmentObj.endLatency(S.subs{1})]);%#ok
            eval(obj.cmd1);
            segment = eval(['obj.segStreamObj.data(t1:t2,' obj.cmd2 ');']);
            eval(obj.cmd3);
        end
        %%
        function disp(obj)
            disp(['<' num2str(obj.segStreamObj.numberOfSegments) 'x1 cell>']);
        end
    end
end