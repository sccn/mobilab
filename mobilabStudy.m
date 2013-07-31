classdef mobilabStudy
    properties
        namespace
        path
        script
        dataTable
        hashTable
    end
    properties(Dependent)
        condition
        streamName
    end
    methods
        function obj = mobilabStudy(varargin)
            objfields = {'namespace','path','script','dataTable','hashTable'};
            N = length(varargin);
            if N==1, varargin = varargin{1};N = length(varargin);end
            
            for it=1:2:N-1
                ind = ismember(objfields,varargin{it});
                if any(ind), eval(['obj.' objfields{ind} '= varargin{it+1};']);end
            end
        end
        %%
        function condition = get.condition(obj)
            [~,p]= size(obj.dataTable);
            if p==1, condition = obj.dataTable{1,1}.condition;return;end
            condition = cell(1,p);
            for it=1:p, condition{it} = obj.dataTable{1,it}.condition;end
        end
        %%
        function streamName = get.streamName(obj)
            n= size(obj.dataTable,n);
            if n==1, streamName = obj.dataTable{1,1}.name;return;end
            streamName = cell(1,n);
            for it=1:n, streamName{it} = obj.dataTable{it,1}.name;end
        end
        %%
        function obj = loadStudy(studyFile)
            T = load(studyFile);
            obj_properties = fieldnames(T);
            obj_values = struct2cell(T);
            varargIn = cat(1,obj_properties,obj_values);
            
            obj = mobilabStudy(varargIn);
            if isempty(obj.path{1}), return;end
            N = length(obj.path);
            for it=1:N
                files = pickfiles(obj.path{it},obj.hashTable);
                if ~isempty(files)
                    folder = fileparts(deblank(files(1,:)));
                    allStreams = dadatSourceMoBI(folder);
                    allStreams.getItemIndexFromItemClass()
                end
            end
        end
    end
    methods(Hidden=true)
        function loadThisDataset(mobiFolder,streamName,hashTable,condition)
            
            allStreams = dadatSourceMoBI(mobiFolder);
            indices = allStreams.findItem(hashTable);
            
            N = length(indices);
            Nc = length(condition);
            for it=1:N
                for it=1:Nc
                    
                end
                
            end
        end
    end
end