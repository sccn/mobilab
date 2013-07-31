classdef icaFields < handle
    properties(GetAccess = public, SetAccess = public, AbortSet = true)
        icawinv     % Inverse of mixing times sphering matrices
        
        icasphere   % Sphering matrix
        
        icaweights  % Mixing matrix
    end
    properties(GetAccess = private, SetAccess = private)
        header
    end
    methods
        function obj = icaFields(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj.header = header;
        end
        %%
        function icawinv = get.icawinv(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'icaFields.set.icawinv'))
                icawinv = obj.icawinv;
                return;
            end 
            if isempty(obj.icawinv), obj.icawinv = retrieveProperty(obj,'icawinv');end
            icawinv = obj.icawinv;
        end
        function set.icawinv(obj,icawinv)
            dim = size(obj);
            if ~ismatrix(icawinv), error(['''icawinv'' must be a matrix <' num2str(dim(2)) 'x' num2str(dim(2)) '>']);end
            stack = dbstack;
            if any(strcmp({stack.name},'icaFields.get.icawinv'))
                obj.icawinv = icawinv;
                return;
            end
            obj.icawinv = icawinv;
            saveProperty(obj,'icawinv',icawinv)
        end
        %%
        function icasphere = get.icasphere(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'icaFields.set.icasphere'))
                icasphere = obj.icasphere;
                return;
            end 
            if isempty(obj.icasphere), obj.icasphere = retrieveProperty(obj,'icasphere');end
            icasphere = obj.icasphere;
        end
        function set.icasphere(obj,icasphere)
            dim = size(obj);
            if ~ismatrix(icasphere), error(['''icasphere'' must be a matrix <' num2str(dim(2)) 'x' num2str(dim(2)) '>']);end
            stack = dbstack;
            if any(strcmp({stack.name},'icaFields.get.icasphere'))
                obj.icasphere = icasphere;
                return;
            end
            obj.icasphere = icasphere;
            saveProperty(obj,'icasphere',icasphere)
        end
         %%
        function icaweights = get.icaweights(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'icaFields.set.icaweights'))
                icaweights = obj.icaweights;
                return;
            end 
            if isempty(obj.icaweights), obj.icaweights = retrieveProperty(obj,'icaweights');end
            icaweights = obj.icaweights;
        end
        function set.icaweights(obj,icaweights)
            dim = size(obj);
            if ~ismatrix(icaweights), error(['''icaweights'' must be a matrix <' num2str(dim(2)) 'x' num2str(dim(2)) '>']);end
            stack = dbstack;
            if any(strcmp({stack.name},'icaStream.get.icaweights'))
                obj.icaweights = icaweights;
                return;
            end
            obj.icaweights = icaweights;
            saveProperty(obj,'icaweights',icaweights)
        end
    end
    methods(Hidden)
        function properyArray = getPropertyGridField(obj)
            dim1 = size(obj.icawinv);
            dim2 = size(obj.icasphere);
            dim3 = size(obj.icaweights);
            precision = class(obj.icaweights);
            properyArray = [...
                PropertyGridField('icawinv',   ['<' num2str(dim1(1)) 'x' num2str(dim1(2)) ' ' precision '>'],'DisplayName','icawinv','ReadOnly',false,'Description','')...
                PropertyGridField('icasphere', ['<' num2str(dim2(1)) 'x' num2str(dim2(2)) ' ' precision '>'],'DisplayName','icasphere','ReadOnly',false,'Description','')...
                PropertyGridField('icaweights',['<' num2str(dim3(1)) 'x' num2str(dim3(2)) ' ' precision '>'],'DisplayName','icaweights','ReadOnly',false,'Description','')...
                ];
        end
    end
end