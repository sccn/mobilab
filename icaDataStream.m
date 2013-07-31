classdef icaDataStream < dataStream & icaFields
    methods
        function obj = icaDataStream(header)
            obj@dataStream(header);
            obj@icaFields(header);
        end
        %%
        function properyArray = getPropertyGridField(obj)
            properyArray1 = getPropertyGridField@dataStream(obj);
            properyArray2 = getPropertyGridField@icaFields(obj);
            properyArray = [properyArray1 properyArray2];
        end
    end
end