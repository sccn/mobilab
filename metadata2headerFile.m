function newHeader = metadata2headerFile(metadata)
metadata.hdrVersion = coreStreamObject.version;
metadata.notes = {};
if isfield(metadata,'dob'), metadata = rmfield(metadata,'dob');end
obj_properties = fieldnames(metadata);
[path,name] = fileparts(metadata.binFile);
metadata.header = fullfile(path,[name '.hdr']);
newHeader = metadata.header;
save(newHeader,'-mat','-struct','metadata',obj_properties{1});
for it=1:length(obj_properties)
    save(newHeader,'-mat','-append','-struct','metadata',obj_properties{it});
end