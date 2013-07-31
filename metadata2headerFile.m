function newHeader = metadata2headerFile(metadata)
try mobilab = evalin('base','mobilab');
catch
    load([getHomeDir filesep '.mobilab.mat'],'-mat')
    mobilab.preferences.username     = configuration.username;
    mobilab.preferences.organization = configuration.organization;
    mobilab.preferences.email        = configuration.email;
end
metadata.owner.name         = mobilab.preferences.username;
metadata.owner.organization = mobilab.preferences.organization;
metadata.owner.email        = mobilab.preferences.email;

metadata.hdrVersion = coreStreamObject.version;
metadata.notes = {};
if isfield(metadata,'dob'), metadata = rmfield(metadata,'dob');end
obj_properties = fieldnames(metadata);
[path,name] = fileparts(metadata.binFile);
metadata.header = fullfile(path,[name '.hdr']);
newHeader = metadata.header;
save(newHeader,'-mat','-struct','metadata',obj_properties{1});
for it=1:length(obj_properties), save(newHeader,'-mat','-append','-struct','metadata',obj_properties{it});end