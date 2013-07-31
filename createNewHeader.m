function newHeader = createNewHeader(metadata)
s = dbstack;
fnames = {s.name};
if length(fnames) < 2
    prename = '';
elseif length(fnames{2}) >= 3
     prename = fnames{2}(1:3);
else prename = fnames{2}(1:end);
end
prename = [prename '_'];
metadata.uuid = generateUUID;
filepath = fileparts(metadata.binFile);
metadata.name = [prename metadata.name];
metadata.binFile = fullfile(filepath,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
allocateFile(metadata.binFile,metadata.precision, [length(metadata.timeStamp) metadata.numberOfChannels]);
newHeader = metadata2headerFile(metadata);