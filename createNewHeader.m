function newHeader = createNewHeader(obj,nameNewObj)
metadata = load( obj.header, '-mat');
metadata.header = obj.header;
metadata.binFile = obj.binFile;
metadata.name = 'my_new_object';
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
metadata.name = [prename nameNewObj];
metadata.binFile = fullfile(filepath,[metadata.name '_' metadata.uuid '_' metadata.sessionUUID '.bin']);
allocateFile(metadata.binFile,metadata.precision, [length(metadata.timeStamp) metadata.numberOfChannels]);
newHeader = metadata2headerFile(metadata);