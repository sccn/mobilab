function mObj = mergeStreams(streamList)

N = length(streamList);
for it=2:N
    if ~isa(streamList{it},class(streamList{1}))
        error('MoBILAB:mergeStream','Cannot merge streams from different type.');
    end
end

t = [];
I = [];
for it=1:N, 
    t = [t streamList{it}.timeStamp]; %#ok
    I = [I; it*ones(length(streamList{it}.timeStamp),1)]; %#ok
end  
[ts,loc] = sort(t);
[tu,locU] = unique(ts);
order = unique(I(loc));

metadata = streamList{1}.saveobj;
metadata.writable = true;
metadata.parentCommand.commandName = 'mergeStream';
metadata.parentCommand.uuid = streamList{1}.uuid;

bsObj = streamList{1}.segmentObj;
for it=2:N
    metadata.parentCommand.varargin{it-1} = streamList{it}.uuid;
    bsObj = cat(bsObj,streamList{order(it)}.segmentObj);
end

metadata.segmentObj = bsObj;
metadata.uuid = java.util.UUID.randomUUID;
path = fileparts(metadata.mmfName);
prename = 'merged_';
metadata.name = [prename metadata.name];
metadata.mmfName = fullfile(path,[metadata.name '_' char(metadata.uuid) '.bin']);
metadata.timeStamp = tu;
obj_properties = fieldnames(metadata);
obj_values     = struct2cell(metadata);
varargIn = cat(1,obj_properties,obj_values);
Np = length(obj_properties);
index = [1:Np; Np+1:2*Np];
varargIn = varargIn(index(:));

Zeros = zeros(length(metadata.timeStamp),1);
fid = fopen(metadata.mmfName,'w');
c = onCleanup(@()fclose(fid));
for it=1:streamList{1}.numberOfChannels, fwrite(fid,Zeros,streamList{1}.precision);end

constructorHandle = eval(['@' metadata.class]);
mObj = constructorHandle(varargIn{:});
streamList{1}.container.item{end+1} = mObj;

for ch=1:mObj.numberOfChannels
    val = [];
    for it=1:N, val = [val; streamList{it}.data(:,ch)];end %#ok
    mObj.data(:,ch) = val(loc(locU));
end

mObj.event = event;
for jt=1:N
    latency = streamList{jt}.timeStamp(streamList{jt}.event.latencyInFrame);
    if ~isempty(latency)
        latencyInsamples = mObj.getTimeIndex(latency);
        mObj.event = mObj.event.addEvent(latencyInsamples,streamList{jt}.event.label);
    end
end





