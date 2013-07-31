function drfFilename = writeDRF(streamObj,drfFilename)
if nargin < 2, error('Not enough input arguments.');end

if length(streamObj) == 1 && ~iscell(streamObj), streamObj = {streamObj};end

[drfFolder,drfName,ext] = fileparts(drfFilename); %#ok
if isempty(drfFolder), drfFolder = pwd;end
ext =  '.drf';
Nstreams = length(streamObj);

docNode = com.mathworks.xml.XMLUtils.createDocument('streamsample');
streamsample = docNode.getDocumentElement;
reservedItem = docNode.createElement('reserved');
reservedItem.setAttribute('bytes',num2str(streamObj{1}.hardwareMetaData.reserved_bytes));
streamsample.appendChild(reservedItem);

filesampleItem = docNode.createElement('filesample');
stream_count = docNode.createElement('stream_count');
stream_count.setAttribute('bytes','4');
filesampleItem.appendChild(stream_count);
sample = docNode.createElement('sample');

offset = zeros(streamObj{1}.hardwareMetaData.offset,1,'int8');
offset(1) = Nstreams;

% Uff this is so boring!!
for streamsIt=1:Nstreams
    if isempty(streamObj{streamsIt}.hardwareMetaData)
        error('''hardwareMetaData'' field is not available, you have to re-import the .drf file in order to get this info.');
    end
    stream = streamObj{streamsIt}.hardwareMetaData.insertInHeader(docNode);
    sample.appendChild(stream);
end
filesampleItem.appendChild(sample);
streamsample.appendChild(filesampleItem);

xmlwrite([drfFolder filesep drfName '.xml'],docNode);

eventChannel = cell(length(streamObj),1);
uV = cell(length(streamObj),1);
for streamsIt=1:Nstreams
    hardwareMetaDataObj = streamObj{streamsIt}.hardwareMetaData;
    eventChannel{streamsIt} = streamObj{streamsIt}.event.event2vector(streamObj{streamsIt}.timeStamp);
    if strcmp(hardwareMetaDataObj.name,'wii')
        uV{streamsIt} = hardwareMetaDataObj.uV(1);
    elseif length(hardwareMetaDataObj.uV)==1
        uV{streamsIt} = hardwareMetaDataObj.uV;
    else
        for kk=1:length(hardwareMetaDataObj.uV)
            uV{streamsIt} = [uV{streamsIt} repmat(hardwareMetaDataObj.uV(kk),1,hardwareMetaDataObj.count(kk))];
        end
    end
end

startLatency = streamObj{1}.timeStamp(1);
endLatency = streamObj{1}.timeStamp(end);
I = false(1,length(streamObj{1}.timeStamp));
I = I | streamObj{1}.timeStamp >= startLatency & streamObj{1}.timeStamp <= endLatency;

I = find(I);
N = length(I);
fid = fopen([drfFolder filesep drfName ext],'w');
hwait = waitbar(0,'writing binary data...','Color',[0.66 0.76 1]);
for it=1:N
    fwrite(fid,offset);
    for streamsIt=1:Nstreams
        hardwareMetaDataObj = streamObj{streamsIt}.hardwareMetaData;
        fwrite(fid,hardwareMetaDataObj.originalTimeStamp(I(it)),'int32');
        fwrite(fid,eventChannel{streamsIt}(I(it)),'int32');
        fwrite(fid,streamObj{streamsIt}.numberOfChannels,'int32');
        fwrite(fid,hardwareMetaDataObj.item_size,'int32');
        precision  = sprintf('int%i',8*hardwareMetaDataObj.item_size);
        if strcmp(streamObj{streamsIt}.isMemoryMappingActive,'not active')
            fwrite(fid,zeros(streamObj{streamsIt}.numberOfChannels,1,'single'),precision);
        else
            fwrite(fid,streamObj{streamsIt}.data(I(it),:).*uV{streamsIt},precision);
        end
    end
    waitbar(it/N,hwait);
end
close(hwait);
fclose(fid);
end