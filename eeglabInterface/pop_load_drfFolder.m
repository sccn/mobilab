function allDataStreams = pop_load_drfFolder(sourceDir,mobiDataDirectory)
if nargin < 1, sourceDir = [];end
if nargin < 2, mobiDataDirectory = [];end

flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag
    allDataStreams = evalin('base','allDataStreams');
else
    allDataStreams = [];
end
BDF = false;
streamListFlag = false;
if ~exist(sourceDir,'dir') || ~exist(mobiDataDirectory,'dir')
    handle = ImportFromDatariverFolder;
    uiwait(handle);
    userData = get(handle,'UserData');
    delete(handle);
    drawnow;
    if isempty(userData), return;end
    sourceDir = userData.sourceDir;
    mobiDataDirectory = userData.mobiDataDirectory;
    BDF = userData.BDF;
    streamListFlag = userData.streamListFlag;
end

try
    allDataStreams = dataSourceFromDRFFolder(sourceDir,mobiDataDirectory);
    allDataStreams.container.allStreams = allDataStreams;
    N = length(allDataStreams.item);    
    if streamListFlag
        h = EventsToImport(allDataStreams);
        uiwait(h);
        streamList = get(findobj(h,'tag','listbox2'),'String');
        try close(h);end %#ok
    else
        streamList = [];
    end
    
    if BDF
        rmIndex = [];
        dataObjIndex = 1:length(allDataStreams.item);
        data_identifier = 'eventcodes';
        for it=1:N
            if ~isempty(strfind(allDataStreams.item{it}.name,data_identifier))
                rmIndex = it;
                break
            end
        end
        dataObjIndex(dataObjIndex==rmIndex) = [];
    else
        dataObjIndex = 1;
        data_identifier = allDataStreams.item{1}.name;
        for it=1:N
            if strcmp(allDataStreams.item{it}.name,data_identifier)
                dataObjIndex = it;
                break
            end
        end
    end
    
    eventObjIndex = [];
    if isempty(streamList), streamList = {'biosemi' 'eventcodes'};end
    for it=1:N
        for jt=1:length(streamList)
            if ~isempty(strfind(allDataStreams.item{it}.name,streamList{jt}))
                eventObjIndex(end+1) = it;%#ok
                break
            end
        end
    end
    eventObjIndex = unique(eventObjIndex);
    if isempty(eventObjIndex), eventObjIndex = dataObjIndex;end 
           
    allDataStreams.export2eeglab(dataObjIndex,eventObjIndex);
catch ME
    if ~isempty(allDataStreams)
        N = length(allDataStreams.item);
        for it=1:N
            if exist(allDataStreams.item{it}.mmfName,'file')
                delete(allDataStreams.item{it}.mmfName);
            end
        end
        if exist(allDataStreams.dataSourceLocation,'file'), delete(allDataStreams.dataSourceLocation);end
    end
    errordlg(ME.message);
end
