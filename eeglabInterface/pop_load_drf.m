%% pop_load_drf() - Import Datariver .DRF file
%
%% Description
% Imports Datariver .DRF or MOBILAB .mat file into EEGLAB. This function uses the class dataSourceDRF and dataSourceMAT respectively
% to load MoBI data. The program inserts in ALLEEG all the EEG data extracted from the file source and returns in EEG the 
% last one. The program also returns in allDataStreams object, a list of mobi handlers to the different streams imported.
%
%% Usage:
%   >> [ALLEEG EEG CURRENTSET,allDataStreams] = pop_load_drf(source,ALLEEG,EEG,CURRENTSET);
%
%% Inputs:
%   source             - .DRF file name
%
%% Outputs:
%   ALLEEG, EEG, and CURRENTSET - The updated EEGLAB standard structures.
%   allDataStreams                     - datasource containing the list of the whole MoBI data imported.     
%
% Author: Alejandro Ojeda SCCN, INC, UCSD, July 2006
%
% See also: eeglab()

%%
function allDataStreams = pop_load_drf(source,mobiDataDirectory,tmpDir)
if nargin < 1, source = [];end
if nargin < 2, mobiDataDirectory = [];end
if nargin < 3, tmpDir = [];end

flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag
    allDataStreams = evalin('base','allDataStreams');
else
    allDataStreams = [];
end

BDF = 0;
streamListFlag = false;
if ~exist(source,'file') || ~exist(mobiDataDirectory,'dir')    
    handle = ImportFromDatariver;
    uiwait(handle);
    userData = get(handle,'UserData');
    delete(handle);
    drawnow;
    if isempty(userData), return;end
    source = userData.source;
    mobiDataDirectory = userData.mobiDataDirectory;
    BDF = userData.BDF;
    streamListFlag = userData.streamListFlag;
end
[~,~,ext] = fileparts(source);

try
    if ~strcmp(ext,'.drf'), error(['Cannot read ' ext ' data']);end
    %hwait = mobilab.waitCircle('Reading...');
    allDataStreams = dataSourceDRF(source,mobiDataDirectory,tmpDir);
    allDataStreams.container.allStreams = allDataStreams;
    %try close(hwait);end %#ok
    
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
        rmIndex = 0;
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
        data_identifier = 'biosemi';
        for it=1:N
            if ~isempty(strfind(allDataStreams.item{it}.name,data_identifier))
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