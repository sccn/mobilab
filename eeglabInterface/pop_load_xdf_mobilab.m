%% pop_load_xdf_mobilab() - Import.XDF file
%
% Based on load_xdf.m (Christian Kothe). See http://code.google.com/p/xdf/ for more information on XDF. 
%
%% Usage:
%   >> [ALLEEG EEG CURRENTSET,allDataStreams] = pop_load_drf(source,ALLEEG,EEG,CURRENTSET);
%
%% Inputs:
%   source             - .XDF or .XDFZ file name
%
%% Outputs:
%   ALLEEG, EEG, and CURRENTSET - The updated EEGLAB standard structures.
%   allDataStreams              - datasource containing the list of all the streams imported.     
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, July 2012
%
% See also: eeglab()

%%
function allDataStreams = pop_load_xdf_mobilab(source,mobiDataDirectory)
if nargin < 1, source = [];end
if nargin < 2, mobiDataDirectory = [];end

flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag
    allDataStreams = evalin('base','allDataStreams');
else
    allDataStreams = [];
end
dispCommand = false;
if ~exist(source,'file') || ~exist(mobiDataDirectory,'dir')    
    handle = ImportFromXDF2;
    uiwait(handle);
    userData = get(handle,'UserData');
    delete(handle);
    drawnow;
    if isempty(userData), return;end
    if ~isfield(userData,'source'), return;end
    if ~isfield(userData,'mobiDataDirectory'), return;end
    source = userData.source;
    mobiDataDirectory = userData.mobiDataDirectory;
    streamListFlag = userData.streamListFlag;
    dispCommand = true;
end
[~,~,ext] = fileparts(source);
try
    if ~any(ismember({'.xdf' '.xdfz'},ext)), error('MoBILAB:isNotXDF','Error in dataSourceXDF constructor.\n This class only reads .xdf or .xdfz files.');end
    
    if dispCommand
        disp('Running:');
        disp(['  allDataStreams = dataSourceXDF( ''' source ''' , ''' mobiDataDirectory ''');' ]);
    end
    
    allDataStreams = dataSourceXDF(source,mobiDataDirectory);
    allDataStreams.container.allStreams = allDataStreams;
    if streamListFlag
        h = EventsToImport(allDataStreams);
        uiwait(h);
        streamList = get(findobj(h,'tag','listbox2'),'String');
        try close(h);end %#ok
    else
        streamList = {'markers'};
    end
    
    N = length(allDataStreams.item);
    dataObjIndex = zeros(N,1);
    data_type = 'eeg';
    for it=1:N
        if strcmpi(allDataStreams.item{it}.hardwareMetaData.type,data_type)
            dataObjIndex(it) = it;
            break
        end
    end
    dataObjIndex(dataObjIndex==0) = [];
    
    % if isempty(dataObjIndex), warndlg2('There is no EEG here!!! You may try MoBILAB.gui');return;end
    if isempty(dataObjIndex), warndlg2('There is no EEG here!!!. EEG structure will be created with the first stream.');
        allDataStreams.export2eeglab(1);
        return;
    end
    
    eventObjIndex = zeros(N,1);
    for it=1:N
        I = ismember(streamList,lower(allDataStreams.item{it}.hardwareMetaData.type)) | ismember(streamList,lower(allDataStreams.item{it}.name));
        if any(I)
            eventObjIndex(it) = it;
        end
    end
    eventObjIndex(eventObjIndex==0) = [];
    
    if dispCommand
        disp('Running:');
        disp(['  allDataStreams.export2eeglab( [' num2str(dataObjIndex) '] , [' num2str(eventObjIndex(:)') '] );']);
    end
    
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