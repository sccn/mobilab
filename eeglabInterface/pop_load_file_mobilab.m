%% pop_load_file_mobilab() - Imports multi-stream files into EEGLAB/MoBILAB
%
%% Description
% Imports any any supported file format, so far .xdf (Lab Stream Layer) and .drf (DataRiver), into EEGLAB/MoBILAB. This function
% uses the class dataSourceDRF. The program inserts in ALLEEG all the EEG data extracted from the file source and returns in EEG the 
% last one. The program also returns in allDataStreams object, a list of mobi handlers to the different stream objects imported.
%
%% Usage:
%   >> pop_load_file_mobilab( source);                 % creates a destinyFolder in the same directory named source-name_MoBI
%   >> pop_load_file_mobilab( source, destinyFolder);  % non existent destinyFolder?, creates it
%   >> pop_load_file_mobilab( source, destinyFolder);  % sourceFile's parent folder == destinyFolder?, throws an error
%   >> pop_load_file_mobilab;                              % pops up the gui

% If in any case destinyFolder exist and is not empty a warning message is produced saying that the content of the folder is about to
% be deleted, asking to the user if he/she wants to continue, any answer different than 'yes' will exit the program.
%
%
%% Inputs:
%   source             - .DRF file name
%   destinyFolder      - folder where dump all the files resulting from MoBILAB's multi-modal data analysis
%
%% Outputs:
%   ALLEEG, EEG, and CURRENTSET - The update of these variables is done implicitly, even when they do not return explicitly in the function's call.
%   allDataStreams              - datasource containing the list of the different stream objects imported.     
%
% Author: Alejandro Ojeda SCCN, INC, UCSD, July 2012

%%
function allDataStreams = pop_load_file_mobilab(source,mobiDataDirectory)
if nargin < 1, source = [];end
if nargin < 2, mobiDataDirectory = [];end
launchGui = false;
dispCommand = false;
streamListFlag = false;
allDataStreams = [];
flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag, allDataStreams = evalin('base','allDataStreams');end
if isempty(source) || ~exist(source,'file'), launchGui = true;end
if exist(source,'file') && ~exist(mobiDataDirectory,'dir') 
    [mobiDataDirectory,filename] = fileparts(source);
    mobiDataDirectory = fullfile(mobiDataDirectory,[filename '_MoBI']);
end
if exist(mobiDataDirectory,'dir')
    files = dir(mobiDataDirectory);
    files = {files.name};
    files(1:2) = [];
    if ~isempty(files)
        choice = input('The destiny folder is not empty. All its content will be removed. Would you like to continue? ','s');
        if ~strcmpi(choice,'yes'), return;end
    end
end
if launchGui
    handle = ImportFile2EEGLAB;
    uiwait(handle);
    userData = get(handle,'UserData');
    delete(handle);
    drawnow;
    if isempty(userData), return;end
    source = userData.source;
    mobiDataDirectory = userData.mobiDataDirectory;
    streamListFlag = userData.streamListFlag;
    dispCommand = true;
end
[~,~,ext] = fileparts(source);

try
    switch ext
        case '.xdf', 
            importFun = @dataSourceXDF;
            cmd = 'dataSourceXDF';
        case '.xdfz'
            importFun = @dataSourceXDF;
            cmd = 'dataSourceXDF';
        case '.set'
            importFun = @dataSourceSET;
            cmd = 'dataSourceSET';
        case '.drf'
            importFun = @dataSourceDRF;
            cmd = 'dataSourceDRF';
        case '.bdf'
            disp('To import .bdf files recorded by DataRiver use the option ''Import from DataRiver .bdf file'' or ''pop_load_bdf'' from the command line.')
            return;
        otherwise error('Unknown format.');
    end
        
    if dispCommand
        disp('Running:');
        disp(['  allDataStreams = ' cmd '( ''' source ''' , ''' mobiDataDirectory ''');' ]);
    end
    
    allDataStreams = importFun(source,mobiDataDirectory);
    allDataStreams.container.allStreams = allDataStreams;
    
    N = length(allDataStreams.item);
    if streamListFlag
        h = EventsToImport(allDataStreams);
        uiwait(h);
        streamList = get(findobj(h,'tag','listbox2'),'String');
        try close(h);end %#ok
    else streamList = [];
    end
    
    dataObjIndex = getItemIndexFromItemClass(allDataStreams,'eeg');
    dataObjIndex(isempty(dataObjIndex)) = 1;
    
    eventObjIndex = [];
    if isempty(streamList), streamList = {'snap-markers' 'eventcodes'};end
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
    
    if dispCommand
        disp('Running:');
        disp(['  allDataStreams.export2eeglab( [' num2str(dataObjIndex(:)') '] , [' num2str(eventObjIndex(:)') '] );']);
    end
    allDataStreams.export2eeglab(dataObjIndex,eventObjIndex);
catch ME
    if ~isempty(allDataStreams)
        N = length(allDataStreams.item);
        for it=1:N, if exist(allDataStreams.item{it}.binFile,'file'), java.io.File(allDataStreams.item{it}.binFile).delete();end;end
    end
    errordlg(ME.message);
end