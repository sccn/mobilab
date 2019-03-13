%% pop_load_bdf() - Import Datariver .BDF file
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
% Author: Alejandro Ojeda, Nima Bigdely Shamlo, and Christian Kothe, SCCN, INC, UCSD, July 2006
%
% See also: eeglab()

%%
function allDataStreams = pop_load_bdf(source,mobiDataDirectory,configList)
if nargin < 1, source = [];end
if nargin < 2, mobiDataDirectory = [];end
if nargin < 3, configList = [];end

flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag
    allDataStreams = evalin('base','allDataStreams');
else allDataStreams = [];
end
dispCommand = false;
if ~exist(source,'file') || ~exist(mobiDataDirectory,'dir')
    handle = ImportFromDatariverBDF;
    uiwait(handle);
    userData = get(handle,'UserData');
    delete(handle);
    drawnow;
    if isempty(userData), return;end
    source = userData.source;
    mobiDataDirectory = userData.mobiDataDirectory;
    configList = userData.configList;
    dispCommand = true;
end
[~,~,ext] = fileparts(source);

if dispCommand
    disp('Running:');
    disp(['  allDataStreams = dataSourceBDF( ''' source ''' , ''' mobiDataDirectory ''',configList);' ]);
end

try
    if strcmp(ext,'.bdf')
        allDataStreams = dataSourceBDF(source,mobiDataDirectory,configList);
        allDataStreams.container.allStreams = allDataStreams;
    else error(['Cannot read ' ext ' files']);
    end
    allDataStreams.export2eeglab(1);
catch ME
    if ~isempty(allDataStreams)
        N = length(allDataStreams.item);
        for it=1:N
            if exist(allDataStreams.item{it}.binFile,'file')
                delete(allDataStreams.item{it}.binFile);
                delete(allDataStreams.item{it}.header);
            end
        end
    end
    errordlg(ME.message);
end