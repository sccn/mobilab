%% pop_load_MoBIf() - Import from MoBILAB
%
% Author: Alejandro Ojeda, Nima Bigdely Shamlo, and Christian Kothe, SCCN, INC, UCSD, July 2006
%
% See also: eeglab()

%%
function allDataStreams = pop_load_MoBI(mobiDataDirectory)
if nargin < 1, mobiDataDirectory = [];end

flag = evalin('base','exist(''allDataStreams'',''var'')');
if flag
    allDataStreams = evalin('base','allDataStreams');
else
    allDataStreams = [];
end

if ~exist(mobiDataDirectory,'dir') 
    mobiDataDirectory = uigetdir('Select the _MoBI folder');
    if isnumeric(mobiDataDirectory), return;end
    if ~exist(mobiDataDirectory,'dir'), return;end
    suffix = '_MoBI';
    suffixLength = length(suffix);
    if ~strcmp(mobiDataDirectory(end-suffixLength+1:end),suffix)
        errordlg([repmat(' ',1,18) 'This is not a _MoBI folder' repmat(' ',1,18)],'Error loading MoBILAB dataSource');
        return
    end
end

try
    allDataStreams = dataSourceMoBI(mobiDataDirectory);
    allDataStreams.container.allStreams = allDataStreams;
    dataObjIndex = getItemIndexFromItemClass(allDataStreams,'eeg');
    dataObjIndex(isempty(dataObjIndex)) = 1;
    N = length(allDataStreams.item);
    eventObjIndex = [];
    streamList = {'snap-markers' 'eventcodes'};
    for it=1:N
        for jt=1:length(streamList)
            if ~isempty(strfind(allDataStreams.item{it}.name,streamList{jt}))
                eventObjIndex(end+1) = it;%#ok
                break
            end
        end
    end
%     
%     if exist(allDataStreams.eeglabCacheFile,'file')
%         [~,name] = fileparts(allDataStreams.eeglabCacheFile);
%         EEG = pop_loadset([name '.set'],allDataStreams.mobiDataDirectory);
%         flag = false;
%         if isempty(EEG.filepath), EEG.filepath = allDataStreams.mobiDataDirectory;flag = true;end
%         if isempty(EEG.filename), EEG.filename = name;flag = true;end
%         if isempty(EEG.setname),  EEG.setname = name;flag = true;end
%         if flag, pop_saveset( EEG, [name '.set'],obj.mobiDataDirectory);end
%         try
%             ALLEEG = evalin('base','ALLEEG');
%         catch %#ok
%             ALLEEG = [];
%         end
%         [ALLEEG,~,CURRENTSET] = eeg_store(ALLEEG, EEG);
%         assignin('base','ALLEEG',ALLEEG);
%         assignin('base','CURRENTSET',CURRENTSET);
%         assignin('base','EEG',EEG);
%         evalin('base','eeglab(''redraw'');');
%     else
        if numel(dataObjIndex) > 1,
            disp(['The following items contain eeg data: ' num2str(dataObjIndex(:)')])
            dataObjIndex = input('  please enter the ones you want to export to EEGLAB:');
            if ischar(dataObjIndex), dataObjIndex = str2double(dataObjIndex);end
            if ~isnumeric(dataObjIndex), error('Wrong input.');end
        end
        allDataStreams.export2eeglab(dataObjIndex,eventObjIndex);
    %end
catch ME
    errordlg(ME.message,'pop_load_MoBI')
end
