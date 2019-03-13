function mocap_pipeline(dataDirectory,mobiDir)
if nargin < 2, mobiDir = fullfile(dataDirectory,'results');end
outFolder = fullfile(mobiDir,'setFiles');
if ~exist(mobiDir,'dir'), mkdir(mobiDir);end
if ~exist(outFolder,'dir'), mkdir(outFolder);end
myCodes = {'2' '3' '4' '5' '6' '7' '9' '10' '11' '12'};
endEventCodes   = {'204'};
checkSegmentLength = false;
numberOfGaussianMixtures = 2;
robustPcaFlag = true;
MaxIter = 2;
updateEEGLABGui = false;

%% reading csv file
csvfile = pickfiles(dataDirectory,'sample_data.csv');
csvfile = deblank(csvfile(end,:));
fid = fopen(csvfile,'r');
fgetl(fid);
count = 1;
OffLatency = zeros(100,1);
MLength = zeros(100,1);
while ~feof(fid)
    s = fgetl(fid);
    ind = find(s==',');
    OffLatency(count) = str2double(s(ind(2)+1:ind(3)-1));
    MLength(count) = str2double(s(ind(3)+1:ind(4)-1));
    count = count+1;
end
fclose(fid);
OffLatency(OffLatency==0) = [];
MLength(MLength==0) = [];

logFile = fullfile(mobiDir,'logfile.txt');
fid = fopen(logFile,'w');

c = onCleanup(@()fclose(fid));

folders = dir(dataDirectory);
folders(1:2) = [];
I = cell2mat({folders.isdir});
folders = {folders.name};
folders(~I) = [];
if strcmp(folders{end},'setFiles'), folders(end) = [];end
    

% drfFiles = pickfiles(dataDirectory,'.drf');
for k=1:length(folders)
    %[~,filename] = fileparts(deblank(drfFiles(k,:)));
    %foldername = [filename '_MoBI'];
    filename = folders{k}(1:strfind(folders{k},'_')-1);
    foldername = folders{k};
    try
        disp(['Processing case: ' foldername]);
        tmpDir = fullfile(mobiDir,foldername);
        
        %% loading up the data
%         try
            allDataStreams = dataSourceMoBI(tmpDir);
%         catch %#ok
%             if ~exist(tmpDir,'dir'), mkdir(tmpDir);end
%             file = deblank(drfFiles(k,:));
%             allDataStreams = dataSourceDRF(file,tmpDir);            
%         end
        
        %% checking current state
        stateFile = fullfile(allDataStreams.mobiDataDirectory,'state.mat');
        if exist(stateFile,'file'),
            load(stateFile);
            
            if ~strcmp(state.accomplished,'done')
                mocapIndex = allDataStreams.getItemIndexFromItemNameSimilarTo('phasespace');
                if isempty(mocapIndex), mocapIndex = allDataStreams.getItemIndexFromItemNameSimilarTo('mocap');end
                mocapIndex = mocapIndex(end);
                allDataStreams.deleteItem(mocapIndex:length(allDataStreams.item));
                clear state;
                state.accomplished = 'none';
                state.noCutId = [];
                state.id = [];
                state.Ids2merge = [];
            end
            
            if isfield(state,'obj')
                tmpIndex = allDataStreams.findItem(state.obj);
                state.obj = allDataStreams.item{tmpIndex};
            end
            if isfield(state,'mocapObj')
                tmpIndex = allDataStreams.findItem(state.mocapObj);
                state.mocapObj = allDataStreams.item{tmpIndex};
            end
        else
            state.accomplished = 'none';
            state.noCutId = [];
            state.id = [];
            state.Ids2merge = [];
            if isfield(state,'mocapObj'), state = rmfield(state,'mocapObj');end
        end
        state.file = fullfile(allDataStreams.mobiDataDirectory,'state.mat');
        
        if ~isfield(state,'id'), state.id = [];end
        if ~isfield(state,'noCutId'), state.noCutId = [];end 
        if ~isfield(state,'Ids2merge'), state.Ids2merge = [];end 
        if ~isfield(state,'rmThis'), state.rmThis = [];end 
        [~,gObj] = allDataStreams.viewLogicalStructure('',false);
        children = gObj.getDescendants(1)-1;
        
        for it=1:length(children)
            if ~isempty(strfind(allDataStreams.item{children(it)}.name,'biosemi')) || ~isempty(strfind(allDataStreams.item{children(it)}.name,'eeg'))
                eegItem = children(it);
            elseif ~isempty(strfind(allDataStreams.item{children(it)}.name,'phasespace')) || ~isempty(strfind(allDataStreams.item{children(it)}.name,'mocap'))
                mocapItem = children(it);
            elseif ~isempty(strfind(allDataStreams.item{children(it)}.name,'audiosend'))
                audiosendIndex = children(it);
            end
        end
        I = ismember(myCodes,allDataStreams.item{audiosendIndex}.event.uniqueLabel);
        startEventCodes = myCodes(I);
        offLatency = OffLatency(I);
        mLength = MLength(I);
        N = length(startEventCodes);
        segmentCount = 1;
        X = [];
        if isfield(state,'segmentCount'), segmentCount = state.segmentCount;end
        iter = 0;
        while ~strcmp(state.accomplished,'done') && iter < MaxIter
            try
                switch state.accomplished
                    case 'none'
                        state.obj = allDataStreams.item{mocapItem}.removeOcclusionArtifact;
                        state.accomplished = 'removeOcclusionArtifact';
                    case 'removeOcclusionArtifact'
                        state.obj = state.obj.lowpass;
                        state.mocapObj = state.obj;
                        segmentCount = 1;
                        state.segmentCount = segmentCount;
                        state.accomplished = 'lowpass';  % next time jump directly to segmenting
                    case 'lowpass'
                        try
                            bsObj = basicSegment(allDataStreams.item{audiosendIndex}, startEventCodes(segmentCount), endEventCodes, startEventCodes{segmentCount}, checkSegmentLength);
                            dLatency = diff(bsObj.startLatency);
                            endFirstBlock = find(dLatency > 3*median(dLatency));
                            if isempty(endFirstBlock), endFirstBlock = length(bsObj.startLatency);end
                            if endFirstBlock < 4, endFirstBlock = length(bsObj.startLatency);end
                            
                            if length(bsObj.startLatency) < 8
                                latenciesE  = [bsObj.startLatency(end-3:end-2)' bsObj.endLatency(end-3:end-2)'];
                                latenciesNE = [bsObj.startLatency(end-1:end)' bsObj.endLatency(end-1:end)'];
                            else
                                latenciesE = [bsObj.startLatency(endFirstBlock-3:endFirstBlock-2)' bsObj.endLatency(endFirstBlock-3:endFirstBlock-2)';...
                                    bsObj.startLatency(end-3:end-2)' bsObj.endLatency(end-3:end-2)'];
                                
                                latenciesNE = [bsObj.startLatency(endFirstBlock-1:endFirstBlock)' bsObj.endLatency(endFirstBlock-1:endFirstBlock)';...
                                    bsObj.startLatency(end-1:end)' bsObj.endLatency(end-1:end)'];
                            end
                            bsObjNoCut = basicSegment(latenciesE, ['e-' startEventCodes{segmentCount}]);
                            latenciesE(:,1) = latenciesE(:,1) + offLatency(segmentCount);
                            latenciesE(:,2) = latenciesE(:,1) + mLength(segmentCount);
                            bsObj = basicSegment(latenciesE, ['short-e-' startEventCodes{segmentCount}]);
                        catch ME
                            if strcmp(ME.identifier,'MATLAB:badsubscript')
                                segmentCount = segmentCount+1;
                                state.segmentCount = segmentCount;
                                if segmentCount > N
                                    state.accomplished = 'segmentingNE';
                                    save(fullfile(allDataStreams.mobiDataDirectory,'X.mat'),'X');
                                end
                                saveLatestState(state);
                            end
                            ME.rethrow;
                        end
                        state.obj = bsObj.apply( state.mocapObj,[1 2 3]);
                        state.rmThis{end+1} = char(state.obj.uuid);
                        state.obj = state.obj.projectDataPCA(robustPcaFlag,numberOfGaussianMixtures);
                        R = state.obj.projectionMatrix;
                        state.obj.smoothDerivative;
                        state.id{end+1} = char(state.obj.uuid);
                        
                        state.obj = bsObjNoCut.apply(state.mocapObj,[1 2 3]);
                        state.obj = state.obj.projectData(R,robustPcaFlag,numberOfGaussianMixtures);
                        state.obj.smoothDerivative;
                        state.noCutId{end+1} = char(state.obj.uuid);
                        state.Ids2merge{end+1} = char(state.obj.uuid);
                        
                        allDataStreams.segmentList.addSegment(bsObjNoCut);
                        
                        state.accomplished = 'segmentingE';
                    case 'segmentingE'
                        bsObjNoCut = basicSegment(latenciesNE, ['ne-' startEventCodes{segmentCount}]);
                        latenciesNE(:,1) = latenciesNE(:,1) + offLatency(segmentCount);
                        latenciesNE(:,2) = latenciesNE(:,1) + mLength(segmentCount);
                        bsObj = basicSegment(latenciesNE, ['short-ne-' startEventCodes{segmentCount}]);
                        
                        state.obj = bsObj.apply( state.mocapObj,[1 2 3]);
                        state.rmThis{end+1} = char(state.obj.uuid);
                        state.obj = state.obj.projectDataPCA(robustPcaFlag,numberOfGaussianMixtures);
                        R = state.obj.projectionMatrix;
                        state.obj.smoothDerivative;
                        state.id{end+1} = char(state.obj.uuid);
                        
                        state.obj = bsObjNoCut.apply(state.mocapObj,[1 2 3]);
                        state.obj = state.obj.projectData(R,robustPcaFlag,numberOfGaussianMixtures);
                        state.obj.smoothDerivative;
                        state.noCutId{end+1} = char(state.obj.uuid);
                        state.Ids2merge{end+1} = char(state.obj.uuid);
                        
                        index = allDataStreams.findItem(state.id);
                        indexNoCut = allDataStreams.findItem(state.noCutId);
                        Ns = length(index);
                        
                        fprintf('Collecting swings: ');
                        for jt=1:Ns
                            fprintf(' %s ',allDataStreams.item{indexNoCut(jt)}.segmentObj.segmentName);
                            state.obj = allDataStreams.item{index(jt)};
                            X{end+1} = make_dataset(state.obj); %#ok
                            state.obj = allDataStreams.item{indexNoCut(jt)};
                            latencyInsamples = state.obj.getTimeIndex(X{end}.rl.latency(:,1));
                            state.obj.event = state.obj.event.addEvent(latencyInsamples,[state.obj.segmentObj.segmentName '-r' ]);
                            
                            latencyInsamples = state.obj.getTimeIndex(X{end}.lr.latency(:,1));
                            state.obj.event = state.obj.event.addEvent(latencyInsamples,[state.obj.segmentObj.segmentName '-l' ]);
                        end
                        fprintf('\n');
                        items2rm = allDataStreams.findItem(state.rmThis);
                        allDataStreams.deleteItem(items2rm);
                        state.id = [];
                        state.noCutId = [];
                        state.rmThis = [];
                        
                        allDataStreams.segmentList.addSegment(bsObjNoCut);
                        
                        segmentCount = segmentCount+1;
                        state.segmentCount = segmentCount;
                        if segmentCount > N
                            save(fullfile(allDataStreams.mobiDataDirectory,'X.mat'),'X');
                            state.accomplished = 'segmentingNE';
                        else
                            state.accomplished = 'lowpass';
                        end
                    case 'segmentingNE'
                        disp('Warping segments')
                        if exist(fullfile(allDataStreams.mobiDataDirectory,'X.mat'),'file')
                            load(fullfile(allDataStreams.mobiDataDirectory,'X.mat'));
                            [Xtw,labels_tw] = timeWarpSwings(X,state.obj.samplingRate);
                            save(fullfile(allDataStreams.mobiDataDirectory,'Xtw.mat'),'Xtw','labels_tw');
                            % state.accomplished = 'timeWarping';
                            state.accomplished = 'mds';
                        else
                            state.accomplished = 'segmentingNE';
                        end
                        state.accomplished = 'mds';
                    case 'timeWarping'
                        if exist(fullfile(allDataStreams.mobiDataDirectory,'Xtw.mat'),'file')
                            load(fullfile(allDataStreams.mobiDataDirectory,'Xtw.mat'));
                            
                            [Y,D] = mds(Xtw,'mahalanobis');  % Mahalanobis distance
                            [h1,h2] = plotMDS(Y,D,labels_tw);
                            disp('Saving images...')
                            disp(['   ' fullfile(allDataStreams.mobiDataDirectory,'mds_Mahalanobis.fig')]);
                            try saveas(h1,fullfile(allDataStreams.mobiDataDirectory,'mds_Mahalanobis.fig'));close(h1);end %#ok
                            disp(['   ' fullfile(allDataStreams.mobiDataDirectory,'similarity_matrix_Mahalanobis.fig')])
                            try saveas(h2,fullfile(allDataStreams.mobiDataDirectory,'similarity_matrix_Mahalanobis.fig'));close(h2);end %#ok
                            
                            [Y,D] = mds(Xtw,'minkowski',1.414);  % Minkowski order p
                            [h1,h2] = plotMDS(Y,D,labels_tw);
                            disp('Saving images...')
                            disp(['   ' fullfile(allDataStreams.mobiDataDirectory,'mds_Minkowski.fig')]);
                            try saveas(h1,fullfile(allDataStreams.mobiDataDirectory,'mds_Minkowski.fig'));close(h1);end %#ok
                            disp(['   ' fullfile(allDataStreams.mobiDataDirectory,'similarity_matrix_Minkowski.fig')])
                            try saveas(h2,fullfile(allDataStreams.mobiDataDirectory,'similarity_matrix_Minkowski.fig'));close(h2);end %#ok
                            state.accomplished = 'mds';
                        else
                            state.accomplished = 'timeWarping';
                        end
                    case 'mds'
                        indexNoCut = allDataStreams.findItem(state.Ids2merge);
                        state.obj = mergeStreams(allDataStreams.item(indexNoCut));
                        state.obj.smoothDerivative;
                        [~,filtEEGobj] = filter(allDataStreams.item{eegItem},'highpass',1);
                        mskEEGobj = maskStream(filtEEGobj,state.obj.timeStamp);
                        indexItems2Export(1) = allDataStreams.findItem(mskEEGobj.uuid); % eeg
                        indexItems2Export(2) = allDataStreams.findItem(state.obj.uuid);    % mocap pca
                        indexItems2Export(3) = indexItems2Export(2)+1;                  % velocity
                        indexItems2Export(4) = indexItems2Export(2)+2;                  % acceleration
                        indexItems2Export(5) = indexItems2Export(2)+3;                  % jerk
                        
                        allDataStreams.export2eeglab(indexItems2Export,indexItems2Export,[outFolder filesep filename '.set'],updateEEGLABGui);
                        state.accomplished = 'done';
                end
                saveLatestState(state);
                iter = 0;
            catch ME
                disp(ME.message)
                if strcmp(ME.message,'Undefined function or variable "latenciesNE".')
                    bsObj = basicSegment(allDataStreams.item{audiosendIndex}, startEventCodes(segmentCount), endEventCodes, startEventCodes{segmentCount}, checkSegmentLength);
                    dLatency = diff(bsObj.startLatency);
                    endFirstBlock = find(dLatency > 3*median(dLatency));
                    if isempty(endFirstBlock), endFirstBlock = length(bsObj.startLatency);end
                    if endFirstBlock < 4, endFirstBlock = length(bsObj.startLatency);end
                    
                    if length(bsObj.startLatency) < 8
                        latenciesE  = [bsObj.startLatency(end-3:end-2)' bsObj.endLatency(end-3:end-2)'];
                        latenciesNE = [bsObj.startLatency(end-1:end)' bsObj.endLatency(end-1:end)'];
                    else
                        latenciesE = [bsObj.startLatency(endFirstBlock-3:endFirstBlock-2)' bsObj.endLatency(endFirstBlock-3:endFirstBlock-2)';...
                            bsObj.startLatency(end-3:end-2)' bsObj.endLatency(end-3:end-2)'];
                        
                        latenciesNE = [bsObj.startLatency(endFirstBlock-1:endFirstBlock)' bsObj.endLatency(endFirstBlock-1:endFirstBlock)';...
                            bsObj.startLatency(end-1:end)' bsObj.endLatency(end-1:end)'];
                    end
                    bsObjNoCut = basicSegment(latenciesE, ['e-' startEventCodes{segmentCount}]);
                    latenciesE(:,1) = latenciesE(:,1) + offLatency(segmentCount);
                    latenciesE(:,2) = latenciesE(:,1) + mLength(segmentCount);
                    bsObj = basicSegment(latenciesE, ['short-e-' startEventCodes{segmentCount}]);
                end
                fprintf(fid,'Error in object %s. Latest accomplished: %s.\n',state.obj.name,state.accomplished);
                iter = iter+1;
                if strcmp(state.accomplished,'segmentingE') && iter > 1
                    items2rm = allDataStreams.findItem(state.rmThis);
                    allDataStreams.deleteItem(items2rm);
                    state.id = [];
                    state.noCutId = [];
                    state.rmThis = [];
                    segmentCount = segmentCount+1;
                    state.segmentCount = segmentCount;
                    if segmentCount > N
                        state.accomplished = 'segmentingNE';
                    else
                        state.accomplished = 'lowpass';
                    end
                    iter = 0;
                end
            end
        end
    catch ME
        disp(ME.message)
        fprintf(fid,'Error in %s, message: %s\n',[num2str(k) '_MoBI'],ME.message);
    end
    disp([num2str(k) ': ' foldername ' => ' state.accomplished]);
    try delete(allDataStreams);end %#ok
    clear allDataStreams
end
disp('Done!!!')
end

%% ----------------------------
function saveLatestState(state)

if isfield(state,'mocapObj'), state.mocapObj = state.mocapObj.uuid;end
if isfield(state,'obj'), state.obj = state.obj.uuid;end
save(state.file,'state');
end
%%
function X = make_dataset(prjObj)
percentileSignificant = 50;
windowWidth = 0.5; % half a second

prjObjIndex = prjObj.container.findItem(prjObj.uuid);
[~,BGobj] = prjObj.container.viewLogicalStructure('',false);
delList = getDescendants(BGobj,prjObjIndex+1)-1;
data(:,:,1) = prjObj.dataInXY;
data(:,:,2) = prjObj.container.item{delList(1)}.dataInXY;
data(:,:,3) = prjObj.container.item{delList(2)}.dataInXY;
data(:,:,4) = prjObj.container.item{delList(3)}.dataInXY;


% %--
% options = statset('Display','final','Robust','on');
% gmObj = gmdistribution.fit(data(:,:,1),3,'Options',options);
% idx = cluster(gmObj,data(:,:,1));
% ne = [sum(idx==1) sum(idx==2) sum(idx==3)];
% [~,loc1] = max(gmObj.PComponents);
% [~,loc2] = max(ne);
% if loc1 == loc2, I = idx==loc1;else I = true(size(data,1),1);end
% %--

curvature = prjObj.curvature;
velocity = prjObj.container.item{delList(1)}.magnitude;
th = prctile(curvature,percentileSignificant); %flattens magnitudes of curvature less than 75th percentile
curvature(curvature < th) = 0;
Ic = searchInSegment( curvature, 'maxima',128 );
Iv = searchInSegment( velocity,  'minima',128);

mask = false(size(prjObj,1),1);
halfWindow = round(windowWidth*prjObj.samplingRate/2);
for jt=1:length(Ic)
    mask(Ic(jt)-halfWindow:Ic(jt)+halfWindow) = true;
    [~,~,loc2] = intersect(Ic(jt)-halfWindow:Ic(jt)+halfWindow,Iv);
    
    [~,loc] = max(velocity(Iv(loc2)));
    mx(Iv(loc2(loc))) = Inf; %#ok
end

mx(mx~=Inf) = 0;
mx(mx==Inf) = 1;
latency = find(mx);

rmThis = data(latency,2,1) < 1.5*median(prjObj.dataInXY(:,2));
latency(rmThis) = [];

pointersR = [];
pointersL = [];
for it=1:length(latency)-2
    if data(latency(it),1,1) > 0 && data(latency(it+1),1,1) < 0
        pointersR(end+1) = latency(it); %#ok
    elseif data(latency(it),1,1) < 0 && data(latency(it+1),1,1) > 0
        pointersL(end+1) = latency(it); %#ok
    end
end

%- RL
prjObj.event = prjObj.event.addEvent(pointersR,'Rstart');
prjObj.event = prjObj.event.addEvent(pointersL,'Rend');

bsObj = basicSegment(prjObj,{'Rstart'},{'Rend'});
prjObj.event = prjObj.event.deleteAllEventsWithThisLabel('Rstart');
prjObj.event = prjObj.event.deleteAllEventsWithThisLabel('Rend');

latencyRL = [bsObj.startLatency' bsObj.endLatency'];
latencyInSamplesRL = reshape(prjObj.getTimeIndex(latencyRL(:)),size(latencyRL));

% prjObj.event = prjObj.event.addEvent(latencyInSamples(:,1),'Rstart');
% prjObj.event = prjObj.event.addEvent(latencyInSamples(:,2),'Rend');
% prjObj.projectionBrowser;

XRL = cell(size(latencyRL,1),1);
tempoRL = diff(latencyRL,[],2);
for it=1:size(latencyRL,1)
    XRL{it} = data(latencyInSamplesRL(it,1):latencyInSamplesRL(it,2),:);
end

%- LR
prjObj.event = prjObj.event.addEvent(pointersL,'Lstart');
prjObj.event = prjObj.event.addEvent(pointersR,'Lend');

bsObj = basicSegment(prjObj,{'Lstart'},{'Lend'});
prjObj.event = prjObj.event.deleteAllEventsWithThisLabel('Lstart');
prjObj.event = prjObj.event.deleteAllEventsWithThisLabel('Lend');

latencyLR = [bsObj.startLatency' bsObj.endLatency'];
latencyInSamplesLR = reshape(prjObj.getTimeIndex(latencyLR(:)),size(latencyLR));

% prjObj.event = prjObj.event.addEvent(latencyInSamples(:,1),'Lstart');
% prjObj.event = prjObj.event.addEvent(latencyInSamples(:,2),'Lend');
% prjObj.projectionBrowser;

XLR = cell(size(latencyLR,1),1);
tempoLR = diff(latencyLR,[],2);
for it=1:size(latencyLR,1)
    XLR{it} = data(latencyInSamplesLR(it,1):latencyInSamplesLR(it,2),:);
end

X.rl.data = XRL;
X.rl.swingLength = tempoRL;
X.rl.latency = latencyRL;

X.lr.data = XLR;
X.lr.swingLength = tempoLR;
X.lr.latency = latencyLR;

ind = find(prjObj.segmentObj.segmentName=='-');
X.label = prjObj.segmentObj.segmentName(ind+1:end);
X.maxLength = max([diff(latencyRL,[],2);diff(latencyInSamplesLR,[],2)]);
end
%%
function [Xtw,labels_tw] = timeWarpSwings(X,samplingRate)
%% time warping (it takes into account the length of each trial)
average_opt = 2;
Ns = length(X);
maxLength = zeros(Ns,1);
for jt=1:Ns, maxLength(jt) = X{jt}.maxLength;end
maxLength = max(maxLength);
Xtw = [];
labels_tw = {};
for jt=1:Ns
    k = X{jt}.rl.swingLength/(maxLength/samplingRate);
    ntr = length(X{jt}.rl.data);
    x = zeros(maxLength,2,4,ntr);
    for h=1:ntr
        n1 = size(X{jt}.rl.data{h},1);
        t1 = linspace(1,maxLength,n1)';
        tmp = interp1(t1,X{jt}.rl.data{h},(1:maxLength)');
        x(:,:,:,h) = reshape(tmp,[maxLength 2 4]);
        
        %x(:,:,2,h) = x(:,:,2,h)*k(h);
        %x(:,:,3,h) = x(:,:,3,h)*k(h).^2;
        %x(:,:,4,h) = x(:,:,4,h)*k(h).^3;
    end
    kk = squeeze(sqrt(sum(x.^2,2)));
    tmp2.rl = [];
    tmp2.rl(:,1,:) = squeeze(x(:,1,1,:));
    tmp2.rl(:,2,:) = squeeze(x(:,2,1,:));
    tmp2.rl(:,3,:) = kk(:,2,:);
    tmp2.rl(:,4,:) = kk(:,3,:);
    tmp2.rl(:,5,:) = kk(:,4,:);
    
    k = X{jt}.lr.swingLength/(maxLength/samplingRate);
    ntl = length(X{jt}.lr.data);
    x = zeros(maxLength,2,4,ntl);
    for h=1:ntl
        n1 = size(X{jt}.lr.data{h},1);
        t1 = linspace(1,maxLength,n1)';
        tmp = interp1(t1,X{jt}.lr.data{h},(1:maxLength)');
        x(:,:,:,h) = reshape(tmp,[maxLength 2 4]);
        
        %x(:,:,2,h) = x(:,:,2,h)*k(h);
        %x(:,:,3,h) = x(:,:,3,h)*k(h).^2;
        %x(:,:,4,h) = x(:,:,4,h)*k(h).^3;
    end
    kk = squeeze(sqrt(sum(x.^2,2)));
    tmp2.lr = [];
    tmp2.lr(:,1,:) = squeeze(x(:,1,1,:));
    tmp2.lr(:,2,:) = squeeze(x(:,2,1,:));
    tmp2.lr(:,3,:) = kk(:,2,:);
    tmp2.lr(:,4,:) = kk(:,3,:);
    tmp2.lr(:,5,:) = kk(:,4,:);
    
    if average_opt == 1
        tmp2.rl = mean(tmp2.rl,3);
        tmp2.lr = mean(tmp2.lr,3);
        
        Xtw(:,:,end+1) = tmp2.rl; %#ok
        Xtw(:,:,end+1) = tmp2.lr; %#ok
        labels_tw{end+1} = [X{jt}.label '-r']; %#ok
        labels_tw{end+1} = [X{jt}.label '-l']; %#ok
    else
        Xtw(:,:,end+1:end+1+ntr-1) = tmp2.rl;
        Xtw(:,:,end+1:end+1+ntl-1) = tmp2.lr;
        labels_tw = cat(1,labels_tw,repmat({[X{jt}.label '-r']},ntr,1));
        labels_tw = cat(1,labels_tw,repmat({[X{jt}.label '-l']},ntl,1));
    end
end
Xtw(:,:,1) = [];
end
%%
function [h1,h2] = plotMDS(Y,D,labels,colormap)
if nargin < 4, colormap = 'hsv';end
ntrials = length(labels);
uLabels = unique(labels);
color = eval([colormap '(length(uLabels));']);
warning off
h1 = figure;hold on;
for jt=1:ntrials
    I = ismember(uLabels,labels(jt));
    plot3(Y(jt,1),Y(jt,2),Y(jt,3),'o','Color',color(I,:),'MarkerFaceColor',color(I,:),'MarkerSize',25,'MarkerEdgeColor','k')
    text('Position',Y(jt,1:3),'String',labels{jt},'Color','k','FontWeight','bold')
end
title('MDS');
xlabel('X'),ylabel('Y');zlabel('Z');
grid on

h2 = figure;
imagesc(abs(D));
title('Similarity matrix')
warning on
end
%%
function mObj = maskStream(obj,timeMask)
loc = ismember(obj.timeStamp,timeMask);
if ~any(loc)
    error('MoBILAB:maskStream','Time stamps don''t match  at all.');
end
I = diff(loc);
seMatrix = obj.timeStamp([find(I==1)'+1 find(I==-1)'+1]);
bsObj = basicSegment(seMatrix,'msk');
mObj = bsObj.apply(obj);
end
%%
function mObj = mergeStreams(streamList)
N = length(streamList);
for it=2:N
    if ~isa(streamList{it},class(streamList{1}))
        error('MoBILAB:mergeStream','Cannot merge streams from different type.');
    end
end
t = [];
I = [];
for it=1:N 
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
end