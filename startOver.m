function startOver(inputDir)
files = pickfiles(inputDir,'descriptor.MoBI');
N = size(files,1);
mobiFolder = cell(N,1);
loc = zeros(N,1);
for it=1:N
    ind = find(files(it,:) == filesep);
    mobiFolder{it} = files(it,1:ind(end)-1);
    ind2 = strfind(files(it,:),'_MoBI')';
    loc(it) = str2double(files(it,ind(end-1)+1:ind2(end)-1));
end
[~,loc] = sort(loc);
mobiFolder = mobiFolder(loc);

for k=1:length(mobiFolder)
    [~,name] = fileparts(mobiFolder{k});
    disp(['Cleaning case: ' name]);
    
    allDataStreams = dataSourceMoBI(mobiFolder{k});
    %allDataStreams.segmentList = segmentList(allDataStreams);
    items2keep = allDataStreams.gObj.getDescendants(1)-1;
    if items2keep(end) < length(allDataStreams.item)
        allDataStreams.deleteItem(items2keep(end)+1:length(allDataStreams.item));
    end
    warning off %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'X.mat']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'Xtw.mat']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'state.mat']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'mds_Mahalanobis.fig']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'mds_Minkowski.fig']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'similarity_matrix_Mahalanobis.fig']);end %#ok
    try delete([allDataStreams.mobiDataDirectory filesep 'similarity_matrix_Minkowski.fig']);end %#ok
    warning on %#ok
    delete(allDataStreams);
    clear allDataStreams;
end