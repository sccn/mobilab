function folder = getMobilabHomeFolder
homeDir = getHomeDir;
folder = [homeDir filesep '.mobilab'];
if ~exist(folder,'dir'), mkdir(folder);end
    