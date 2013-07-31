function folder = uigetdir2(title,startDir)
if nargin < 1, title = '';end
if nargin < 2
    cfgfile = [getHomeDir filesep 'MoBILAB.cfg'];
    if exist(cfgfile,'file')
        load(cfgfile,'-mat');
        if ~exist('cfg','var'), cfg.lastDir = pwd;end
        if ~isfield(cfg,'lastDir'), cfg.lastDir = pwd;end
    else
        cfg.lastDir = pwd;
    end
    startDir = cfg.lastDir;
end
folder = uigetdir(startDir,title);
if ~isnumeric(folder)
    cfg.lastDir = folder;
    save(cfgfile,'cfg','-mat');
end