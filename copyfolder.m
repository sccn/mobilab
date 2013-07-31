function copyfolder(source,destination)
if ~exist(destination,'dir'), mkdir(destination);end
if strcmp(source,destination), return;end

list = dir(source);
list(1:2) = [];
if isempty(list), return;end

isFile = logical(~cell2mat({list.isdir}));
names = {list.name};
filesnames = names(isFile);
for it=1:length(filesnames)
   toCopy = fullfile(source,filesnames{it});
   disp(['cp: ' toCopy]);
   copyfile(toCopy,destination);
end

[~,destName] = fileparts(destination);
foldernames = names(~isFile);
for it=1:length(foldernames)
    if ~(strfind(destination,source) == 1 && strcmp(foldernames{it},destName))
        src  = fullfile(source,foldernames{it});
        dest = fullfile(destination,foldernames{it});
        copyfolder(src,dest);
    end
end