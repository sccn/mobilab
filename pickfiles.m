function filesf = pickfiles(directory0,filtand,filtor,filtnot)

% Usage:
% files = pickfiles(directories,filtand,filtor,filtnot)
% 
% files : vector of files found
% directories: cell or vector with directories to look in
% filtand: cell or vector of simultaneously wanted strings
% filtor: cell or vector of optionally wanted strings
% filtnot: cell or vector of unwated strings
%
% files accomplish the logical statement (filtand(1) AND filtand(2) AND ...)
%          AND (filtor(1) OR filtor(2) OR ...) AND (~filtnot(1) AND ~filtnot(2) ...)
% 
% version 1.0
% Pedro Antonio Valdés-Hernández
% Cuban Neuroscience Center
% Havana, Cuba

filesf = '';
if iscell(directory0)
    directory0 = strvcat(directory0{:}); %#ok
end
for d = 1:size(directory0,1),
    directory = deblank(directory0(d,:));
    if iscell(filtand)
        filtand = strvcat(filtand{:}); %#ok
    end
    if nargin > 2 && iscell(filtor),
        filtor = strvcat(filtor{:}); %#ok
    end
    if nargin == 4 && iscell(filtnot),
        filtnot = strvcat(filtnot{:}); %#ok
    end
    files = get_all(directory); ind = [];
    for file = 1:size(files,1)
        chkand = false(zeros(1,size(filtand,1)));
        for i = 1:size(filtand,1)
            chkand(i) = ~isempty(findstr(deblank(files(file,:)),deblank(filtand(i,:))));
        end
        if all(chkand)
            if nargin > 2
                chkor = false(zeros(1,size(filtor,1)));
                for i = 1:size(filtor,1)
                    chkor(i) = ~isempty(findstr(deblank(files(file,:)),deblank(filtor(i,:))));
                end
            else
                chkor = true;
            end
            if nargin == 4
                chknot = false(zeros(1,size(filtnot,1)));
                for i = 1:size(filtnot,1)
                    chknot(i) = isempty(findstr(deblank(files(file,:)),deblank(filtnot(i,:))));
                end
            else
                chknot = true;
            end
            if all([all(chkand) any(chkor) all(chknot)])
                ind = [ind file]; %#ok
            end
        end
    end
    filesf = strvcat(filesf,files(ind,:)); %#ok
end
filesf = deblank(filesf);

function files = get_all(directory)

% Pick all files in a folder
% Pedro A Valdes-Hernandez 

directory = deblank(directory);
list = dirall(directory);
files = cell(length(list),1);
for i = 1:length(list)
    if ~list(i).isdir 
        files{i} = list(i).name;
    end
end
if ~isempty(list)
    files = strvcat(files{:}); %#ok
else
    files = []; 
end

function list = dirall(folder,level)

list = dir(folder); 
if strcmp([list(1:2).name],'...')
    list([1 2]) = [];
end
for i = 1:length(list), 
    list(i).name = sprintf('%s%s%s',folder,filesep,list(i).name);
end
if nargin == 2 && level == 1, return; end
for i = 1:length(list)
    if list(i).isdir
        switch nargin
            case 1
                newlist = dirall(list(i).name);
            case 2
                newlist = dirall(list(i).name,level-1);
        end
        list = [list; newlist]; %#ok
    end
end
