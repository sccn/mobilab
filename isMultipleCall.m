function flag = isMultipleCall
flag = false;
% Get the stack
s = dbstack;
% Stack too short for a multiple call
if numel(s) <=2, return;end

% How many calls to the calling function are in the stack?
names = {s(:).name};
TF = strcmp(s(2).name,names);
count = sum(TF);
% More than 1
if count>1, flag = true;end