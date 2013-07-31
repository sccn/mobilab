function index = getIndex4aBranch(bgObj,seed)
if nargin < 2,
    bgObjChilds = bgObj.getdescendants;
    bgObjChilds(bgObjChilds == bgObj) = [];
else
    bgObjChilds = bgObj.Nodes(seed+1).getdescendants;
    bgObjChilds(bgObjChilds == bgObj.Nodes(seed+1)) = [];
end


if isempty(bgObjChilds)
    index = [];
    return;
else
    I = [];
    index = zeros(length(bgObjChilds),1);
    for it=1:length(bgObjChilds)
        loc1 = strfind(bgObjChilds(it).ID,'(');
        loc2 = strfind(bgObjChilds(it).ID,')');
        index(it) = str2double(bgObjChilds(it).ID(loc1+1:loc2-1));
        index2 = getIndex4aBranch(bgObjChilds(it));
        I = [I; index2];
    end
    index = unique([I(:); index(:)]);
end