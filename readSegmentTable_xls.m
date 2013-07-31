function [treeNodes,kinematicTree,nodes,connectivity,ls_marker2nodesMapping,segment] = readSegmentTable_xls(file)
[~,table] = xlsread(file);
markers = table(2:end,4);
nodes = table(2:end,1);
for it=1:length(markers)
    loc = find(markers{it}==' ');
    if isempty(loc)
        markers{it} = '0';
    else
        markers{it} = markers{it}(loc+2:end-1);
    end
end
markers = str2double(markers);
isMarker = markers ~= 0;
markers = markers(isMarker);
nodes = nodes(~isMarker);
for it=2:length(nodes)
    loc = find(nodes{it}=='_');
    nodes{it} = nodes{it}(loc(1)+1:end);
end
nodes = unique(nodes,'stable');
nNodes = length(nodes);
states = table(2:end,2);
states = states(isMarker);

iG = zeros(max(markers),nNodes);
for it=1:length(nodes)
    ind = ~cellfun(@isempty,strfind(states,nodes{it}));
    iG(markers(ind),it) = 1;
end
ind = [find(~cellfun(@isempty,strfind(states,'hip_r')));find(~cellfun(@isempty,strfind(states,'hip_l')))];
iG(markers(ind),1) = 1;
iG = bsxfun(@rdivide,iG,sum(iG));
ls_marker2nodesMapping = kron(iG,speye(3));

treeNodes = table(2:end,1);
treeNodes(isMarker) = [];
parentNodes = table(2:end,2);
parentNodes(isMarker) = [];

connectivity = zeros(length(treeNodes));
for it=1:length(treeNodes)
    ind = find(ismember(treeNodes,parentNodes{it}));
    connectivity(it,ind) = 1;
    connectivity(ind,it) = 1;
end
connectivity = sparse(connectivity);
kinematicTree = connectivity;

tmpTreeNodes = treeNodes;
for it=2:length(treeNodes), tmpTreeNodes{it}(1:2) = [];end
for it=1:length(treeNodes)
    if parentNodes{it}(2) == '_', parentNodes{it}(1:2) = [];end
    parentNodes{it}(find(parentNodes{it}==' '):end) = [];
end
connectivity = zeros(nNodes);
for it=1:length(treeNodes)
    loc1 = find(ismember(nodes,tmpTreeNodes{it}));
    loc2 = find(ismember(nodes,parentNodes{it}));
    if loc1~=loc2
        connectivity(loc1,loc2) = 1;
        connectivity(loc2,loc1) = 1;
    end
end
connectivity = sparse(connectivity);
table(1,:) = [];
table(:,1) = [];
dim = size(table);
segment = zeros(dim);
for it=1:dim(1)
    for jt=1:dim(2)
        if strfind(table{it,jt},'none')
            segment(it,jt) = 0;
        else 
            switch jt
                case 1
                    loc1 = find(table{it,jt}=='(')+1;
                    loc2 = find(table{it,jt}==')')-1;
                    segment(it,jt) = str2double(table{it,jt}(loc1:loc2)); % find(ismember(tmpNodeNames,table{it,jt}));
                case 2
                    switch table{it,jt}
                        case 'translation', segment(it,jt) = 2;
                        case 'hinge',       segment(it,jt) = 3;
                        case 'rotation',    segment(it,jt) = 4;
                    end
                case 3
                    loc1 = find(table{it,jt}=='(')+1;
                    loc2 = find(table{it,jt}==')')-1;
                    segment(it,jt) = str2double(table{it,jt}(loc1:loc2));
                case 4
                    switch table{it,jt}(1:find(table{it,jt}==' ')-1)
                        case 'x',  segment(it,jt) = 1;
                        case 'y',  segment(it,jt) = 2;
                        case 'z',  segment(it,jt) = 3;
                        otherwise, segment(it,jt) = 0;
                    end
                case 5, if strcmp(table{it,jt},'varying'), segment(it,jt) = 1;else segment(it,jt) = 2;end
            end
        end
    end
end