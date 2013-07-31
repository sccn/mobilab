classdef graphCoreObject
    properties 
        adjacencyMatrix
    end
    methods
        function obj = graphCoreObject(adjacencyMatrix)
            if nargin < 1, error('Not enough input aruments.');end
            obj.adjacencyMatrix = sparse(adjacencyMatrix);
        end
        %%
        function indices = getDescendants(obj,index)
            if index > size(obj.adjacencyMatrix,1), index = [];end
            indices = find(obj.adjacencyMatrix(index,:));
            if ~isempty(indices) && index ~= 1, indices(1) = [];end
        end
        %%
        function indices = getAncestors(obj,index)
            if index > size(obj.adjacencyMatrix,1), index = [];end
            indices = find(obj.adjacencyMatrix(index,:),1);
            if indices > index, indices = [];return;end
            if ~isempty(indices), indices2 = getAncestors(obj,indices);end
            indices = unique([indices indices2]);
        end
        %%
        function nodeList = getIndex4aBranch(obj,index)
            nodeList = [];
            indices = getDescendants(obj,index);
            indices(indices==index) = [];
            N = length(indices);
            if N, for it=1:N, nodeList = [nodeList getIndex4aBranch(obj,indices(it))];end; end %#ok
            nodeList = unique([index indices nodeList]);
        end
    end
end