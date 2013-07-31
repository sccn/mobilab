%% Defines objects from the class  jtreeGraph.
% These objects display a tree structure encoded in an adjacency matrix using
% a Java jtree component embeded on a figure.
% 
% obj = jtreeGraph(adjacencyMatrix,labels);
%
% Inputs:
%   adjacencyMatrix: Adjacency matrix (could be sparse) defined as follows:
%                    element i,j = 1 if node_i is connected with node_j, 
%                    otherwise the entry is 0. With i,j = 1: number of nodes.
%   labels:          Cell array of labels, one per node.
%   rootNodeName:    String with the name of the root node, default: 'root'.
%
% Example:
% Given the following array of hedTags (Hierarchical Event Descriptor Tags):
%    A / A1 / A2 / A3
%    A / A4 / A5 / A6
%    B / B1 / B2 / B3
% 
% An adjacency matrix can be extracted to produce the following tree:
%
%  root
%    |
%    |-- A
%    |   |
%    |   |-- A1
%    |   |-- A2
%    |   |-- A3
%    |   |-- A4
%    |   |-- A5
%    |   |-- A6
%    |
%    |-- B
%        |
%        |-- B1
%        |-- B2
%        |-- B3
% 
% Where the array of labels is {A, A1, ..., A6, B, B1, ..., B3}.
% 
% 
% Author: Alejandro Ojeda, SCCN, INC, UCSD, 19-Mar-2013

%%
classdef jtreeGraph < graphCoreObject
    properties 
        hFigure
        nodeLabels
    end
    methods
        function obj = jtreeGraph(adjacencyMatrix,labels, rootNodeName)
            if nargin < 2, error('Not enough input arguments.');end 
            if nargin < 3, rootNodeName = 'root';end
            obj@graphCoreObject(adjacencyMatrix);
            obj.nodeLabels = labels;
            N = size(obj.adjacencyMatrix,1);
            node = cell(N-1,1);
            nodeIDs = cell(N,1);
            nodeIDs{1} = rootNodeName;
            for it=2:N, nodeIDs{it} = labels{it-1};end
            root = javax.swing.tree.DefaultMutableTreeNode( nodeIDs{1} );
            javaObjectEDT(root);
            for it=1:N-1, node{it} = javax.swing.tree.DefaultMutableTreeNode(nodeIDs{it+1});end
            ind = find(obj.adjacencyMatrix(1,:));
            for jt=1:length(ind), root.add(node{ind(jt)-1});end
            for it=1:N-1
                ind = getDescendants(obj,it+1);
                for jt=1:length(ind), try node{it}.add(node{ind(jt)-1});end;end %#ok
            end
            
            jTree = javax.swing.JTree(root);
            jTree.setShowsRootHandles(true)
            javaObjectEDT(jTree);
            jScrollPane = com.mathworks.mwswing.MJScrollPane(jTree);
            javaObjectEDT(jScrollPane)
            obj.hFigure = figure('Name','Hed tag tree','Color',[0.93 0.96 1]);
            position = get(obj.hFigure,'Position');
            [~,hc ] = javacomponent(jScrollPane,[2 1.5 ,position(3:4)],obj.hFigure);
            set(hc,'Units','Normalized','Tag','hedTagTree','Position',[0 0 1 1]);
        end
    end
end