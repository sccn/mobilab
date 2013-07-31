classdef bodyStream < mocap
    properties(SetObservable)
        connectivity
        nodes
    end
    properties(Hidden = true)
        kinematicTree
        treeNodes
    end
    methods
        function obj = bodyStream(header)
            if nargin < 1, error('Not enough input arguments.');end
            obj@mocap(header);
        end
        %%
        function connectivity = get.connectivity(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.set.connectivity'))
                connectivity = obj.connectivity;
                return;
            end
            if isempty(obj.connectivity), obj.connectivity = retrieveProperty(obj,'connectivity');end
            connectivity = obj.connectivity;
        end
        function set.connectivity(obj,connectivity)
            if ~ismatrix(connectivity), error('''connectivity'' must be a matrix');end
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.get.connectivity'))
                obj.connectivity = connectivity;
                return;
            end
            obj.connectivity = connectivity;
            saveProperty(obj,'connectivity',connectivity)
        end
        %%
        function nodes = get.nodes(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.set.nodes'))
                nodes = obj.nodes;
                return;
            end
            if isempty(obj.nodes), obj.nodes = retrieveProperty(obj,'nodes');end
            nodes = obj.nodes;
        end
        function set.nodes(obj,nodes)
            if ~cellstr(nodes), error('''nodes'' must be a cell array of string');end
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.get.nodes'))
                obj.nodes = nodes;
                return;
            end
            obj.nodes = nodes;
            saveProperty(obj,'nodes',nodes);
        end
        %%
        function kinematicTree = get.kinematicTree(obj)
            tion set.nodes(obj,nodes)
            % if ~cellstr(kinematicTree), error('''nodes'' must be a cell array of string');end
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.set.kinematicTree'))
                kinematicTree = obj.kinematicTree;
                return;
            end
            if isempty(obj.kinematicTree), obj.kinematicTree = retrieveProperty(obj,'kinematicTree');end
            kinematicTree = obj.kinematicTree;
        end
        function set.kinematicTree(obj,kinematicTree)
            % if ~ismatrix(kinematicTree), error('''nodes'' must be a cell array of string');end
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.get.kinematicTree'))
                obj.kinematicTree = kinematicTree;
                return;
            end
            obj.kinematicTree = kinematicTree;
            saveProperty(obj,'kinematicTree',kinematicTree);
        end
        %%
        function treeNodes = get.treeNodes(obj)
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.set.treeNodes'))
                treeNodes = obj.treeNodes;
                return;
            end
            if isempty(obj.treeNodes), obj.treeNodes = retrieveProperty(obj,'treeNodes');end
            treeNodes = obj.treeNodes;
        end
        function set.treeNodes(obj,treeNodes)
            if ~cellstr(treeNodes), error('''treeNodes'' must be a cell array of string');end
            stack = dbstack;
            if any(strcmp({stack.name},'bodyStream.get.treeNodes'))
                obj.treeNodes = treeNodes;
                return;
            end
            obj.treeNodes = treeNodes;
            saveProperty(obj,'treeNodes',treeNodes);
        end
        %%
        function BGobj = viewModel(obj,showFlag)
            if isempty(which('biograph'))
                warning('Bioinfo toolbox is missing.');
                return
            end
            if nargin < 2, showFlag = true;end
            transfType = cell(length(obj.treeNodes),1);
            transfType{1} = 'none';
            label = cell(length(obj.treeNodes),1);
            label{1} = [obj.treeNodes{1} ' (root)'];
            for it=2:length(obj.treeNodes)
                if strcmp(obj.treeNodes{it}(1:2),'t_')
                    transf = 'tranlation: ';
                else
                    transf = 'rotation: ';
                end
                label{it} = [transf,obj.treeNodes{it}(3:end)];
                transfType{it} = transf;
            end
            BGobj = biograph(triu(obj.kinematicTree), label);
            if showFlag
                g = biograph.bggui(BGobj);
                set(g.hgFigure,'Name','Body model');
            end
        end
    end
    
end

