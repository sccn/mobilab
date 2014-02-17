classdef hedManager
    properties
        hedNode
        hedVersion
    end;
    
    methods
        function [answer description levelIsValid levelMatchedWithHEDNode autoCompleteAlternative] = isNodeSequenceValid(obj, nodeSequence, varargin)
            % [answer description levelIsValid levelMatchedWithHEDNode autoCompleteAlternative description] = isNodeSequenceValid(obj, nodeSequence)
            %
            % Input:
            %
            % nodeSequence              an HED node sequence, such as {'Time-Locked Event' 'Stimulus' 'Visual'}.
            %
            % Output:
            %
            % answer                    a boolean scalar, whether the sequence is a valid HED tag sequence
            %
            % description               a string that contain a text description of the HED node, if
            %                           availabe in HED xml (inside <description> tags).
            %
            % levelIsValid              a boolean vector, whether there was a macth with HED schema at each level (level is
            %                            the index in node sequence, for example 'Stimulus'is at level 2 in {'Time-Locked Event' 'Stimulus' 'Visual'}
            %
            % levelMatchedWithHEDNode   a boolean vector, whether the was a match with an exact HED.
            %                           A level could be valid either because of an exact match or due to extension (when allowed explicitly for a node or at the end of all levels) 
            %                         
            % autoCompleteAlternative   a cell array that contains all partial matches so they can
            %                           be used in an auto-complete feature. For example, if the input is 
            %                           {'Time-Locked Event' 'S'}, this variable would return
            %                           {'Stimulus'}. If the last node is empty in the sequence, all
            %                           HEd nodes in the lower HED level are returned. For example
            %                           is the input is {'Time-Locked Event' 'Stimulus' ''}, the
            %                           return value would contain all HED nodes under /Time-Locked
            %                           Event/Stimulus/ such as {'Visual' 'Auditory' 'TMS',...}.
            
            levelIsValid = false(length(nodeSequence),1);
            
            % whether the level matched exactly with a node currently in the HED specification.
            levelMatchedWithHEDNode = false(length(nodeSequence),1);
            
            % this gets the top HED node.
            currentNode = obj.hedNode;
            
            autoCompleteAlternative = {};
            description = '';
            
            % go recursively through xml nodes and see if the leaf with the correct name at that level exist.
            if currentNode.hasChildNodes
                for currentLevel = 1:length(nodeSequence)
                    
                    % by default extension is not allowed
                    extensionIsAllowed = false;
                    
                    % check if extension is allowed
                    if currentNode.hasAttributes
                        if strcmpi('true', char(currentNode.getAttribute('extensionAllowed')))
                            extensionIsAllowed = true;
                        end;
                    end;
                    
                    % the node that only has one child must only contant some text and no children.
                    % in HED it is allowed to extend any end node, for example '/Stimulus/Visual/Target' to
                    % /Stimulus/Visual/Target/Plane'
                    endOfHedHirerachyHasReached = false;
                    if currentNode.getChildNodes.getLength == 1
                        endOfHedHirerachyHasReached = true;                       
                    else
                        children = currentNode.getChildNodes;
                        hasAChildNode = false;                        
                        for i=1:children.getLength                            
                            if strcmpi('node', char(children.item(i-1).getNodeName))
                                hasAChildNode = true;
                            end;
                        end;
                        
                        endOfHedHirerachyHasReached = ~hasAChildNode;
                    end;
                    
                    if extensionIsAllowed || endOfHedHirerachyHasReached
                        levelIsValid(currentLevel) = true;
                    end;
                    
                    children = currentNode.getChildNodes;
                    
                    for i=1:children.getLength
                        % if the leaf found, go one level in and look for the next leaf
                        if children.item(i-1).hasChildNodes
                            % get the text content of the node, if existed. This is located in the first child.
                            
                            % before converting to lowercase, to be used in auto-complete
                            if strcmpi('node', char(children.item(i-1).getNodeName))
                                if strcmpi('name', char(children.item(i-1).getFirstChild.getNodeName))
                                    originalTextContent = strtrim(char(children.item(i-1).getFirstChild.getFirstChild.getData));
                                    textContent = lower(originalTextContent);
                                else
                                    textContent = '';
                                    originalTextContent = '';
                                end
                            else
                                textContent = '';
                                originalTextContent = '';
                            end;
                        else
                            textContent = '';
                            originalTextContent = '';
                        end;
                        
                        % for auto-complete, look for partial matches
                        % we only look for matches at the last level of HEd hierarchy.
                        % if the last level is empty
                        if  nargout > 3 && currentLevel == length(nodeSequence) && ~isempty(textContent) && (strncmpi(nodeSequence{currentLevel}, textContent, length(nodeSequence{currentLevel})))
                            autoCompleteAlternative{end+1} = originalTextContent;
                        end;
                        
                        % if there is a match, prevent matching to an empty node
                        exactMatch = strcmpi(textContent, nodeSequence{currentLevel}) && ~isempty(nodeSequence{currentLevel});                                                
                        
                        exactMatch = exactMatch | (textContent =='#' & any(ismember(nodeSequence{currentLevel}, ['1' '2' '3' '4' '5' '6' '7' '8' '9' '0'])));
                                                                        
                        if exactMatch
                            currentNode = children.item(i-1);
                            levelIsValid(currentLevel) = true;
                            levelMatchedWithHEDNode(currentLevel) = true;
                            break;
                        end;
                    end;
                    
                end;
            end;                       
            
            % HED tag is valid only if all the levels are valid.
            answer = all(levelIsValid);            
            
            if answer
                % get description text if available.
                potentialDescriptionNodeArray = currentNode.getElementsByTagName('description');
                
                
                for i=1:potentialDescriptionNodeArray.getLength
                    if potentialDescriptionNodeArray.item(i-1).getParentNode.isEqualNode(currentNode), 
                       description = strtrim(char(potentialDescriptionNodeArray.item(i-1).getFirstChild.getData));
                    end;
                end;
                
            end;
        end;
    end;
    
    methods
        
        function obj = hedManager(xmlFilePath, varargin)                        
            
            % if no path is specified, look for HEDSpecification.xml file under the same directory
            % where this class is located.            
            if nargin < 1
                fullPath = which('hedManager');
                xmlFilePath = pickfiles(fileparts(fullPath),{'Specification' '.xml'});
                xmlFilePath = deblank(xmlFilePath(end,:));
                % xmlFilePath  = [fullPath(1:end - length('hedManager.m')) 'HEDSpecification.xml'];
            end;
            
            xmlDocument = xmlread(xmlFilePath);
            obj.hedNode = xmlDocument.getDocumentElement;
            obj.hedVersion = str2double(char(obj.hedNode.getAttribute('version')));
        end;
        
        function [answer description levelIsValid levelMatchedWithHEDNode nodeSequence] = isValidHedTag(obj, hedTag, varargin)
           % [answer description levelIsValid levelMatchedWithHEDNode nodeSequence] = isValidHedTag(obj, hedTag)
           %
           % Returns true if a given HED tag, e.g. '/stimulus/visual' is valid in HED schema 
           %
           % Input
           % answer                    a boolean scalar that is true if the tag is valid and fale
           %                           otherwise.
           %
           % description               a string that contain a text description of the HED node, if
           %                           availabe in HED xml (inside <description> tags).
            
            nodeSequence = regexp(hedTag, '[/]', 'split');
            
            % remove / from start and end
            nodeSequence(cellfun(@isempty, nodeSequence)) = [];
            
            % remove extra spaces from start and end
            for i=1:length(nodeSequence)
                nodeSequence{i} = strtrim(nodeSequence{i});
            end;
            
            [answer description levelIsValid levelMatchedWithHEDNode] = isNodeSequenceValid(obj, nodeSequence);
            
            % if there was no match, assume 'Time-Locked Event' at the first level.
            if ~answer
                nodeSequence = cat(2, 'Time-Locked Event', nodeSequence);
                [answer description levelIsValid levelMatchedWithHEDNode] = isNodeSequenceValid(obj, nodeSequence);                
            end;
        end;
        
        function answer = isValidHedString(obj, hedString, varargin)
            % answer = isValidHedString(obj, hedString)
            %
            % Returns true if a given HED st, e.g. '/stimulus/visual, /response/button press, /stimulus/auditory' is 
            % valid in HED schema.
            
            hedTag =  regexp(hedString, '[;,]', 'split');
            
            answerForTag = false(length(hedTag),1);
            for i=1:length(hedTag)
                answerForTag(i) = isValidHedTag(obj, strtrim(hedTag{i}));
            end;
            
            % check if all the tags are valid
            answer = all(answerForTag);
        end;
        
        function autoCompleteAlternativeCell = getAutoCompleteAlternativesForTag(obj, hedTag, varargin)
            % [autoCompleteAlternativeCell] = getAutoCompleteAlternativesForTag(obj, hedTag)
            %
            % returns auto-complete alternatives for a given HED tag. For example '/stimulus/vi'
            % will return '/stimulus/visual', or '/stimulus/visual/' will return all tags with lower
            % level HEd nodes, e.g. {'/stimulus/visual/shape' '/stimulus/visual/fixation point' ...} 
            %
            % Please note that it make a difference whether to have '/' character at the end of the
            % input tag or not: if there is no '/', no lower level HED node will be returned.
                       
            nodeSequence = regexp(hedTag, '[/]', 'split');
            
            % remove / from start (but not end, so /Stimulus will be different from /Stimulus/
            % because in /Stimulus/ the matches at the lower level will be returned, but not in /Stimulus
            if isempty(nodeSequence{1})
                nodeSequence(1) = [];
            end;
            
            
            % remove extra spaces from start and end
            for i=1:length(nodeSequence)
                nodeSequence{i} = strtrim(nodeSequence{i});
            end;
            
            [answer1 description1 levelIsValid1 levelMatchedWithHEDNode1 autoCompleteAlternative1] = isNodeSequenceValid(obj, nodeSequence);
            
            nodeSequenceWithDefault = cat(2, 'Time-Locked Event', nodeSequence);
            [answer2 description2 levelIsValid2 levelMatchedWithHEDNode2 autoCompleteAlternative2] = isNodeSequenceValid(obj, nodeSequenceWithDefault);
            
            % pools alternatives across these these two (default to be 'Time-Locked Event' or not)
            autoCompleteAlternativeCell = {};
            
            if all(levelIsValid1(1:end-1)) % all except the last one should be valid
                for i=1:length(autoCompleteAlternative1)
                    autoCompleteAlternativeCell{end+1} =  strjoin('/', cat(2, nodeSequence{1:end-1}, autoCompleteAlternative1(i)));
                end;
            end;
            
            if all(levelIsValid2(1:end-1)) % all except the last one should be valid
                for i=1:length(autoCompleteAlternative2)
                    autoCompleteAlternativeCell{end+1} =  strjoin('/', cat(2, nodeSequenceWithDefault{1:end-1}, autoCompleteAlternative2(i)));
                end;
            end;
            
            
            autoCompleteAlternativeCell = unique(autoCompleteAlternativeCell);
        end;
        
        
        function [autoCompleteAlternativeCell] = getAutoCompleteAlternativesForString(obj, hedString, varargin)
            % [autoCompleteAlternativeCell] = getAutoCompleteAlternativesForString(obj, hedString, varargin)
            %
            % returns auto-complete alternatives for a given HED string. For example '/response, /stimulus/vi'
            % will return '/response, /stimulus/visual', or '/response, /stimulus/visual/' will return all tags with lower
            % level HEd nodes, e.g. {'/response, /stimulus/visual/shape' '/response, /stimulus/visual/fixation point' ...}
            % Only the last HED tag in the string will be considered for auto-completion..
            %
            % Please note that it make a difference whether to have '/' character at the end of the
            % input tag or not: if there is no '/', no lower level HED node will be returned.
            
            hedTag =  regexp(hedString, '[;,]', 'split');
            
            % pick the last tag for auto-complete and ignore all other beginning tags
            if length(hedTag) > 1
                ignoredTag = hedTag(1:end-1);
                activeTag = hedTag{end};
            else
                activeTag = hedTag{1};
                ignoredTag = {};
            end;
            
            autoCompleteAlternativeCellForTag = getAutoCompleteAlternativesForTag(obj, activeTag, varargin{:});
            
            if isempty(ignoredTag)
                ignoredSectionString = '';
            else
                ignoredSectionString = strjoin(', ', ignoredTag);
            end;
            
            if isempty(ignoredTag)
                autoCompleteAlternativeCell = autoCompleteAlternativeCellForTag;
            else
                autoCompleteAlternativeCell = cell(1, length(autoCompleteAlternativeCellForTag));
                for i=1:length(autoCompleteAlternativeCellForTag)
                    autoCompleteAlternativeCell{i} = [ignoredSectionString ', ' autoCompleteAlternativeCellForTag{i}];
                end;
            end;
        end;
        
        function [childNodesCell description] = getChildNodeNames(obj, parentNodeName, varargin)
            % [childNodesCell description] = getChildNodeNames(obj, parentNodeName)
            %
            % Returns a cell array that contains the names of all child nodes under a given parent
            % HED node. Thsi function can be used to create a tree of HED nodes by recursively
            % reading child nodes.
            %
            % Input:
            %
            % parentNodeName        a cell array of strings indicating the parent node (e.g.
            %                       {'Stimulus' 'Visual'} (this would correspond to /Stimulus/Visual
            %                       HED tag).
            %
            % Outputs:
            %
            % childNodesCell         a cell array of strings containing names of child nodes under
            %                        specified parent HEd node. For example 
            %                         {'Checkerboard'  'Fixation Point'} 
            %
            % description               a string that contain a text description of the parent node, if
            %                           availabe in HED xml (inside <description> tags).
            
            % if no input is given, return the top HED node names
            if nargin < 2
                nodeSequence = {''};
            else
                nodeSequence = parentNodeName;
                
                % there needs to be an empty string at the end to ask for matches as the next level
                if ~isempty(nodeSequence)                    
                    nodeSequence{end+1} = '';
                end;
            end;
                        
            [answer description levelIsValid levelMatchedWithHEDNode autoCompleteAlternative] = obj.isNodeSequenceValid(nodeSequence);
            
            % sinc the last node in the sequence is set to '', the parent should be valid only if
            % all except the last level matched. Also, the match should be exact (as extension does not apply here).
            isParentValid = all(levelMatchedWithHEDNode(1:end-1));
            
            if ~isParentValid
                nodeSequence = cat(2, 'Time-Locked Event', nodeSequence);
                [answer description levelIsValid levelMatchedWithHEDNode autoCompleteAlternative] = obj.isNodeSequenceValid(nodeSequence);                
                isParentValid = all(levelMatchedWithHEDNode(1:end-1));
            end;
            
            if isParentValid
                childNodesCell = autoCompleteAlternative;
                
                % get decription
                % remove the extra '' so we search for one level higher and get the description
                [answer description] = obj.isNodeSequenceValid(nodeSequence(1:(end-1)));                
            else
                childNodesCell = {};
                error('Provided parent node is not valid.');
            end;
        end;
        
        function hedTagArray = separateIntoTags(obj, hedString)
            hedString = strtrim(hedString);
            hedTagArray =  strtrim(regexp(hedString, '[;,]', 'split'));
            
            % remove extra slashes / from start and end.
            for i=1:length(hedTagArray)
                hedTagArray{i} = obj.removeExtraSlash(hedTagArray{i});
            end;
        end;
        
        function hedTag = removeExtraSlash(obj, hedTag)
            % hedTag = removeExtraSlash(obj, hedTag)
            % remove / from start and end of hedTag, as it interferes with string matching.
            if ~isempty(hedTag)
                if hedTag(1) == '/'
                    hedTag(1) = [];
                end;
                
                if hedTag(end) == '/'
                    hedTag(end) = [];
                end;
            end;            
        end;
            
            
        function [answer matchedTag] = stringMatchesTag(obj, hedString, hedTag)
            % [answer matchedTag] = stringMatchesTag(obj, hedString, hedTag)
            %
            % checks if a given HED tag can be considered to have a shared HED tag with a given HED
            % string. For example, whether (an event of) HED tag '/stimulus/visual/ellipse' falls under any of the
            % tags in the HED string '/response/auditory, /stimulus/visual'. The answer here is yes
            % since '/stimulus/visual/ellipse' is of type /stimulus/visual.
            %
            % Output:
            %
            % answer                a boolean scalar, whether the input HED tag matched the HED string.
            % matchedTag            a cell array of strings that contains the potential matched tags
            %                       from the HED string.
            %
            % Example:
            %
            %    [answer matchedTag] = hedManagerObject.stringMatchesTag('/stimulus/, response/visual/target', '/response/visual')
            %
            % returns answer = true, matchedTag = 'response/visual/target'
            
            
            % remove / from start and end of hedTag, as it interferes with string matching.
            hedTag = removeExtraSlash(obj, hedTag);
            
            % separate HED string into several tags
            hedTagToMatch = obj.separateIntoTags(hedString);
            
            matchFound = false(length(hedTagToMatch), 1);
            for i=1:length(hedTagToMatch)
                matchFound(i) = strncmpi(hedTag, hedTagToMatch{i}, length(hedTag));
            end;
            
            answer = any(matchFound);
            
            matchedTag = hedTagToMatch(matchFound);            
        end;
        
        function [answer matchedTag]= stringMatchesQueryString(obj, hedString, queryHedString)
            % [answer matchedTag]= stringMatchesQueryString(hedString, queryHedString)             
            %
            % checks if a given HED string can be considered to 'match' a given 'query' HED
            % string. To match here means that it has to contain all the tags in the query string.
            % For example, '/stimulus/visual/ellipse, /response/auditory, stimulus/tactile' matches 
            % the HED string '/response/auditory, /stimulus/visual' since it contains both these tags
            %
            % Output:
            %
            % answer                a boolean scalar, whether the input HED string matched the query HED string.
            % matchedTag            a cell array of strings that contains the potential matched tags
            %                       from the HED string.
            %
            % Example:
            %
            %    [answer matchedTag] = hedManagerObj.stringMatchesQueryString('/stimulus/visual/ellipse,  /response/auditory/10Hz, stimulus/tactile', '/stimulus/visual, response/auditory')
            %
            % returns answer = true, matchedTag = '/response/auditory/10Hz, /stimulus/visual/ellipse'
            
            queryHedTagsToMatch = obj.separateIntoTags(queryHedString);
            answer = true;
            matchedTag = {};
            for i=1:length(queryHedTagsToMatch)
                if answer
                    [answer matchedsingleTag]= obj.stringMatchesTag(hedString, queryHedTagsToMatch{i});
                    matchedTag = cat(2, matchedTag, matchedsingleTag);                    
                else
                    break;
                end;
            end;
            
            % if a fullo match cannot be made return an empty matchedTag
            if ~answer
                matchedTag = {};  
            end;
        end;
        
        function [answerArray matchedTagArray]= stringArrayMatchesQueryString(obj, hedStringArray, queryHedString)
            % [answerArray matchedTagArray] = stringArrayMatchesQueryString(obj, hedStringArray, queryHedString)
            % same as stringMatchesQueryString() but acts on an array of HED strings and one query
            % string.
            
%             answerArray = false(length(hedStringArray), 1);
%             if nargout > 1
%                 matchedTagArray = cell(length(hedStringArray), 1);
%             end;
%             for i=1:length(hedStringArray)
%                 if nargout > 1
%                     [answerArray(i), matchedTagArray{i}] =  obj.stringMatchesQueryString(hedStringArray{i}, queryHedString);
%                 else
%                     answerArray(i) =  obj.stringMatchesQueryString(hedStringArray{i}, queryHedString);
%                 end;
%             end;
            
            
            % first reduce to a possible subset of solutions with the first loop by calling strfind on
            % cells.
            answerArray = true(length(hedStringArray), 1);
            queryHedTagsToMatch = obj.separateIntoTags(queryHedString);
            for j = 1:length(queryHedTagsToMatch)
                answerArray = answerArray & ~cellfun(@isempty,strfind(hedStringArray,queryHedTagsToMatch{j}));
            end
            
            % next, perform StringMatchesQueryString only on the 'true'
            % indices of the string array.
            if nargout > 1
                matchedTagArray = cell(length(hedStringArray), 1);
            end;
            
            candidates = find(answerArray);
            for j = 1:length(candidates)
                i = candidates(j);
                if nargout > 1
                    [answerArray(i), matchedTagArray{i}] =  obj.stringMatchesQueryString(hedStringArray{i}, queryHedString);
                else
                    answerArray(i) =  obj.stringMatchesQueryString(hedStringArray{i}, queryHedString);
                end;
            end
            
        end;
    end;
end