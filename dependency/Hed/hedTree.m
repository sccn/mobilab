classdef hedTree
    properties
        uniqueTag               % a cell array with unique HED tags in the input HED string cell array (derived from all input HED strings)
        uniqueTagCount          % a numrival array containing the number of occurances of each unique tag across all input HED string cell array.
        originalHedStringId     % a cell array where each cell contains indices in the input HED string cell array where each uniqueTag
        adjacencyMatrix         %  adjacencyMatrix: Adjacency matrix (could be sparse) defined as follows:
        %  element i,j = 1 if node_i is connected with node_j,
        %  therwise the entry is 0. With i,j = 1: number of nodes.
    end;
    methods (Static = true)
        
        function n = progress(rate,title)
            %PROGRESS   Text progress bar
            %   Similar to waitbar but without the figure display.
            %
            %   Start:
            %   PROGRESS('init'); initializes the progress bar with title 'Please wait...'
            %   PROGRESS('init',TITLE); initializes the progress bar with title TITLE
            %   PROGRESS(RATE); sets the length of the bar to RATE (between 0 and 1)
            %   PROGRESS(RATE,TITLE); sets the RATE and the TITLE
            %   PROGRESS('close'); (optionnal) closes the bar
            %
            %   Faster version for high number of loops:
            %   The function returns a integer indicating the length of the bar.
            %   This can be use to speed up the computation by avoiding unnecessary
            %   refresh of the display
            %   N = PROGRESS('init'); or N = PROGRESS('init',TITLE);
            %   N = PROGRESS(RATE,N); changes the length of the bar only if different
            %   from the previous one
            %   N = PROGRESS(RATE,TITLE); changes the RATE and the TITLE
            %   PROGRESS('close'); (optionnal) closes the bar
            %
            %   The previous state could be kept in a global variable, but it is a bit
            %   slower and doesn't allows nested waitbars (see examples)
            %
            %   Known bug: Calling progress('close') shortly afer another call of the
            %   function may cause strange errors. I guess it is because of the
            %   backspace char. You can add a pause(0.01) before to avoid this.
            %
            %   Examples:
            %       progress('init');
            %       for i=1:100
            %           progress(i/100, sprintf('loop %d/100',i));
            %
            %           % computing something ...
            %           pause(.1)
            %       end
            %       progress('close'); % optionnal
            %
            %
            %       % Inside a script you may use:
            %       n = progress('init','wait for ... whatever');
            %       for i=1:100
            %           n = progress(i/100,n);
            %
            %           % computing something ...
            %           pause(.1)
            %       end
            %       progress('close');
            %
            %
            %       % Add a time estimation:
            %       progress('init','Processing...');
            % 		tic       % only if not already called
            % 		t0 = toc; % or toc if tic has already been called
            % 		tm = t0;
            % 		L  = 100;
            % 		for i=1:L
            % 			tt = ceil((toc-t0)*(L-i)/i);
            % 			progress(i/L,sprintf('Processing... (estimated time: %ds)',tt));
            %
            % 			% computing something ...
            % 			pause(.1)
            % 		end
            % 		progress('close');
            %
            %
            %       % Add a faster time estimation:
            % 		n  = progress('init','Processing...');
            % 		tic       % only if not already called
            % 		t0 = toc; % or toc if tic has already been called
            % 		tm = t0;
            % 		L  = 100;
            % 		for i=1:L
            % 			if tm+1 < toc % refresh time every 1s only
            % 				tm = toc;
            % 				tt = ceil((toc-t0)*(L-i)/i);
            % 				n  = progress(i/L,sprintf('Processing... (estimated time: %ds)',tt));
            % 			else
            % 				n  = progress(i/L,n);
            % 			end
            %
            % 			% computing something ...
            % 			pause(.1)
            % 		end
            % 		progress('close');
            %
            %       % Nested loops:
            %       % One loop...
            % 		n1 = progress('init','Main loop');
            % 		for i=0:7
            % 			n1 = progress(i/7,n1);
            %
            % 			% ... and another, inside the first one.
            % 			n2 = progress('init','Inside loop');
            % 			for j=0:50
            % 				n2 = progress(j/50,n2);
            %
            % 				% computing something ...
            % 				pause(.01)
            % 			end
            % 			progress('close');
            % 		end
            % 		pause(.01)
            % 		progress('close');
            
            %   31-08-2007
            %   By Joseph martinot-Lagarde
            %   joseph.martinot-lagarde@m4x.org
            
            %   Adapted from:
            %   MMA 31-8-2005, martinho@fis.ua.pt
            %   Department of Physics
            %   University of Aveiro, Portugal
            
            %% The simplest way to bypass it...
            % n = 0; return
            
            %% Width of the bar
            %If changes are made here, change also the default title
            lmax=70;  % TM: changed from lmax=50;
            
            %% Erasing the bar if necessary
            % not needed, but one could find it prettier
            if isequal(rate,'close')
                % there were 3 '\n' added plus the title and the bar itself
                fprintf(rep('\b',2*lmax+3))
                return
            end
            
            %% The init
            if isequal(rate,'init') % If in init stage
                cont = 0;           % we don't continue a previous bar
                back = '\n';        % add a blank line at the begining
                rate = 0;           % start from 0
            else
                cont = 1;           % we continue a previous bar
            end
            
            %% No need to update the view if not necessary
            % optional, but saves a LOT of time
            
            % length of the displayer bar in number of char
            % double([0,1]) to int([0,lmax-1])
            n = min(max( ceil(rate*(lmax-2)) ,0),lmax-2);
            
            % If the 2nd arg is numeric, assumed to be the previous bar length
            if nargin >=2 && isnumeric(title)
                if n == title % If no change, do nothing
                    return
                else          % otherwise continue
                    n_ = title;
                    clear title
                end
            else % draw the whole bar
                n_ = -1;
            end
            
            %% The title
            % If a new title is given, display it
            if exist('title','var')
                Ltitle = length(title);
                if Ltitle > lmax % If too long, cut it
                    title = [title(1:lmax) '\n']
                else             % otherwise center it
                    title = [rep(' ',floor((lmax-Ltitle)/2)) title rep(' ',ceil((lmax-Ltitle)/2)) '\n'];
                end
                if cont % If not in init stage, erase the '\n' and the previous title
                    back = rep('\b',lmax+1);
                end
            else
                if cont % If not in init stage, give a void title
                    title = '';
                    back  = ''; % has to be set
                else    % else set a default title
                    title = '                  Please wait...                  \n';
                end
            end
            
            %% The bar
            % '\f' should draw a small square (at least in Windows XP, Matlab 7.3.0 R2006b)
            % If not, change to any desired single character, like '*' or '#'
            if ~cont || n_ == -1 % at the begining disp the whole bar
                str = ['[' rep('*',n) rep(' ',lmax-n-2) ']\n'];
                if cont % If not in init stage, erase the previous bar
                    back = [back, rep('\b',lmax+1)];
                end
            else % draw only the part that changed
                str  = [rep('*',n-n_) rep(' ',lmax-n-2) ']\n'];
                back = [back, rep('\b',lmax-n_)];
            end
            
            %% The print
            % Actually make the change
            fprintf([back title str]);
            return
            
            %% Function to repeat a char n times
            function cout = rep(cin,n)
                if n==0
                    cout = [];
                    return
                elseif length(cin)==1
                    cout = cin(ones(1,n));
                    return
                else
                    d    = [1; 2];
                    d    = d(:,ones(1,n));
                    cout = cin(reshape(d,1,2*n));
                    return
                end
            end
        end
        
        function [uniqueTag, uniqueTagCount, originalHedStringId] = hed_tag_count(hedStringArray)
            % separate HED string in the array into indivudal tags and removing the ones with " or numbers.
            
            hedTree.progress('init');           
            
            trimmed = strtrim(hedStringArray);
            hasDoublequote = ~cellfun(@isempty, strfind(hedStringArray, '"'));
            separatd =  strtrim(regexp(trimmed, '[;,]', 'split'));
            
            allTags = cell(length(hedStringArray) * 3,1);
            allTagId = zeros(length(hedStringArray) * 3,1);
            counter = 1;
            for i=1:length(separatd)
                
                if mod(i, 200) == 0
                    hedTree.progress(i/length(separatd), 'Step 1/4');
                end;
                
                if ~hasDoublequote(i)
                    inside = separatd{i};
                    allTags(counter:(counter+length(inside) -1)) = inside;
                    allTagId(counter:(counter + length(inside) - 1)) = i;
                    counter = counter  + length(inside);
                end;
            end;
            
            allTags(counter:end) = [];
            allTagId(counter:end) = [];
            
            % remove numbers, for some reason some tags are just numbers
            isaNumber =  ~isnan(str2double(allTags));
            allTags(isaNumber) = [];
            allTagId(isaNumber) = [];
            
            %% unroll the tags so the hierarchy is turned into multiple nodes. For example /Stimulus/Visual/Red becomes three tags: /Stimulus/, Stimulus/Visual and /Stimulus/Visual/Red. This lets us count the higher hierarchy levels.
            
            combinedTag = cell(length(allTags) * 5, 1);
            combinedId = zeros(length(allTags) * 5, 1);
            counter = 1;
            
            hedTree.progress(10, '10 percent');
            
            for i = 1:length(allTags)
                
                if mod(i, 200) == 0
                    hedTree.progress(i/length(allTags), 'Step 2/4');
                end;
                
                nodeSequence = regexp(allTags{i}, '[/]', 'split');
                
                % remove / from start and end
                nodeSequence(cellfun(@isempty, nodeSequence)) = [];
                
                newTags = {};
                for j=1:length(nodeSequence)
                    if j==1
                        newTags{j} = nodeSequence{1};
                    else
                        newTags{j} = strjoin('/', nodeSequence(1:j));
                    end;
                end;
                
                combinedTag(counter:(counter + length(newTags) - 1)) = newTags;
                
                newTagsId = ones(length(newTags),1) * i;
                %combinedId = cat(1, combinedId, newTagsId);
                combinedId(counter:(counter + length(newTags) - 1)) = newTagsId;
                
                counter = counter + length(newTags);
            end;
            
            if counter < length(combinedTag)
                combinedTag(counter:end) = [];
                combinedId(counter:end) = [];
            end
            
            %% find unique tags and count them. Use sorting to speed this up.
            
            [sortedCombinedTag ord]= sort(combinedTag);
            sortedCombinedId = combinedId(ord);
            
            [uniqueTag firstIndexUnique]= unique(sortedCombinedTag, 'first');
            [uniqueTag lastIndexUnique]= unique(sortedCombinedTag, 'last');
            
            uniqueTagCount = lastIndexUnique-firstIndexUnique+1;
            
            uniqueTagId = cell(length(lastIndexUnique),1);
            originalHedStringId = cell(length(lastIndexUnique),1);
            for i=1:length(lastIndexUnique)
                
                if mod(i, 200) == 0
                    hedTree.progress(i/length(lastIndexUnique), 'Step 3/4');
                end;
                
                uniqueTagId{i} = unique(sortedCombinedId(firstIndexUnique(i):lastIndexUnique(i))); % these are IDs of allTags.
                originalHedStringId{i} = allTagId(uniqueTagId{i}); % these are IDs of input HED string.
            end;
            
            pause(.1);
            hedTree.progress('close'); % duo to some bug need a pause() before
            fprintf('\n');
        end;
    end;
    
    methods
        function obj = hedTree(hedStringArray)                        
            [obj.uniqueTag, obj.uniqueTagCount, obj.originalHedStringId] = hedTree.hed_tag_count(hedStringArray);
            obj = makeAdjacencyMatrix(obj);
        end;
        
        function obj = makeAdjacencyMatrix(obj)
            isParentyMatrix = false(length(obj.uniqueTag)+1, length(obj.uniqueTag)+1);
            
            hedTree.progress('init');     
            for i=1:length(obj.uniqueTag)
                
                if mod(i, 200) == 0
                    hedTree.progress(i/length(obj.uniqueTag), 'Step 4/4');
                end;
                
                isParentyMatrix(i+1,2:end) = strncmpi(obj.uniqueTag{i}, obj.uniqueTag, length(obj.uniqueTag{i}));
            end;
            
            
            % find top-level nodes to be connected to the 'root' node, they are recognized as having no parents
            isParentyMatrix = logical(isParentyMatrix - diag(diag(isParentyMatrix)));
            isTopLevel = ~any(isParentyMatrix);
            
            obj.adjacencyMatrix = isParentyMatrix;
            
            obj.adjacencyMatrix(1,isTopLevel) = true;
            obj.adjacencyMatrix(1,1) = false; % the root node is not considered a child of itself.
            obj.adjacencyMatrix = obj.adjacencyMatrix | obj.adjacencyMatrix';
            
            pause(.1);
            hedTree.progress('close'); % duo to some bug need a pause() before
            fprintf('\n');
        end;
        
        function hFigure = plot(obj, varargin)
            uniqueTagLabel = cell(length(obj.uniqueTag),1);
            
            for i=1:length(obj.uniqueTag)
                locationOfSlash = find(obj.uniqueTag{i} == '/', 1, 'last');
                if isempty(locationOfSlash)
                    uniqueTagLabel{i} = obj.uniqueTag{i};
                else
                    uniqueTagLabel{i} = obj.uniqueTag{i}(locationOfSlash+1:end);
                end;
                
                uniqueTagLabel{i}(1) = upper(uniqueTagLabel{i}(1));
                
                uniqueTagLabel{i} = [uniqueTagLabel{i} ' (' num2str(obj.uniqueTagCount(i)) ')'];
            end;
            
            jtreeGraphObj = jtreeGraph(obj.adjacencyMatrix, uniqueTagLabel, 'Hed Tag');
            hFigure = jtreeGraphObj.hFigure;
        end;
    end
end