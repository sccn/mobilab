classdef swingEpoch < mocapEpoch
    methods
        function obj = swingEpoch(varargin)
            if length(varargin)==1, varargin = varargin{1};end
            if isa(varargin,'mocapEpoch')
                mocapEpochObj = varargin;
                varargin = {'data',mocapEpochObj.data,'time',mocapEpochObj.time,'channelLabel',mocapEpochObj.channelLabel,...
                'condition',mocapEpochObj.condition,'eventInterval',mocapEpochObj.eventInterval,'xy',mocapEpochObj.xy,...
                'derivativeLabel',mocapEpochObj.derivativeLabel,'subjectID',mocapEpochObj.subjectID};
            end
            obj@mocapEpoch(varargin);
            stack = dbstack;
            if any(ismember({'epochObject.copyobj' 'epochObject.loadobj'},{stack.name})), return;end
            [~,loc] = min(abs(obj.time));
            if loc == 1, loc = fix(length(obj.time)/2);end
            obj.xy(1:loc-1,1) = -(obj.xy(1:loc-1,1)-obj.xy(loc-1,1));
            obj.xy(loc:end,1) = obj.xy(loc:end,1) - obj.xy(loc,1);
        end
        %%
        function [objArrayR, objArrayL] = splitRLswings(obj)
            N = length(obj);
            objArrayR = copyobj(obj);
            objArrayL = copyobj(obj);
            for it=1:N
                [~,loc0] = min(abs(obj(it).time));
                
                objArrayR(it).condition = ['R-' obj(it).condition];
                objArrayR(it).time = obj(it).time(1:loc0);
                objArrayR(it).time = objArrayR(it).time-mean(objArrayR(it).time);
                objArrayR(it).xy(loc0+1:end,:) = [];
                objArrayR(it).xy(:,1) = -objArrayR(it).xy(:,1);
                objArrayR(it).xy = bsxfun(@minus,objArrayR(it).xy,mean(objArrayR(it).xy));
                objArrayR(it).data(loc0+1:end,:,:) = [];
                
                objArrayL(it).condition = ['L-' obj(it).condition];
                objArrayL(it).time = obj(it).time(loc0+1:end);
                objArrayL(it).time = objArrayL(it).time-min(objArrayL(it).time);
                objArrayL(it).time = objArrayL(it).time-mean(objArrayL(it).time);
                objArrayL(it).xy(1:loc0,:) = [];
                objArrayL(it).xy = bsxfun(@minus,objArrayL(it).xy,mean(objArrayL(it).xy));
                objArrayL(it).data(1:loc0,:,:) = [];
            end
        end
        %
        function scObj = restoreAxis(obj)
            scObj = copyobj(obj);
            Nobj = length(obj);
            for it=1:Nobj
                [~,zeroLoc] = min(abs(scObj(it).xy(:,1)));
                scObj(it).xy(1:zeroLoc,1) = -scObj(it).xy(1:zeroLoc,1);
                scObj(it).xy(:,1) = scObj(it).xy(:,1) - mean(scObj(it).xy(:,1));
            end
        end
        %%
        function hFigure = plot(obj,sortOrder,channel)
            Nobj = length(obj);
            if Nobj > 1, plotArray(obj);return;end  
            if nargin < 2, sortOrder = 1:size(obj.data,3);end
            if nargin < 3, channel = 1;end
            time = obj.time;
            [~,zeroLoc] = min(abs(obj.time));
            t1 = linspace(0,50,zeroLoc);
            t2 = linspace(50,100,length(obj.time)-zeroLoc+1);
            obj.time = [t1 t2(2:end)];
            hFigure = plot@mocapEpoch(obj,sortOrder,channel);
            hAxes = findobj(hFigure,'Tag','kprofiles');
            for it=1:length(hAxes), xlabel(hAxes(it),'Cycle (%)');end
            obj.time = time;
        end
        %%
        function [group,xy,vel,acc,jerk,id_rm_conditions] = saveInArray(obj,file)
            N = length(obj);
            if N > 1 && length(obj(1).time) ~= length(obj(2).time)
                 nObj = normalize(obj);
            else nObj = obj;
            end
            group = repmat(struct('subject','','condition',[],'song',[]),N,1);
            xy    = zeros([size(nObj(1).xy) N]);
            velCell   = cell(N,1);
            accCell   = cell(N,1);
            jerkCell  = cell(N,1);
            
            Acc   = [];
            for it=1:N
                loc = strfind(nObj(it).condition,'-');
                group(it).subject = nObj(it).subjectID;
                group(it).condition = nObj(it).condition(1:loc(1)-1);
                group(it).song = nObj(it).condition(loc(1)+1:loc(2)-1);
                xy(:,:,it) = nObj(it).xy;
                velCell{it}  = squeeze(nObj(it).data(:,1,:));
                accCell{it}  = squeeze(nObj(it).data(:,2,:));
                jerkCell{it} = squeeze(nObj(it).data(:,3,:));
                Acc = [Acc accCell{it}];
            end
            
            alpha = 5;
            th = [prctile(min(Acc),alpha) prctile(max(Acc),100-alpha)];
            ind = find(any(Acc < th(1) | Acc > th(2)));
            fprintf('Removing %f %% of the trials\n', 100*length(ind)/size(Acc,2))
            rmThis = false(N,1);
            for it=1:N
                ind = find(any(accCell{it} < th(1) | accCell{it} > th(2)));
                velCell{it}(:,ind)  = [];
                accCell{it}(:,ind)  = [];
                jerkCell{it}(:,ind) = [];
                if isempty(velCell{it}), rmThis(it) = true;end
            end
            
            id = cell(length(group),1);
            for it=1:length(group), id{it} = [group(it).subject '_' group(it).song '_' group(it).condition];end
            id_rm_conditions = id(rmThis);
            xy(:,:,rmThis) = [];
            velCell(rmThis)  = [];
            accCell(rmThis)  = [];
            jerkCell(rmThis) = [];
            group(rmThis)  = [];
            
            N = length(velCell);
            Nt = length(nObj(1).time);
            vel  = zeros(Nt,N);
            acc  = zeros(Nt,N);
            jerk = zeros(Nt,N);
            for it=1:N
                vel(:,it) = mean(velCell{it},2);
                acc(:,it) = mean(accCell{it},2);
                jerk(:,it) = mean(jerkCell{it},2);
            end
            if nargin < 2, file = '';end
            if isempty(file), return;end
            %save(file,'-mat','group','xy','velCell','accCell','jerkCell');
            save(file,'-mat','group','xy','vel','acc','jerk');
        end
    end
    methods(Hidden = true)
        %%
        function hFigure = plotArray(obj)
            if length(obj) < 2, return;end
            hFigure = plotArray@mocapEpoch(obj);
        end
    end
end
