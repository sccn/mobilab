classdef mocapEpoch < epochObject
    properties
        derivativeLabel
        xy
    end
    methods
        function obj = mocapEpoch(varargin)
            if length(varargin) < 7, error('Not enough input arguments.');end

            obj@epochObject(varargin{:});
            obj.xy = varargin{7};
            
            if length(varargin) < 8
                n = size(obj.data,2);
                derivativeLabel = cell(n,1);                             %#ok
                for it=1:n, derivativeLabel{it} = ['Dt' num2str(it)];end %#ok
            else derivativeLabel = varargin{8};                          %#ok
            end
            obj.derivativeLabel = derivativeLabel;                       %#ok
            obj.sorting = 1:size(obj.data,3);
        end
        %%
        function hFigure = plot(obj,sortOrder,channel)
            Nobj = length(obj);
            if Nobj > 1, plotArray(obj);return;end  
            if nargin < 2, sortOrder = 1:size(obj.data,3);end
            if nargin < 3, channel = 1;end
            
            channel = channel(1);
            X = obj.data(:,:,sortOrder,channel);
            XY = obj.xy(:,:,channel);
            %X = bsxfun(@rdivide,X,std(X,[],3));
            mu = mean(X,3);
            % mu(:,2:end) =  bsxfun(@minus,mu(:,2:end),mean(mu(:,2:end)));
            % X = bsxfun(@minus,X,mu);
            
            
            X = permute(X,[1 3 2]);
            nd = size(X,3);
            ind = 1:size(X,2);
            hFigure = figure('Name',['Condition: ' obj.condition],'Color',[0.93 0.96 1]);
            for it=1:nd
                h = subplot(3,nd,it);imagesc(obj.timeStamp,ind,X(:,obj.sorting,it)');
                mx = max(max(abs(X(:,:,it))));
                set(h,'YDir','normal','Clim',[-mx mx],'Tag','kprofiles');
                title(obj.derivativeLabel{it},'FontSize',14);
                xlabel('Time (sec)','FontSize',14);
                ylabel('Trials','FontSize',14);
                h = subplot(3,nd,nd+it);plot(obj.timeStamp,mu(:,it));
                set(h,'Xlim',obj.timeStamp([1 end]),'Tag','kprofiles')
                title(['Mean ' obj.derivativeLabel{it}],'FontSize',14);
                xlabel('Time (sec)','FontSize',14);grid on;
                if it > 1
                    ylabel(texlabel([obj.derivativeLabel{it} ' (m/s^' num2str(it) ')'],'literal'),'FontSize',14);
                else
                    ylabel([obj.derivativeLabel{it} ' (m/s)'],'FontSize',14);
                end
                grid on;
                
                h = subplot(3,nd,2*nd+it);scatter(XY(:,1),XY(:,2),'filled','CData',mu(:,it));title(['Mean trajectory colored by ' obj.derivativeLabel{it}],'FontSize',14);
                mx = max(abs(mu(:,it)));
                set(h,'Clim',[-mx mx]);
                xlabel('X (mm)','FontSize',14);
                ylabel('Y (mm)','FontSize',14)
                grid on;
            end
        end
        %%
        function hFigure = plotOnTrajectory(obj,kIndex)
            Nobj = length(obj);
            if Nobj < 2, return;end
            if nargin < 2, kIndex = size(obj(1).data,2);end
            labels = {obj.condition};
            hFigure = figure('Color',[0.93 0.96 1]);
            hold on;
            for it=1:Nobj
                One = ones(size(obj(it).data,1),1);
                mu = squeeze(median(obj(it).data(:,kIndex,:),3));
                scatter3(it*One,obj(it).xy(:,1),obj(it).xy(:,2),'filled','CData',mu);
            end
            title(['Mean trajectory colored by mean ' obj(1).derivativeLabel{kIndex}]);
            xlabel('X (mm)','FontSize',14);
            ylabel('Y (mm)','FontSize',14)
            set(gca,'XTickLabel',labels,'XTick',1:Nobj);
            grid on
            rotate3d
        end
        %%
        function rmThis = detectOutliers(obj, threshold,plotFlag)
            if nargin < 2, threshold = 0.99;end
            if nargin < 3, plotFlag = false;end
            rmThis = false(size(obj.data,3),1);
            for it=1:length(obj.channelLabel)
                Y = mds(obj.data(:,:,:,it));
                r = sqrt(sum(Y(:,1:2).^2,2));
                B = raylfit(r);
                th = raylinv(threshold,B);
                rmThis = any([rmThis r>th(1)],2);
            end
            if plotFlag
                figure('Color',[0.93 0.96 1]);hold on;
                scatter(Y(:,1),Y(:,2),'.','linewidth',2);
                scatter(Y(rmThis,1),Y(rmThis,2),'r.','linewidth',2);
                title('Trials');grid on;
                if any(rmThis), legend({'normal' 'outliers'});
                else legend({'normal'});
                end
                axis xy
            end
        end
        %%
        function removeOutliers(obj,rmThis)
            if nargin < 2, rmThis = detectOutliers(obj);end
            if ~any(rmThis), return;end
            data = obj.data(:,:,~rmThis,:);
            fid = fopen(obj.binFile,'w');fwrite(fid,data(:),class(data));fclose(fid);
            obj.mmfObj = memmapfile(obj.binFile,'Format',{class(data) size(data) 'x'},'Writable',true);
            sortingByTrialSimilarity(obj)
        end
        %%
        function sortingByTrialSimilarity(obj)
%             dim = size(obj.data);
%             X = zscore(obj.data);
%             if length(obj.channelLabel) > 1
%                 X = permute(X,[1 3 2]);
%                 X = reshape(X,[dim(1)*dim(3) dim(2)]);                
%             end
            Y = mds(obj.data(:,:,:,1));
            r = sqrt(sum(Y.^2,2));
            [~,obj.sorting] = sort(r);
        end
        %%
        function rmExtremeValuesInSignal(obj)
            Nd = size(obj.data,2);
            for it=1:Nd
                data = squeeze(obj.data(:,it,:));               
                dataSVD = svdDenoising4ERP(data,8);
                obj.data(:,it,:) = dataSVD;
            end
        end
        %%
        function nObj = normalize(obj)
            if length(obj) < 2, nObj = [];return;end
            
            T = diff(obj(1).time(1:2));
            N = length(obj);
            maxLength = zeros(N,1);
            for it=1:N, maxLength(it) = length(obj(it).time);end
            maxLength = round(mean(maxLength));
            
            time = (0:maxLength-1)*T;
            nObj = copyobj(obj);
            
            % normalize in time
            for it=1:N
                t = linspace(0,time(end),length(obj(it).time));
                nObj(it).xy = interp1(t,obj(it).xy,time);
                dim = size(obj(it).data);
                data = reshape(obj(it).data,[dim(1) prod(dim(2:end))]);
                datai = interp1(t,data,time);
                nObj(it).data = reshape(datai,[maxLength dim(2:end)]);
                nObj(it).time = time;
            end
            
            xLimits = zeros(N,1);
            yLimits = zeros(N,1);
            
            % normalize in xy
            %for it=1:N
                %nObj(it).xy(:,1) = nObj(it).xy(:,1) - nObj(it).xy(1,1);
                %nObj(it).xy(:,2) = nObj(it).xy(:,2) - nObj(it).xy(1,2);
                %ang = -pi/2+acos(dot( nObj(it).xy(end,:), [0 1] )./sqrt(sum(nObj(it).xy(end,:).^2)));
                %R = [cos(ang) -sin(ang);sin(ang) cos(ang)];
                %nObj(it).xy = nObj(it).xy*R';
            %end
            
            for it=1:N
                xLimits(it) = diff(nObj(it).xy([1 end],1));
                yLimits(it) = max(abs(nObj(it).xy(:,2)));
            end          
                        
            meanX = mean(xLimits);
            meanY = mean(yLimits);
%             k = xLimits.* yLimits/(meanX* meanY);
            
%             nd = size(nObj(1).data,2);
            for it=1:N
                nObj(it).xy = [nObj(it).xy(:,1)./xLimits(it)*meanX nObj(it).xy(:,2)./yLimits(it)*meanY];
                nObj(it).data = bsxfun(@rdivide,nObj(it).data,std(nObj(it).data));
                %for jt=1:nd
                    % nObj(it).data(:,jt,:) = nObj(it).data(:,jt,:)*k(it)^jt;
                %end
            end
            
%             for it=1:N
%                 nObj(it).xy(:,1) = nObj(it).xy(:,1) + meanX/2;
%                 nObj(it).xy(:,2) = nObj(it).xy(:,2) + meanY/2;
%             end
        end
        %%
        function nObj = scale(obj)
            if length(obj) < 2, nObj = [];return;end
            N = length(obj);
            xLimits = zeros(N,1);
            yLimits = zeros(N,1);
            nObj = copyobj(obj);
            for it=1:N
                xLimits(it) = diff(nObj(it).xy([1 end],1));
                yLimits(it) = max(abs(nObj(it).xy(:,2)));
            end                
            meanX = mean(xLimits);
            meanY = mean(yLimits);
            % k = xLimits.* yLimits/(meanX* meanY);
            % nd = size(nObj(1).data,2);
            for it=1:N
                nObj(it).xy = [nObj(it).xy(:,1)./xLimits(it)*meanX nObj(it).xy(:,2)./yLimits(it)*meanY];
                % for jt=1:nd, nObj(it).data(:,jt,:) = nObj(it).data(:,jt,:)*k(it)^jt;end
            end
        end
        %%
        function mu = subtractMean(obj)
            N = length(obj);
            if N==1, return;end
            y = cell(N,1);
            yi = cell(N,1);
            ni = zeros(N,1);
            for it=1:N
                y{it} = squeeze(obj(it).data);
                ni(it) = length(obj(it).time);
            end
            n = round(median(ni));
            xi = 1:n;
            for it=1:N
                x = linspace(1,n,ni(it));
                yi{it} = interp1(x,y{it},xi,'linear');
            end
            mu = 0;
            for it=1:N, mu = mu + mean(yi{it},3);end
            mu = mu/N;
            for it=1:N
                x = linspace(1,ni(it),n);
                xi = 1:ni(it);
                mui = interp1(x,mu,xi,'linear');
                obj(it).data = bsxfun(@minus,obj(it).data,mui);
            end
        end
        %%
        function changeTimeBase(obj,newTime)
            maxLength = length(newTime);
            for it=1:length(obj)
                ti = linspace(obj(it).time(1),obj(it).time(end),maxLength);
                obj(it).xy = interp1(obj(it).time,obj(it).xy,ti);
                dim = size(obj(it).data);
                data = reshape(obj(it).data,[dim(1) prod(dim(2:end))]);
                datai = interp1(obj(it).time,data,ti);
                obj(it).data = reshape(datai,[maxLength dim(2:end)]);
                obj(it).time = newTime;
            end
        end
        %%
        function [coefficients,ersp,itc,frequency,time] = waveletTimeFrequencyAnalysis(obj,wname,fmin,fmax,numFreq,plotFlag)
            T = diff(obj.timeStamp([1 2]));
            if nargin < 2, wname = 'cmor1-1.5';end
            if nargin < 3, fmin = 0.01/T;end
            if nargin < 4, fmax = 1/T/2;end
            if nargin < 5, numFreq = 64;end
            if nargin < 6, plotFlag = true;end
                        
            data = obj.mmfHandler.Data.x;
            data = bsxfun(@minus,data,mean(data,3));
            dim = size(data);
            data = reshape(data,[size(data,1) prod(dim(2:end))]);
            scales = freq2scales(fmin, fmax, numFreq, wname, T);
            frequency = scal2frq(scales,wname,T);
            frequency = fliplr(frequency);
            toCut = round(0.05*length(obj.timeStamp));
            preStimulusMaxLatency = 1:round(1/T/100);
            time = obj.timeStamp(toCut:end-toCut);
            
            %-- computing wavelet coefficients
            coefficients = zeros([length(scales) dim(1) prod(dim(2:end))]);
            for it=1:prod(dim(2:end)), coefficients(:,:,it) = cwt(data(:,it),scales,wname);end
            
            % fliping frequency dimension
            coefficients = permute(coefficients,[2 1 3]);
            coefficients = reshape(coefficients,[dim(1) length(scales) dim(2:end)]);
            coefficients = flipdim(coefficients,2);
            
            coefficientsDB = 10*log10(abs(coefficients).^2+eps);
            base = mean(coefficientsDB(preStimulusMaxLatency,:,:,:));
            ersp = bsxfun(@minus,coefficientsDB,(base)+eps);
            ersp = squeeze(mean(ersp,ndims(ersp)));
            ersp = ersp(toCut:end-toCut,:,:);
            if plotFlag
                for it=1:size(ersp,3)
                    imageLogData(time,frequency,ersp(:,:,it));
                    title(['ERSP (dB) ' obj.derivativeLabel{it} '  Condition: ' obj.condition])
                end
            end
            
            
            itc = coefficients./abs(coefficients);
            itc = squeeze(abs(mean(itc,3)));
            itc = itc(toCut:end-toCut,:,:);
            if plotFlag
                for it=1:size(ersp,3)
                    strTitle = ['ITC ' obj.derivativeLabel{it} '  Condition: ' obj.condition];
                    imageLogData(time,frequency,itc(:,:,it),strTitle);
                end
            end
            
            coefficients = coefficients(toCut:end-toCut,:,:,:);
        end
        %%
        function movieFile = movieMaker(obj,backgroundColor,tailLength,cycles,period,axesLimits)
            if nargin < 2, backgroundColor = [0 0 0];end
            if nargin < 3, tailLength = round(0.1*length(obj(1).time));end
            if isempty(tailLength), tailLength = round(0.1*length(obj(1).time));end
            if nargin < 4, cycles = 2;end
            if nargin < 5, period = 0.25;end
            N = length(obj);
            if nargin < 6
                axesLimits = zeros(N,2);
                for it=1:N
                    axesLimits(it,:) = max(abs(obj(it).xy));
                end
                axesLimits = max(axesLimits);
            end
            
            if N > 1
                movieFile = cell(N,1);
                for it=1:N, movieFile{it} = movieMaker(obj(it),backgroundColor,tailLength,cycles,period,axesLimits);end
                return
            end
            
            color = gray(tailLength);
            if mean(backgroundColor) > 0.5, color = flipud(color);end
            
            openFrames = ones(tailLength,1)*obj.xy(1,:);
            X = repmat(obj.xy(:,1)-mean(obj.xy(:,1)),1,cycles*2);
            Y = repmat(obj.xy(:,2)-mean(obj.xy(:,2)),1,cycles*2);
            for it=2:2:cycles*2
                X(:,it) = flipud(X(:,it));
                Y(:,it) = flipud(Y(:,it));
            end
            X = [openFrames(:,1);X(:);openFrames(:,1)];
            Y = [openFrames(:,2);Y(:);openFrames(:,2)];
            time = (0:length(X)-1)*diff(obj.timeStamp([1 2]));
            
            hFigure = figure('MenuBar','None','ToolBar','None','Renderer','zbuffer','Color',[0.93 0.96 1],'Units','Points','Resize','off','NumberTitle','off','Name',obj.condition);
            markerHandle = scatter(X(1:tailLength),Y(1:tailLength),'filled','CData',color);
            hAxes = findobj(hFigure,'Type','axes');
            set(hAxes,'Units','Points','Color',backgroundColor,'YTickLabel',[],'XTickLabel',[],'Xlim',1.4*(axesLimits(1)*[-1 1]),'Ylim',1.4*(axesLimits(1)*[-1 1]));
            % set(hAxes,'Units','Points','Color',backgroundColor,'YTickLabel',[],'XTickLabel',[],'Xlim',1.4*[min(X) max(X)],'Ylim',1.4*[min(Y) max(Y)]);
            set(markerHandle,'userData',{[X Y time(:)] tailLength});
            drawnow
            
            movieFile = [obj.subjectID '_' obj.condition '.avi']; 
            videoMaker(obj, time(1), time(end), period, movieFile, markerHandle, @mocapEpoch.paintCallback);
        end
    end
    methods(Static)
        function paintCallback(hGraphic,nowCursor)
            
            userData = get(hGraphic,'userData');
            data = userData{1}(:,1:2);
            time = userData{1}(:,3)';
            tailLength = userData{2};
            t = interp1(time,1:length(time),nowCursor,'nearest');
            
            try 
                x = data(t-tailLength+1:t,1);
                y = data(t-tailLength+1:t,2);
            catch
                delta = t-tailLength;
                if delta < 0
                    x = [data(1:abs(delta),1); data(1:t,1)];
                    y = [data(1:abs(delta),2); data(1:t,2)];
                end
            end
            set(hGraphic,'XData',x,'YData',y);
        end
    end
    methods(Hidden = true)
        %%
        function hFigure = plotArray(obj)
            Nobj = length(obj);
            if Nobj < 2, return;end
            try
                dim = size(obj(1).data);
                X = zeros([dim(1:2) Nobj]);
                XY   = zeros([dim(1) 2 Nobj]);
                for it=1:Nobj
                    X(:,:,it) = mean(obj(it).data,3);
                    XY(:,:,it) = obj(it).xy;
                end
                labels = {obj.condition};
                X = permute(X,[1 3 2]);
                nd = size(X,3);
                ind = 1:size(X,2);
                One = ones(dim(1),1);
                hFigure = figure('Color',[0.93 0.96 1]);
                for it=1:nd
                    hAxes = subplot(2,nd,it);hold on;
                    % tmp = X(:,:,it);
                    % color = bsxfun(@minus,tmp,mean(tmp,2));
                    color = X(:,:,it);
                    mx = max(max(abs(color)));
                    imagesc(obj(1).time,ind,color');
                    set(hAxes,'YDir','normal','yTickLabel',labels,'YTick',1:Nobj,'Clim',[-mx mx],'Xlim',obj(1).time([1 end]),'Ylim',[1 Nobj]);
                    title(obj(1).derivativeLabel{it},'FontSize',14);
                    xlabel('Time (sec)','FontSize',14);
                    ylabel('Trials','FontSize',14);
                    
                    hAxes = subplot(2,nd,nd+it);hold on;
                    for jt=1:Nobj, scatter3((Nobj-jt)*One,obj(jt).xy(:,1),obj(jt).xy(:,2),'filled','CData',color(:,jt));end
                    view([102 34]);
                    title(obj(1).derivativeLabel{it},'FontSize',14);
                    ylabel('X (mm)','FontSize',14);
                    zlabel('Y (mm)','FontSize',14)
                    set(hAxes,'xTickLabel',fliplr({obj.condition}),'XTick',0:Nobj-1,'Clim',[-mx mx]);
                    grid on
                end
            catch ME
                if strcmp(ME.identifier,'MATLAB:subsassigndimmismatch')
                    disp('To plot epochs with different length they have to be normalized first. Run "normalize(obj)", where obj is the array of epochs.')
                else
                    ME.rethrow;
                end
            end
        end
    end
end

%--
function imageLogData(time,frequency,data,strTitle)
if nargin < 4, strTitle = '';end
figure('Color',[0.93 0.96 1]);
imagesc(time,log10(frequency),data');
hAxes = gca;
tick = get(hAxes,'Ytick');
fval = 10.^tick;
Nf = length(tick);
yLabel = cell(Nf,1);
fval(fval >= 10) = round(fval(fval >= 10));
for it=1:Nf, yLabel{it} = num2str(fval(it),3);end
mx = max(data(:));
if min(data(:)) < 0,
    mn = -mx;
else
    mn = min(data(:));
end
set(hAxes,'YDir','normal','Ytick',tick,'YTickLabel',yLabel,'CLim',[mn mx]);
[~,loc] = min(abs(time));
hold(hAxes,'on');plot([1 1]*time(loc),get(hAxes,'YLim'),'k-.','LineWidth',2);
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title(strTitle)
colorbar;
end
