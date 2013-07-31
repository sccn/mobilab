classdef tfEpoch < epochObject
    properties
        frequency
        base
        headModelInfo
        ersp 
        itc
    end
    properties(Dependent)
        tfData
    end
    properties(Hidden,SetAccess=protected)
        mmfObjTF
        mmfNameTF
        svdFilter = 1;
    end
    methods
        function obj = tfEpoch(varargin)
            N = length(varargin);
            if N==1, varargin = varargin{1};N = length(varargin);end
            
            ind = find(ismember(varargin(1:2:length(varargin)-1),'frequency'));
            if ~isempty(ind), frequency = varargin{ind*2}; else error('Not enough input agguments.');end %#ok
            
            ind = find(ismember(varargin(1:2:length(varargin)-1),'tfData'));
            if ~isempty(ind), tfData = varargin{ind*2}; else error('Not enough input agguments.');end
            
            ind = find(ismember(varargin(1:2:length(varargin)-1),'headModelInfo'));
            if ~isempty(ind), headModelInfo = varargin{ind*2}; else error('Not enough input agguments.');end %#ok
            
            ind = find(ismember(varargin(1:2:length(varargin)-1),'base'));
            if ~isempty(ind), base = varargin{ind*2}; else base = [];end %#ok
            
            ind = find(ismember(varargin(1:2:length(varargin)-1),'svdFilter'));
            if ~isempty(ind), svdFilter = varargin{ind*2}; else svdFilter = 1;end %#ok
            
            obj@epochObject(varargin);
            
            % [~,name] = fileparts(tempname);
            % obj.mmfNameTF = fullfile(pwd,['.' name]);
            obj.mmfNameTF = tempname;
            fid = fopen(obj.mmfNameTF,'w');
            classData = class(tfData(1));
            fwrite(fid,real(tfData),classData);
            fwrite(fid,imag(tfData),classData);
            fclose(fid);
            obj.mmfObjTF = memmapfile(obj.mmfNameTF,'Format',{classData size(tfData) 'x';classData size(tfData) 'y'},'Writable',true);
            
            obj.frequency = frequency; %#ok
            obj.headModelInfo = headModelInfo; %#ok
            obj.base = base; %#ok
            obj.svdFilter = svdFilter; %#ok
            obj.computeERSP;
            obj.computeITC;
            obj.sorting = 1:size(obj.data,2);
            % obj.sortingByTrialSimilarity;
            % obj.rmExtremeValuesInSignal;
            
        end
        %%
        function ersp = get.ersp(obj)
            %if isempty(obj.ersp), obj.ersp = computeERSP(obj);end
            ersp = obj.ersp;
        end
        %%
        function itc = get.itc(obj)
            %if isempty(obj.itc), obj.itc = computeITC(obj);end
            itc = obj.itc;
        end
        %%
        function sortingByTrialSimilarity(obj)
            dim = size(obj.data);
            
            dataSVD = svdDenoising4ERP(obj.data(:,:,1),8);
            X = zscore(dataSVD);
            % X = zscore(obj.data);
            if length(obj.channelLabel) > 1
                X = permute(X,[1 3 2]);
                X = reshape(X,[dim(1)*dim(3) dim(2)]);                
            end
            D= pdist(X');
            Y = cmdscale(D);
            r = sqrt(sum(Y.^2,2));
            [~,obj.sorting] = sort(r);
        end
        %%
        function rmThis = detectOuliers(obj,threshold)
            if nargin < 2, threshold = 0.99;end
            rmThis1 = false(size(obj.data,2),1);
            rmThis2 = false(size(obj.data,2),1);
            for it=1:length(obj.channelLabel)
                D= pdist(obj.data(:,:,it)');
                Y1 = cmdscale(D);
                r1 = sqrt(sum(Y1(:,1:2).^2,2));
                Y2 = mds(abs(obj.tfData(:,:,:,it)).^2);
                r2 = sqrt(sum(Y2(:,1:2).^2,2));
                th = [raylinv(threshold, raylfit(r1)) raylinv(threshold, raylfit(r2))];
                rmThis1 = any([rmThis1 r1>th(1)],2);
                rmThis2 = any([rmThis2 r2>th(2)],2);
            end
            figure('Color',[0.93 0.96 1]);
            subplot(121);scatter(Y1(:,1),Y1(:,2),'.','linewidth',2);hold on;scatter(Y1(rmThis1,1),Y1(rmThis1,2),'r.','linewidth',2);title('MDS spectrum trials');grid on;
            subplot(122);scatter(Y2(:,1),Y2(:,2),'.','linewidth',2);hold on;scatter(Y2(rmThis2,1),Y2(rmThis2,2),'r.','linewidth',2);title('MDS component trials');grid on;
            rmThis = rmThis1 | rmThis2;
        end
        %%
        function rmExtremeValuesInSignal(obj)
            warning off all;
            options = statset('Display','off');
            gm = gmdistribution.fit(obj.data(:),2,'Options',options);
            cl = gm.cluster(obj.data(:));
            [~,sloc] = sort(gm.Sigma);
            if gm.PComponents(sloc(1))/gm.PComponents(sloc(2)) > 2
                outliers = cl==sloc(2);
                p = gm.posterior(obj.data(outliers));
                obj.data(outliers) = p(:,sloc(1)).*obj.data(outliers);
                disp('Removing outliers.')
            end
            warning on all;

%             z = zscore(obj.data(:));
%             th = norminv([0.01 0.99]);
%             I = z < th(1) | z > th(2);
%             dim = size(obj.data);
%             [xi,yi] = ind2sub(dim,find(I));
%             [x,y] = ind2sub(dim,find(~I));
%             xi = obj.time(xi);
%             x = obj.time(x);
%             tmp = (1:dim(1))';
%             y = tmp(y);
%             
%             F = TriScatteredInterp(x,y,obj.data(~I));
%             obj.data(I) = F(xi,yi);
%             
%             order = 3;
%             win = 127;
%             sdata = obj.data(:);
%             dim = size(sdata);
%             sdata = smooth(sdata,win,'sgolay',order);
%             obj.data = reshape(sdata,dim);
        end
        %%
        function removeOutliers(obj,rmThis)
            persistent flag
            if ~isempty(flag), disp('You have removed outliers already.');return;end
            if nargin < 2, rmThis = detectOuliers(obj,0.95);end
            data = obj.data(:,~rmThis,:);
            tfData = obj.tfData(:,:,~rmThis,:);
            flag = 1;
            
            fid = fopen(obj.mmfName,'w');fwrite(fid,data(:),class(data));fclose(fid);
            obj.mmfObj = memmapfile(obj.mmfName,'Format',{class(data) size(data) 'x'},'Writable',true);
            
            classData = class(tfData(1));
            fid = fopen(obj.mmfNameTF,'w');
            fwrite(fid,real(tfData),classData);
            fwrite(fid,imag(tfData),classData);
            fclose(fid);
            obj.mmfObjTF = memmapfile(obj.mmfNameTF,'Format',{classData size(tfData) 'x';classData size(tfData) 'y'},'Writable',true);
            
            sortingByTrialSimilarity(obj);
            computeERSP(obj);
            computeITC(obj);
        end
        %%
                    
        %%
        function tfData = get.tfData(obj)
            tfData = obj.mmfObjTF.Data.x + 1i*obj.mmfObjTF.Data.y;
        end
        %%
        function set.tfData(obj,tfData)
            obj.mmfObjTF.Data.x = real(tfData);
            obj.mmfObjTF.Data.y = imag(tfData);
        end 
        %%
        function delete(obj)
            % obj.mmfObjTF = [];
            delete(obj.mmfNameTF);
        end
        %%
        function hFigure = plot(obj,channel)
            dim = obj.mmfObjTF.Format{1,2};
            if nargin < 2, channel = 1:dim(end);end
            Nch = length(channel);
            if Nch>1
                for it=1:Nch, hFigure = plot(obj,channel(it));end
                return
            end
            erspData = obj.ersp;
            erspData = erspData(:,:,channel);
                       
            hFigure = figure('Name',obj.condition,'Color',[0.93 0.96 1],'renderer','opengl');
            subplot(221);
            h = imagesc(obj.time,log10(obj.frequency),erspData');
            hAxes = get(h,'parent');
            tick = get(hAxes,'Ytick');
            fval = 10.^tick;
            Nf = length(tick);
            yLabel = cell(Nf,1);
            fval(fval >= 10) = round(fval(fval >= 10));
            for it=1:Nf, yLabel{it} = num2str(fval(it),3);end
            mx = max(erspData(:));
            set(hAxes,'YDir','normal','Ytick',tick,'YTickLabel',yLabel,'Tag','ersp','CLim',[-mx mx],'FontSize',14);
            [~,loc] = min(abs(obj.time));
            hold(hAxes,'on');plot([1 1]*obj.time(loc),get(hAxes,'YLim'),'m-.','LineWidth',2);
            title([obj.channelLabel{channel} ' ERSP condition: ' obj.condition],'FontSize',14);
            xlabel('Time (sec)','FontSize',14);
            ylabel('Frequency (Hz)','FontSize',14);
            cbar_axes = colorbar;
            title(cbar_axes,'ERSP (dB)','FontSize',14)
            
            h = subplot(223);
            itcData = obj.itc;
            itcData = itcData(:,:,channel);
            imagesc(obj.time,log10(obj.frequency),itcData');
            mx = max(itcData(:));
            set(h,'YDir','normal','Ytick',tick,'YTickLabel',yLabel,'Tag','itc','CLim',[0 mx],'FontSize',14);
            hold(h,'on');plot([1 1]*obj.time(loc),get(h,'YLim'),'m-.','LineWidth',2);
            title([obj.channelLabel{channel}  ' ITC condition: ' obj.condition],'FontSize',14);
            xlabel('Time (sec)','FontSize',14);
            ylabel('Frequency (Hz)','FontSize',14);
            colorbar;

            cAxes = subplot(222);set(cAxes,'Tag','cortex');
            csvObj = currentSourceViewer(obj.headModelInfo,obj.headModelInfo.J(:,channel),obj.headModelInfo.V(:,channel),obj.condition);
            csvObj.hLabels = copyobj(csvObj.hLabels,cAxes);
            csvObj.hSensors = copyobj(csvObj.hSensors,cAxes);
            csvObj.hScalp = copyobj(csvObj.hScalp,cAxes);
            csvObj.hCortex = copyobj(csvObj.hCortex,cAxes);
            csvObj.hAxes = cAxes;
           
            toolbarHandle = findall(csvObj.hFigure,'Type','uitoolbar');
            th = copyobj(toolbarHandle,hFigure);
            hcb = findall(th,'TooltipString','Labels On/Off');
            set(hcb,'OnCallback',@(src,event)csvObj.rePaint(hcb(1),'labelsOn'),'OffCallback',@(src, event)csvObj.rePaint(hcb(1),'labelsOff'));
            
            hcb = findall(th,'TooltipString','Sensors On/Off');
            set(hcb,'OnCallback',@(src,event)csvObj.rePaint(hcb(1),'sensorsOn'),'OffCallback',@(src, event)csvObj.rePaint(hcb(1),'sensorsOff'));
            
            hcb = findall(th,'TooltipString','Scalp On/Off');
            set(hcb,'OnCallback',@(src,event)csvObj.rePaint(hcb(1),'scalpOn'),'OffCallback',@(src, event)csvObj.rePaint(hcb(1),'scalpOff'));
            
            view(cAxes,[90 0]);
            axis(cAxes,'equal');
            axis(cAxes,'off')
            val = get(csvObj.hCortex,'FaceVertexCData');
            mx = max(abs([min(val) max(val)]));
            set(cAxes,'Clim',[-mx mx]);
            
            close(csvObj.hFigure);
            csvObj.hFigure = hFigure;
            colorbar;
            
            dcHandle = datacursormode(hFigure);
            set(dcHandle,'UpdateFcn',@(src,event)showLabel(obj,event),'Enable','off','SnapToDataVertex','off');
            set(hFigure,'userData',csvObj);
            camlight(0,180);camlight(0,0)
            
            mPath = which('mobilabApplication');
            if ~isempty(mPath)
                path = fullfile(fileparts(mPath),'skin');
                erpIcon  = imread([path filesep 'erpImage.png']);
            else
                erpIcon = rand(22);
            end
            h = subplot(224);
            %data = mean(zscore(obj.data(:,:,1)),2);
            data = mean(obj.data(:,:,1),2);
            data = data - mean(data);
            plot(obj.time,data);
            hold(h,'on');plot([1 1]*obj.time(loc),get(h,'YLim'),'m-.','LineWidth',2);
            %plot(obj.time,mean(zscore(obj.data(:,:,channel)),2));
            title([obj.channelLabel{channel} ' ERP condition: ' obj.condition],'FontSize',14);
            xlabel('Time (sec)','FontSize',14);
            set(h,'Tag','erp','FontSize',14);   
            hcb = uitoggletool(th,'CData',erpIcon,'HandleVisibility','off','TooltipString','ERP Image','State','off');
            set(hcb,'OnCallback',@(src,event)plotErpImage(obj,h,'on'),'OffCallback',@(src, event)plotErpImage(obj,h,'off'));
            
        end
        %%
        function erspData = computeERSP(obj, base, channel)
            dim = obj.mmfObjTF.Format{1,2};
            if nargin < 2, base = [];end
            if nargin < 3, channel = 1:dim(dim(end));end
            
            Nch = length(channel);
            if Nch > 1
                erspData = zeros(dim(1),dim(2),Nch);
                for it=1:Nch, erspData(:,:,it) = computeERSP(obj, base, channel(it));end
                return
            end
            power = obj.mmfObjTF.Data.x(:,:,:,channel).^2 + obj.mmfObjTF.Data.y(:,:,:,channel).^2;
            if numel(obj.svdFilter) > 1 && any(all(obj.svdFilter(:,:,channel)))
                for it=1:dim(3), power(:,:,it) = log( power(:,:,it))*obj.svdFilter(:,:,channel);end
                power = exp(power);
            else
               obj.svdFilter = zeros(dim(2),dim(2),obj.mmfObjTF.Format{1,2}(end));
               dim = size(power); 
               power = permute(power,[1 3 2]);
               power = reshape(power,[dim(1)*dim(3) dim(2)]);
               [U,S,V] = svds(log(power));
               power = exp(U*S*V');
               power = reshape(power,[dim(1) dim(3) dim(2)]);
               power = permute(power,[1 3 2]);
               iS = diag(1./diag(S));
               obj.svdFilter(:,:,channel) = V*iS*S*V';
            end
            dataDB = 10*log10(power);
            if nargin < 2, 
                base = 10*log10(obj.base(:,:,channel));
            elseif isempty(base)
                base = 10*log10(obj.base(:,:,channel));
            else 
                if length(base) > 1, error('Second input argument ''base'' must be the end of the baseline spectrum (tmin,base]. Enter 0.2 to take the first 200 ms of each trial as a baseline spectra.');end
                [~,loc] = min(abs(obj.time-base));
                baseInd = 1:loc;
                base = mean(squeeze(mean(dataDB(baseInd,:,:,:))),2);
            end
            erspData = bsxfun(@minus,permute(dataDB,[2 3 1]),base(:));
            erspData = permute(erspData,[3 1 2]);
            erspData = mean(erspData,3);
            mu = mean(erspData(:));
            erspData = erspData-mu;
            % erspData = (erspData-mu).*Pm;
            obj.ersp(:,:,channel) = erspData;
        end
        %%
        function  itcData = computeITC(obj)
            P = obj.tfData./abs(obj.tfData);
            itcData = squeeze(abs(mean(P,3)));
            obj.itc = itcData;
        end
    end
end

function output_txt = showLabel(obj,event_obj)
persistent DT
hAxes = gca;
tag = get(hAxes,'Tag');
pos = get(event_obj,'Position');
switch tag
    case 'cortex'
        if isempty(DT)
            load(obj.headModelInfo.surfaces);
            vertices = surfData(3).vertices;
            DT = DelaunayTri(vertices(:,1),vertices(:,2),vertices(:,3));
        end
        loc = nearestNeighbor(DT, pos);
        output_txt = obj.headModelInfo.atlas.label{obj.headModelInfo.atlas.color(loc)};
    case 'ersp'
        [~,t] = min(abs(obj.time-pos(1)));
        [~,f] = min(abs(obj.frequency-10^pos(2)));
        output_txt = sprintf('latency=%0.3f\nfrequebcy=%0.3f\nersp=%0.3f',pos(1),obj.frequency(f),obj.ersp(t,f));
    case 'itc'
        [~,t] = min(abs(obj.time-pos(1)));
        [~,f] = min(abs(obj.frequency-10^pos(2)));
        output_txt = sprintf('latency=%0.3f\nfrequebcy=%0.3f\nitc=%0.3f',pos(1),obj.frequency(f),obj.itc(t,f));
    case {'erp' 'erp2'}
        output_txt = sprintf('latency=%0.3f\nvoltage=%0.3f',pos(1),pos(2));
    otherwise
        output_txt = '';
end
end
%%
function plotErpImage(obj,hAxes,state)
ylimits = getappdata(hAxes,'ylimits');
if isempty(ylimits)
    ylimits = get(hAxes,'YLim');
    dataSVD = svdDenoising4ERP(obj.data(:,:,1),8);
    setappdata(hAxes,'ylimits',ylimits);
    setappdata(hAxes,'dataSVD',dataSVD);
end
switch state
    case 'on'
        dataSVD = getappdata(hAxes,'dataSVD');
        cla(hAxes);
        imagesc(obj.time,1:size(obj.data,2),dataSVD(:,obj.sorting)','Parent',hAxes);
        title(hAxes,[obj.channelLabel{1} ' ERP Image condition: ' obj.condition]);
        xlabel(hAxes,'Time (sec)');
        ylabel(hAxes,'Trials');
        set(hAxes,'Tag','erp2','Xlim',obj.time([1 end]),'YLim',[1 size(obj.data,2)]);
        colorbar('peer',hAxes)
    case 'off'
        [~,loc] = min(abs(obj.time));
        cla(hAxes);
        % erp = mean(zscore(obj.data(:,:,1)),2);
        erp = mean(obj.data(:,:,1),2);
        erp = erp - mean(erp);
        plot(hAxes,obj.time,erp);
        set(hAxes,'Ylim',ylimits);
        hold(hAxes,'on');plot(hAxes,[1 1]*obj.time(loc),get(hAxes,'YLim'),'m-.','LineWidth',2);
        title(hAxes,[obj.channelLabel{1} ' ERP condition: ' obj.condition]);
        xlabel(hAxes,'Time (sec)');
        ylabel(hAxes,'');
        set(hAxes,'Tag','erp');
        colorbar('off','peer',hAxes)
end
end