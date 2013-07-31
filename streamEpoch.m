classdef streamEpoch < epochObject
    methods
        function obj = streamEpoch(varargin)
            N = length(varargin);
            if N==1, varargin = varargin{1};end
            obj@epochObject(varargin);
        end
        %%
        function hFigure = plot(obj,channel)
            dim = obj.mmfObj.Format{1,2};
            if length(dim) == 3, Nch = dim(3);else Nch = 1;end
            if nargin < 2, channel = 1:Nch;end
            Nch = length(channel);
            if Nch > 1
                for it=1:Nch, hFigure = plot(obj,channel(it));end
                return
            end
            
            if nargin < 2, channel = 1;end
            channel = channel(1);
            X = obj.data(:,obj.sorting,channel);
            mu = geometric_median(X');
            p = size(X,2);
            [~,loc] = min(abs(obj.time));
            hFigure = figure('Color',[0.93 0.96 1]);
            subplot(211);imagesc(obj.time,1:p,X');set(gca,'YDir','normal');
            title(['Trials ' obj.channelLabel{channel} '  Condition: ' obj.condition]);xlabel('Time (sec)')
            hold on;plot([1 1]*obj.time(loc),get(gca,'YLim'),'k-.','LineWidth',2);
            subplot(212);plot(obj.time,mu);title(['Mean ' obj.condition]);
            hold on;plot([1 1]*obj.time(loc),get(gca,'YLim'),'k-.','LineWidth',2);
            title(['Mean ' obj.channelLabel{channel} '  Condition: ' obj.condition]);
        end
        %%
        function rmThis = detectOutliers(obj,threshold)
            if nargin < 2, threshold = 0.99;end
            rmThis = false(size(obj.data,2),1);
            for it=1:length(obj.channelLabel)
                D= pdist(obj.data(:,:,it)');
                Y = mdscale(D,3);
                r = sqrt(sum(Y.^2,2));
                th = raylinv(threshold, raylfit(r));
                rmThis = any([rmThis r>th],2);
            end
            figure('Color',[0.93 0.96 1]);
            scatter(Y(:,1),Y(:,2),'.','linewidth',2);hold on;scatter(Y(rmThis,1),Y(rmThis,2),'r.','linewidth',2);title('MDS trials');grid on;
        end
        function removeOutliers(obj,rmThis)
            persistent flag
            if ~isempty(flag), disp('You have removed outliers already.');return;end
            if nargin < 2, rmThis = detectOuliers(obj,0.95);end
            data = obj.data(:,~rmThis,:);
            fid = fopen(obj.mmfName,'w');fwrite(fid,data(:),class(data));fclose(fid);
            obj.mmfObj = memmapfile(obj.mmfName,'Format',{class(data) size(data) 'x'},'Writable',true);
            flag = 1;
        end
        function sortingByTrialSimilarity(obj)
            dim = size(obj.data);
            % dataSVD = svdDenoising4ERP(obj.data(:,:,1),8);
            % X = zscore(dataSVD);
            X = zscore(obj.data);
            if length(obj.channelLabel) > 1
                X = permute(X,[1 3 2]);
                X = reshape(X,[dim(1)*dim(3) dim(2)]);                
            end
            D= pdist(X');
            Y = mdscale(D,2);
            r = sqrt(sum(Y.^2,2));
            [~,obj.sorting] = sort(r);
        end
        %%
        function [coefficients,ersp,itc,frequency,time] = waveletTimeFrequencyAnalysis(obj,wname,fmin,fmax,numFreq,plotFlag,numberOfBoundarySamples,multCompCorrectionMethod, varargin)
            T = diff(obj.time([1 2]));
            if nargin < 2, wname = 'cmor1-1.5';end
            if nargin < 3, fmin = 1;end
            if nargin < 4, fmax = 1/T/2;end
            if nargin < 5, numFreq = 64;end
            if nargin < 6, plotFlag = true;end
            if nargin < 7, numberOfBoundarySamples = 0;end
            if nargin < 8, multCompCorrectionMethod = 'none';end
           
            preStimulusMaxLatency = [1 floor(length(obj.time)/2)]; 
            data = obj.mmfObj.Data.x;
            dim = size(obj.data);
            data = reshape(data,[size(data,1) prod(dim(2:end))]);
            scales = freq2scales(fmin, fmax, numFreq, wname, T);
            frequency = scal2frq(scales,wname,T);
            frequency = fliplr(frequency);
            
            if ~numberOfBoundarySamples
                toCut = round(0.05*length(obj.time));
            else
                toCut = numberOfBoundarySamples;
            end
            time = obj.time(toCut:end-toCut-1);
            %-- computing wavelet coefficients
            coefficients = zeros([length(scales) dim(1) prod(dim(2:end))]);
            hwait = waitbar(0,'Computing cwt...','Color',[0.93 0.96 1]);
            prodDim = prod(dim(2:end));
            for it=1:prodDim
               coefficients(:,:,it) = cwt(data(:,it),scales,wname);
               waitbar(it/prodDim,hwait);
            end
            close(hwait);
            
            
            % fliping frequency dimension
            coefficients = permute(coefficients,[2 1 3]);
            coefficients = reshape(coefficients,[dim(1) length(scales) dim(2:end)]);
            coefficients = flipdim(coefficients,2);
                        
            if toCut > preStimulusMaxLatency(1), t1 = toCut; else t1 = preStimulusMaxLatency(1);end
            if length(obj.time)-toCut <= preStimulusMaxLatency(2)
                preStimulusMaxLatency(2) = length(obj.time)-toCut-t1;
                t2 = preStimulusMaxLatency(2);
            else
                t2 = preStimulusMaxLatency(2);
            end
            
            coefficientsDB = 10*log10(abs(coefficients).^2+eps);
            base = mean(coefficientsDB(t1:t2,:,:,:));
            coefficients   = coefficients(toCut:end-toCut-1,:,:,:);
            coefficientsDB = 10*log10(abs(coefficients).^2+eps);
            
            ersp = bsxfun(@minus,coefficientsDB,(base)+eps);
            ersp = squeeze(mean(ersp,3));
                        
            itc = coefficients./abs(coefficients);
            itc = squeeze(abs(mean(itc,3)));
            
            Nv = length(varargin);
            switch multCompCorrectionMethod
                case 'none'
                    % disp('Not significance test was computed.');
                case 'bootstrap'
                    if Nv < 1, nboot = 1000; else nboot = varargin{1};end
                    if Nv < 2, alpha = 0.05; else alpha = varargin{2};end
                    
                    % ersp
                    coefficientsDB = permute(coefficientsDB,[3 setdiff(1:ndims(coefficientsDB),3)]);
                    dim = size(coefficientsDB);
                    coefficientsDB = reshape(coefficientsDB,[dim(1) prod(dim(2:end))]);
                    bootstat = bootstrp(nboot,@boots_ersp,coefficientsDB,ones(dim(1),1)*[t1 t2],ones(dim(1),1)*dim);
                    bootstat = reshape(bootstat,[nboot dim(2:end)]);
                    ersp = reshape(ersp,[prod(dim(2:3)) length(obj.channelLabel)]);
                    I1 = false(prod(dim(2:3)),length(obj.channelLabel));
                    I2 = false(prod(dim(2:3)),length(obj.channelLabel));
                    for it=1:length(obj.channelLabel)
                        tmp = bootstat(:,:,:,it);
                        tmp = reshape(tmp,[nboot prod(dim(2:3))]);
                        maxmin = prctile(tmp,100*[alpha 1-alpha],2);
                        % th = [min(th(:,1)) max(th(:,2))];
                        th(1) = prctile(maxmin(:,1),100*alpha);
                        th(2) = prctile(maxmin(:,2),100*(1-alpha));
                        I = ersp(:,it) > th(1) & ersp(:,it) < th(2);
                        I1(:,it) = I;
                        ersp(I,it) = 0;
                    end
                    ersp = reshape(ersp,[dim(2:3) length(obj.channelLabel)]);
                    
                    % itc
                    coefficientsTmp = permute(coefficients,[3 setdiff(1:ndims(coefficients),3)]);
                    coefficientsTmp = reshape(coefficientsTmp,[dim(1) prod(dim(2:end))]);
                    bootstat = bootstrp(nboot,@boots_itc,coefficientsTmp,ones(dim(1),1)*dim);
                    bootstat = reshape(bootstat,[nboot dim(2:end)]);
                    itc = reshape(itc,[prod(dim(2:3)) length(obj.channelLabel)]);
                    
                    for it=1:length(obj.channelLabel)
                        % th = raylinv((1-alpha), raylfit(itc(:,it)));
                        tmp = bootstat(:,:,:,it);
                        tmp = reshape(tmp,[nboot prod(dim(2:3))]);
                        th = prctile(tmp,100*(1-alpha),2);
                        th = prctile(th,100*(1-alpha));
                        I = itc(:,it) < th;
                        I2(:,it) = I;
                        itc(I,it) = 0;
                    end
                    itc = reshape(itc,[dim(2:3) length(obj.channelLabel)]);
                    
                otherwise
                    error('Unknown method. Stick to bootstrap by now.');
            end
            
            if plotFlag
                G = fspecial('gaussian',[4 4],2);
                ersp_s = ersp;
                itc_s = itc;
                for it=1:length(obj.channelLabel)
                    ersp_s(:,:,it) = imfilter(ersp_s(:,:,it),G,'same');
                    itc_s(:,:,it)  = imfilter(itc_s(:,:,it), G,'same');
                    %ersp(I1) = 0;
                    %itc(I2) = 0;
                    
                    eegEpoch.imageLogData(time,frequency,ersp_s(:,:,it));
                    title(['ERSP (dB) ' obj.channelLabel{it} '  Condition: ' obj.condition]);
                    
                    strTitle = ['ITC ' obj.channelLabel{it} '  Condition: ' obj.condition];
                    eegEpoch.imageLogData(time,frequency,itc_s(:,:,it),strTitle);
                end
            end
            
        end
    end
    methods(Static)
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
    end
end
