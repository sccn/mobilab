classdef mocapEEGdataset < handle
    properties
        mocapTrials
        eegTrials
        boundary
        ersp
        itc
        frequency
    end
    properties(Dependent)
        time
        condition
    end
    methods 
        function obj = mocapEEGdataset(mocapObj,mocapChannels,eegObj,eegChannels,latency,condition,timeLimits)

            if size(latency,2) > 1
                unwrapAxis = true;
                obj.boundary = 0.25*max(mean(diff(latency,[],2)));
                latency = [latency(:,1)-obj.boundary latency latency(:,end)+obj.boundary];
                dim = size(latency);

                latency1 = reshape( mocapObj.getTimeIndex(latency(:)),dim);
                obj.mocapTrials = mkEpochTW(mocapObj, latency1, mocapChannels, condition, unwrapAxis);
                
                latency1 = reshape( eegObj.getTimeIndex(latency(:)),dim);
                obj.eegTrials   = mkEpochTW(eegObj,   latency1, eegChannels,   condition);
                obj.eegTrials.preStimulusMaxLatency = [1 length(obj.eegTrials.time)];
            else
                dim = size(latency);
                latency1 = reshape( mocapObj.getTimeIndex(latency(:)),dim);
                obj.mocapTrials = mkEpoch(mocapObj,latency1, timeLimits, mocapChannels, condition);
                
                latency1 = reshape( mocapObj.getTimeIndex(latency(:)),dim);
                obj.eegTrials   = mkEpoch(eegObj,  latency1, timeLimits, eegChannels,   condition);
            end
            if mocapObj.samplingRate ~= eegObj.samplingRate
               xi = obj.eegTrials.time;
               y = obj.mocapTrials.data;
               x = obj.mocapTrials.time; 
               obj.mocapTrials.data = interp1(x,y,xi,'linear');
               obj.mocapTrials.xy = interp1(x,obj.mocapTrials.xy,xi,'linear');
            end
        end
        %%
        function subtractMeanTraces(obj)
            N = length(obj);
            if N==1, return;end
            for it=1:N, tmpArray(it) = obj(it).mocapTrials;end %#ok
            subtractMean(tmpArray);
        end
        %%
        function mu = meanTraces(obj)
            N = length(obj);
            if N==1, return;end
            for it=1:N, tmpArray(it) = obj(it).mocapTrials;end %#ok
            tmpArray = copyobj(tmpArray);
            mu = subtractMean(tmpArray);
        end
        %%
        function plotMocapTrials(obj)
            N = length(obj);
            boundary = cell2mat({obj.boundary}); %#ok
            time = {obj.time};
            for it=1:N
                cArray(it) = copyobj(obj(it).mocapTrials); %#ok
                [~,loc] = min(abs(boundary(it) - (time{it}-min(time{it}))));%#ok
                cArray(it).xy([1:loc end-loc:end],:) = [];%#ok
                cArray(it).data([1:loc end-loc:end],:,:) = [];%#ok
                cArray(it).time([1:loc end-loc:end]) = [];%#ok
            end 
            nArray = normalize(cArray);
            plot(nArray)
            return
            
            Nd = length(obj(1).mocapTrials.derivativeLabel);
            labels = fliplr({obj.condition});   
            toCut = cell2mat({obj.boundary});
            time = {obj.time};
            X = cell(N,1);
            mx = zeros(N,Nd);
            for it=1:N
                X{it} = mean(obj(it).mocapTrials.data,3); 
                mx(it,:) = max(X{it});
            end
            mx = max(mx);
            figure('Color',[0.93 0.96 1]);hold on;    
            for it=1:Nd
                hAxes = subplot(2,Nd,Nd+it);hold on;
                for jt=1:N
                    [~,loc] = min(abs(toCut(jt) - (time{jt}-min(time{jt}))));
                    color = X{jt}(:,it); 
                    nt = length(obj(jt).mocapTrials.xy(loc:end-loc,1));
                    scatter3((N-jt)*ones(nt,1),obj(jt).mocapTrials.xy(loc:end-loc,1),...
                        obj(jt).mocapTrials.xy(loc:end-loc,2),'filled','CData',color(loc:end-loc));
                end
                title(obj(1).mocapTrials.derivativeLabel{it},'FontSize',14);
                ylabel('X (mm)','FontSize',14);
                zlabel('Y (mm)','FontSize',14)
                set(hAxes,'xTickLabel',labels,'XTick',0:N-1,'Clim',[-mx(it) mx(it)]);
                view([78 34]);
                grid on
            end
        end
        %%
        function time = get.time(obj)
            time = obj.eegTrials.time;
        end
        %%
        function condition = get.condition(obj)
            condition = obj.eegTrials.condition;
        end
        %%
        function ersp = get.ersp(obj)
            N = length(obj);
            if isempty(obj(1).ersp), obj.eegTimeFrequencyAnalysis;end
            if N > 1
                ersp = {obj.ersp};
            else
                ersp = obj.ersp;
            end
        end
        %%
        function plot(obj,derivative)
            if nargin < 2, derivative = 1;end
            N = length(obj);
            for it=1:N 
                y = squeeze(obj(it).mocapTrials.data(:,derivative,:));
                y = mean(y,2);
                [~,loc] = min(abs(obj(it).boundary - (obj(it).time-min(obj(it).time))));
                [~,locZero] = min(abs(obj(it).time));
                
                figure('Name',['Condition: ' obj(it).eegTrials.condition],'Color',[0.93 0.96 1]);
                h = subplot(3,1,1);
                scatter(obj(it).time(loc:end-loc),obj(it).mocapTrials.xy(loc:end-loc,2),'filled','CData',y(loc:end-loc));
                hold on;
                plot([1 1]*obj(it).time(locZero),get(gca,'YLim'),'k-.','LineWidth',2);
                title([obj(it).mocapTrials.derivativeLabel{derivative} '  Condition: ' obj(it).mocapTrials.condition])
                mx = max(abs(y(loc:end-loc)));
                set(h,'Clim',[-mx mx]);
                set(h,'Xlim',obj(it).time([loc end-loc]));
                colorbar
                subplot(3,1,2);
                if isempty(obj(it).ersp), obj.eegTimeFrequencyAnalysis;end
                imageLogData(obj(it).time(loc:end-loc),obj(it).frequency,obj(it).ersp(loc:end-loc,:),'ERSP (dB)');
                subplot(3,1,3);
                imageLogData(obj(it).time(loc:end-loc),obj(it).frequency,obj(it).itc(loc:end-loc,:),'ITC');
            end
        end
        %%
        function [mocapFactors,eegFactors,time] = npls(obj,varargin)
            if isempty(which('calcore')), addpath(genpath('/home/alejandro/Documents/MATLAB/Nway/'));end
            
            if isempty(varargin), varargin{1} = 3;end
            if length(varargin) < 2, varargin{2} = 1;end
            Fac = varargin{1};
            derivativeIndex = varargin{2};
            
            N = length(obj);
            if N < 2, return;end
            nd = length(obj(1).mocapTrials.derivativeLabel);
            tmpX = cell(N,1);
            tmpY = cell(N,1);
            mx = zeros(N,1);
            for it=1:N 
                [~,toCut] = min(abs(obj(it).boundary - (obj(it).time-min(obj(it).time))));
                tmpX{it} = obj(it).ersp(toCut:end-toCut,:);
                tmpY{it} = mean(obj(it).mocapTrials.data(toCut:end-toCut,:,:),3);
                mx(it) = size(tmpX{it},1);
            end
            [mx,loc] = max(mx);
            [~,toCut] = min(abs(obj(loc).boundary - (obj(loc).time-min(obj(loc).time))));
            time = obj(loc).time(toCut:end-toCut);
            [~,locZero] = min(abs(time));
            
            frequency = obj(1).frequency; %#ok
            nf = length(obj(1).frequency);
            condition = {obj.condition};
            xi = 1:mx;
            X = zeros(mx,nf,N);
            Y = zeros(mx,nd,N);
            x = linspace(1,mx,size(tmpX{loc},1));
            Y(:,:,loc) = interp1(x,tmpY{loc},xi,'linear');
            for it=setdiff(1:N,loc)
                x = linspace(1,mx,size(tmpX{it},1));
                X(:,:,it) = interp1(x,tmpX{it},xi,'linear');
                Y(:,:,it) = interp1(x,tmpY{it},xi,'linear');
            end
            
            %Xs = bsxfun(@rdivide,X,std(X,[],1));
            Xs = X;
            y = squeeze(Y(:,derivativeIndex,:));
            % ys = y;
            % ys = bsxfun(@rdivide,y,std(y,[],1));
            ys = zscore(y,[],2);

            [Xfactors,Yfactors,Core,B,ypred,ssx,ssy,reg] = npls(Xs,ys,Fac);
            
            for it=1:Fac
                figure('NumberTitle','off','Color',[0.93 0.96 1],'Name',['Factor ' num2str(it)]);
                subplot(221);
                imageLogData(time,frequency,Xfactors{1}(:,it)*Xfactors{2}(:,it)','EEG spectral factor') %#ok
                colorbar off
                h = subplot(223);
                plot(time,Yfactors{1}(:,it));
                xlabel('Time (sec)');
                title([obj(1).mocapTrials.derivativeLabel{derivativeIndex} ' factor time course']);
                hold on;plot([1 1]*time(locZero),get(gca,'YLim'),'k-.','LineWidth',2);
                grid on;
                set(h,'Xlim',time([1 end]));
                %                 subplot(225);
                %                 scatter(time,obj(loc).mocapTrials.xy(toCut:end-toCut,2),'filled','CData',Yfactors{1}(:,it));
                %                 xlabel('Time (sec)');
                %                 ylabel('Y (mm)');
                %                 title(['Mocap trace colored by ' obj(1).mocapTrials.derivativeLabel{derivativeIndex} ' factor']);
                %                 hold on;
                %                 plot([1 1]*time(locZero),get(gca,'YLim'),'k-.','LineWidth',2);
                %                 set(h,'Xlim',time([1 end]));
                
                h = subplot(222);
                plot(Xfactors{3}(:,it));hold on
                plot(Xfactors{3}(:,it),'ro');ylabel('Score');title('EEG condition factor');grid on;
                set(h,'XTickLabel',condition);
                h = subplot(224);
                plot(Yfactors{2}(:,it));hold on;
                plot(Yfactors{2}(:,it),'ro');ylabel('Score');title([obj(1).mocapTrials.derivativeLabel{derivativeIndex} ' condition factor']);grid on;
                set(h,'XTickLabel',condition);
                
                %                 subplot(326);
                %                 plot(Xfactors{3}(:,it),Yfactors{2}(:,it),'o');
                %                 xlabel('EEG condition factor')
                %                 ylabel([obj(1).mocapTrials.derivativeLabel{derivativeIndex} ' condition factor'])
                %                 [r,p] = corr(Xfactors{3}(:,it),Yfactors{2}(:,it));
                %                 title(['Correlation = ' num2str(r) ',   P-val = ' num2str(p)]);
            end
            
            mocapFactors = Yfactors;
            eegFactors = Xfactors;
        end
        %%
        function [Factors,corcondia] = parafac(obj,varargin)
            if isempty(which('calcore')), addpath(genpath('/home/alejandro/Documents/MATLAB/Nway/'));end
            if isempty(varargin), varargin{1} = 3;end
            if length(varargin) < 2
                varargin{2}(1) = 1;   % Convergence criterion
                varargin{2}(2) = 0;   % Initialization method: 0=>DTLD/GRAM, 1=>SVD, 2=>random, 10=>best-fitting models
                varargin{2}(3) = 0;   % Plotting
                varargin{2}(4) = 2;   % Scaling
                varargin{2}(5) = 10;
                varargin{2}(6) = 2500;
            end
            if length(varargin) < 3
                varargin{3} = [1 1 0];   % 0 => no constraint, 1 => orthogonality, 2 => nonnegativity, 3 => unimodality (and nonnegativitiy)
            end
            if length(varargin) < 4, varargin{4} = true;end
            
            fac = varargin{1};
            opt = varargin{2};
            const = varargin{3};
            plotFlag = varargin{4};
            
            N = length(obj);
            tmpX = cell(N,1);
            ni = zeros(N,1);
            for it=1:N 
                [~,toCut] = min(abs(obj(it).boundary - (obj(it).time-min(obj(it).time))));
                tmpX{it} = obj(it).ersp(toCut:end-toCut,:);
                ni(it) = size(tmpX{it},1);
            end
            n = round(median(ni));
            x = 1:length(obj(1).time(toCut:end-toCut));
            y = obj(1).time(toCut:end-toCut);
            xi = linspace(x(1),x(end),n);
            time = interp1(x,y,xi,'linear');
            frequency = obj(1).frequency; %#ok
            nf = length(obj(1).frequency);
            condition = {obj.condition};
            xi = 1:n;
            X = zeros(n,nf,N);
            for it=1:N
                x = linspace(1,n,ni(it));
                X(:,:,it) = interp1(x,tmpX{it},xi,'linear');
            end
        
            [Factors,~,~,corcondia]=parafac(X,fac,opt,const);
            
            if plotFlag
                figure('NumberTitle','off','Color',[0.93 0.96 1],'Name',['Corcondia = ' num2str(corcondia)]);
                c = 1:2:fac*2;
                r = 2:2:fac*2;
                for f=1:fac
                    subplot(2,fac,c(f));
                    imageLogData(time,frequency,Factors{1}(:,f)*Factors{2}(:,f)',['Spectral factor ' num2str(f)]) %#ok
                    colorbar off
                    h = subplot(2,fac,r(f));
                    plot(Factors{3}(:,f),'ro');
                    hold on;
                    plot(Factors{3}(:,f));
                    set(h,'XTickLabel',condition);
                    title(['Score of condition, factor ' num2str(f)])
                    grid on;
                end
            end
        end
        %%
        function regress(obj,derivative)
            y = squeeze(obj.mocapTrials.data(:,derivative,:))';
            
            coeff = obj.eegTrials.waveletTimeFrequencyAnalysis;
            coeff_dB = 10*log10(abs(coeff).^2+eps);
            X0 = mean(coeff_dB(obj.eegTrials.preStimulusMaxLatency(1):obj.eegTrials.preStimulusMaxLatency(2),:,:));
            X  = bsxfun(@minus,coeff_dB,X0+eps);
            dim = size(X);
            X = permute(X,[3 1 2]);
            X = reshape(X,[dim(3) prod(dim(1:2))]);
            
            [b,~,yhat] = ridgeGCV(y(:,end),X,speye(size(X,2)),100,1);
        end
        %%
        function eegTimeFrequencyAnalysis(obj,wname,fmin,fmax,numFreq,multCompCorrectionMethod,nboot,alpha)
            T = diff(obj(1).time([1 2]));
            if nargin < 2, wname = 'cmor1-1.5';end
            if nargin < 3, fmin = 1;end
            if nargin < 4, fmax = 1/T/2;end
            if nargin < 5, numFreq = 64;end
            if nargin < 6, multCompCorrectionMethod = 'none';end
            if nargin < 7, nboot = 1000;end
            if nargin < 8, alpha = 0.05;end
            
            N = length(obj);
            for it=1:N    
                [~,ersp,itc,frequency,time] = obj(it).eegTrials.waveletTimeFrequencyAnalysis(wname,fmin,fmax,numFreq,false,multCompCorrectionMethod,nboot,alpha); %#ok
                obj(it).frequency = frequency; %#ok
                I = ismember(obj(it).time,time);
                obj(it).ersp = zeros(length(obj(it).time),length(frequency));%#ok
                obj(it).ersp(I,:) = ersp; %#ok
                obj(it).itc = zeros(length(obj(it).time),length(frequency));%#ok
                obj(it).itc(I,:) = itc;%#ok
            end
        end
    end
end

%--
function imageLogData(time,frequency,data,strTitle)
if nargin < 4, strTitle = '';end

imagesc(time,log10(frequency),data');
hAxes = gca;
tick = get(hAxes,'Ytick');
tick = linspace(tick(1)/2,tick(end)+tick(1)/8,8);
fval = 10.^tick;
Nf = length(tick);
yLabel = cell(Nf,1);
fval(fval >= 10) = round(fval(fval >= 10));
for it=1:Nf, yLabel{it} = num2str(fval(it),3);end
mx = max(data(:));
if min(data(:)) < 0,
    mn = -mx;
else
    mn = 0; %min(data(:));
end
set(hAxes,'YDir','normal','Ytick',tick,'YTickLabel',yLabel,'CLim',[mn mx]);
[~,loc] = min(abs(time));
hold(hAxes,'on');plot([1 1]*time(loc),get(hAxes,'YLim'),'k-.','LineWidth',2);
xlabel('Time (sec)');
ylabel('Frequency (Hz)');
title(strTitle)
colorbar;
end
