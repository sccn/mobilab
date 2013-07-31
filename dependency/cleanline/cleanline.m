function [EEG, Sorig, Sclean, f, amps, freqs, g] = cleanline(varargin)

% Mandatory             Information
% --------------------------------------------------------------------------------------------------
% EEG                   EEGLAB data structure
% --------------------------------------------------------------------------------------------------
%
% Optional              Information
% --------------------------------------------------------------------------------------------------
% LineFrequencies:      Line noise frequencies to remove                                                                      
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 60  120                                                                                
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% ScanForLines:         Scan for line noise                                                                                   
%                       This will scan for the exact line frequency in a narrow range around the specified LineFrequencies    
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 1                                                                                      
%                       Input Data Type: boolean                                                                              
%                                                                                                                             
% LineAlpha:            p-value for detection of significant sinusoid                                                                        
%                       Input Range  : [0  1]                                                                                 
%                       Default value: 0.01                                                                                   
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% Bandwidth:            Bandwidth (Hz)                                                                                        
%                       This is the width of a spectral peak for a sinusoid at fixed frequency. As such, this defines the     
%                       multi-taper frequency resolution.                                                                     
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 1                                                                                      
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% SignalType:          Type of signal to clean                                                                               
%                       Cleaned ICA components will be backprojected to channels. If channels are cleaned, ICA activations    
%                       are reconstructed based on clean channels.                                                            
%                       Possible values: 'Components','Channels'                                                              
%                       Default value  : 'Components'                                                                         
%                       Input Data Type: string                                                                               
%                                                                                                                             
% ChanCompIndices:      IDs of Chans/Comps to clean                                                                           
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 1:152                                                                                  
%                       Input Data Type: any evaluable Matlab expression.                                                     
%                                                                                                                             
% SlidingWinLength:     Sliding window length (sec)                                                                           
%                       Default is the epoch length.                                                                          
%                       Input Range  : [0  4]                                                                                 
%                       Default value: 4                                                                                      
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% SlidingWinStep:       Sliding window step size (sec)                                                                        
%                       This determines the amount of overlap between sliding windows. Default is window length (no           
%                       overlap).                                                                                             
%                       Input Range  : [0  4]                                                                                 
%                       Default value: 4                                                                                      
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% SmoothingFactor:      Window overlap smoothing factor                                                                       
%                       A value of 1 means (nearly) linear smoothing between adjacent sliding windows. A value of Inf means   
%                       no smoothing. Intermediate values produce sigmoidal smoothing between adjacent windows.               
%                       Input Range  : [1  Inf]                                                                               
%                       Default value: 100                                                                                    
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% PaddingFactor:        FFT padding factor                                                                                    
%                       Signal will be zero-padded to the desired power of two greater than the sliding window length. The    
%                       formula is NFFT = 2^nextpow2(SlidingWinLen*(PadFactor+1)). e.g. For SlidingWinLen = 500, if PadFactor = -1, we    
%                       do not pad; if PadFactor = 0, we pad the FFT to 512 points, if PadFactor=1, we pad to 1024 points etc.                                                                                                  
%                       Input Range  : [-1  Inf]                                                                              
%                       Default value: 2                                                                                      
%                       Input Data Type: real number (double)                                                                 
%                                                                                                                             
% ComputeSpectralPower: Visualize Original and Cleaned Spectra                                                                
%                       Original and clean spectral power will be computed and visualized at end                              
%                       Input Range  : Unrestricted                                                                           
%                       Default value: true                                                                                      
%                       Input Data Type: boolean                                                                              
%                                                                                                                             
% NormalizeSpectrum:    Normalize log spectrum by detrending (not generally recommended)                                                                     
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 0                                                                                      
%                       Input Data Type: boolean                                                                              
%                                                                                                                             
% VerboseOutput:        Produce verbose output                                                                                
%                       Input Range  : [true false]                                                                           
%                       Default value: true                                                                                      
%                       Input Data Type: boolean                                                                
%                                                                                                                             
% PlotFigures:          Plot Individual Figures                                                                               
%                       This will generate figures of F-statistic, spectrum, etc for each channel/comp while processing       
%                       Input Range  : Unrestricted                                                                           
%                       Default value: 0                                                                                      
%                       Input Data Type: boolean  
%
% --------------------------------------------------------------------------------------------------
% Output                Information
% --------------------------------------------------------------------------------------------------
% EEG                   Cleaned EEG dataset
% Sorig                 Original multitaper spectrum for each component/channel
% Sclean                Cleaned multitaper spectrum for each component/channel
% f                     Frequencies at which spectrum is estimated in Sorig, Sclean
% amps                  Complex amplitudes of sinusoidal lines for each
%                       window (line time-series for window i can be
%                       reconstructed by creating a sinudoid with frequency f{i} and complex 
%                       amplitude amps{i})
% freqs                 Exact frequencies at which lines were removed for
%                       each window (cell array)
% g                     Parameter structure. Function call can be
%                       replicated exactly by calling >> cleanline(EEG,g);
%
% Usage Example:
% EEG = pop_cleanline(EEG, 'Bandwidth',2,'ChanCompIndices',[1:EEG.nbchan],                  ...
%                          'SignalType','Channels','ComputeSpectralPower',true,             ...
%                          'LineFrequencies',[60 120] ,'NormalizeSpectrum',false,           ...
%                          'LineAlpha',0.01,'PaddingFactor',2,'PlotFigures',false,          ...
%                          'ScanForLines',true,'SmoothingFactor',100,'VerboseOutput',1,    ...
%                          'SlidingWinLength',EEG.pnts/EEG.srate,'SlidingWinStep',EEG.pnts/EEG.srate);
%
% See Also:
% pop_cleanline()

% Author: Tim Mullen, SCCN/INC/UCSD Copyright (C) 2011
% Date:   Nov 20, 2011
%
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
EEG = varargin{1};
varargin(1) = [];
argnames = varargin(1:2:end);
for it=1:length(argnames)
    switch lower(argnames{it})
        case 'linefreqs'
            linefreqs = varargin{it*2};
        case 'scanforlines'
            scanforlines = varargin{it*2};
        case 'linealpha'
            alpha = varargin{it*2};
        case 'bandwidth'
            bandwidth = varargin{it*2};
        case 'sigtype'
            sigtype = varargin{it*2};
        case 'chanlist'
            chanlist = varargin{it*2};
        case 'winsize'
            winsize = varargin{it*2};
        case 'winstep'
            winstep = varargin{it*2};
        case 'tau'
            tau = varargin{it*2};
        case 'pad'
            pad = varargin{it*2};
        case 'computepower'
            computepower = varargin{it*2};
        case 'verb'
            verb =  varargin{it*2};
        case 'normspectrum'
            normSpectrum = varargin{it*2};
    end     
end
if ~exist('linefreqs','var'), linefreqs = [60 120];end
if ~exist('scanforlines','var'), scanforlines = true;end
if ~exist('linealpha','var'), alpha = 0.01;end
if ~exist('chanlist','var'), chanlist = 1:EEG.nbchan;end
if ~exist('winsize','var'), winsize = 2;end
if ~exist('winstep','var'), winstep = 1;end
if ~exist('tau','var'), tau = 100;end
if ~exist('pad','var'), pad = 2;end
if ~exist('computepower','var'), computepower = true;end
if ~exist('verb','var'), verb = false;end
if ~exist('normSpectrum','var'), normSpectrum = false;end
if ~exist('plotfigures','var'), plotfigures = false;end
if ~exist('sigtype','var'), sigtype = {'Channels'};end

% if ~isempty(EEG.icawinv);
%     defSigType = {'Components','Channels'};
% else
%     defSigType = {'Channels'};
% end

% g = arg_define([0 1], varargin, ...
%     arg_norep('EEG','mandatory'), ...
%     arg({'linefreqs','LineFrequencies'},[60 120],[],'Line noise frequencies to remove.'),...
%     arg({'scanforlines','ScanForLines'},true,[],'Scan for line noise. This will scan for the exact line frequency in a narrow range around the specified LineFrequencies'),...
%     arg({'p','LineAlpha','alpha'},0.01,[0 1],'p-value for detection of significant sinusoid'), ...
%     arg({'bandwidth','Bandwidth'},2,[],'Bandwidth (Hz). This is the width of a spectral peak for a sinusoid at fixed frequency. As such, this defines the multi-taper frequency resolution.'), ...
%     arg({'sigtype','SignalType','chantype'},defSigType{1},defSigType,'Type of signal to clean. Cleaned ICA components will be backprojected to channels. If channels are cleaned, ICA activations are reconstructed based on clean channels.'), ...
%     arg({'chanlist','ChanCompIndices','ChanComps'},sprintf('1:%d',EEG.nbchan),[],'Indices of Channels/Components to clean.','type','expression'),...
%     arg({'winsize','SlidingWinLength'},fastif(EEG.trials==1,4,EEG.pnts/EEG.srate),[0 EEG.pnts/EEG.srate],'Sliding window length (sec). Default for epoched data is the epoch length. Default for continuous data is 4 seconds'), ...
%     arg({'winstep','SlidingWinStep'},fastif(EEG.trials==1,1,EEG.pnts/EEG.srate),[0 EEG.pnts/EEG.srate],'Sliding window step size (sec). This determines the amount of overlap between sliding windows. Default for epoched data is window length (no overlap). Default for continuous data is 1 second.'), ...
%     arg({'tau','SmoothingFactor'},100,[1 Inf],'Window overlap smoothing factor. A value of 1 means (nearly) linear smoothing between adjacent sliding windows. A value of Inf means no smoothing. Intermediate values produce sigmoidal smoothing between adjacent windows.'), ...
%     arg({'pad','PaddingFactor'},2,[-1 Inf],'FFT padding factor. Signal will be zero-padded to the desired power of two greater than the sliding window length. The formula is NFFT = 2^nextpow2(SlidingWinLen*(PadFactor+1)). e.g. For N = 500, if PadFactor = -1, we do not pad; if PadFactor = 0, we pad the FFT to 512 points, if PadFactor=1, we pad to 1024 points etc.'), ...
%     arg({'computepower','ComputeSpectralPower'},true,[],'Visualize Original and Cleaned Spectra. Original and clean spectral power will be computed and visualized at end'),  ...
%     arg({'normSpectrum','NormalizeSpectrum'},false,[],'Normalize log spectrum by detrending. Not generally recommended.'), ...
%     arg({'verb','VerboseOutput','VerbosityLevel'},true,[],'Produce verbose output.'), ...
%     arg({'plotfigures','PlotFigures'},false,[],'Plot Individual Figures. This will generate figures of F-statistic, spectrum, etc for each channel/comp while processing') ...
%     );
% 
% arg_toworkspace(g);


% defaults
[Sorig, Sclean, f, amps, freqs] = deal([]);

hasica = ~isempty(EEG.icawinv);

% set up multi-taper parameters
hbw = bandwidth/2;   % half-bandwidth
params.tapers = [hbw, winsize, 1];
params.Fs = EEG.srate;
params.pad = pad;
movingwin = [winsize winstep];

% NOTE: params.tapers = [W, T, p] where:
% T==frequency range in Hz over which the spectrum is maximally concentrated 
%    on either side of a center frequency (half of the spectral bandwidth)
% W==time resolution (seconds)
% p is used for num_tapers = 2TW-p (usually p=1).

SlidingWinLen = movingwin(1)*params.Fs;
if params.pad>=0
    NFFT = 2^nextpow2(SlidingWinLen*(params.pad+1));
else
    NFFT = SlidingWinLen;
end

if isempty(EEG.data) && isempty(EEG.icaact)
    fprintf('Hey! Where''s your EEG data?\n');
    return;
end


if verb
    fprintf('\n\nWelcome to the CleanLine line noise removal toolbox!\n');
    fprintf('CleanLine is written by Tim Mullen (tim@sccn.ucsd.edu) and uses multi-taper routines modified from the Chronux toolbox (www.chronux.org)\n');
    fprintf('\nTsk Tsk, you''ve allowed your data to get very dirty!\n');
    fprintf('Let''s roll up our sleeves and do some cleaning!\n');
    fprintf('Today we''re going to be cleaning your %s\n',sigtype);
    if EEG.trials>1
        if winsize~=winstep
            fprintf('\n[!] Yikes! I noticed you have multiple trials, but you''ve selected overlapping windows.\n');
            fprintf('    This probably means one or more of your windows will span two trials, which can be bad news (discontinuities)!\n');
            resp = input('\n    Are you sure you want to continue? (''y'',''n''): ','s');
            if ~strcmpi(resp,'y')
                return;
            end
            
        end
        
        if winsize > EEG.pnts/EEG.srate
            fprintf('\n[!] Yikes! I noticed you have multiple trials, but your window length (%0.4g sec) is greater than the epoch length (%0.4g sec).\n',winsize,EEG.pnts/EEG.srate);
            fprintf('    This means each window will span multiple trials, which can be bad news!\n');
            fprintf('    Ideally, your windows should be less than or equal to the epoch length\n');
            resp = input('\n    Are you sure you want to continue? (''y'',''n''): ','s');
            if ~strcmpi(resp,'y')
                return;
            end
        end
        
        if winsize~=winstep || winsize > EEG.pnts/EEG.srate
            fprintf('\nFine, have it your way, but if results are sub-optimal try selecting window length and step size so your windows don''t span multiple trials.\n\n');
            pause(2);
        end
    end
    
    ndiff = rem(EEG.pnts,(winsize*EEG.srate));
    if ndiff>0
        fprintf('\n[!] Please note that because the selected window length does not divide the data length, \n');
        fprintf('    %0.4g seconds of data at the end of the record will not be cleaned.\n\n',ndiff/EEG.srate);
    end
        
    fprintf('Multi-taper parameters follow:\n');
    fprintf('\tTime-bandwidth product:\t %0.4g\n',hbw*winsize);
    fprintf('\tNumber of tapers:\t %0.4g\n',2*hbw*winsize-1);
    fprintf('\tNumber of FFT points:\t %d\n',NFFT);
    if ~isempty(linefreqs)
        fprintf('I''m going try to remove lines at these frequencies: [%s] Hz\n',strtrim(num2str(linefreqs)));
        if scanforlines
            fprintf('I''m going to scan the range +/-%0.4g Hz around each of the above frequencies for the exact line frequency.\n',params.tapers(1));
            fprintf('I''ll do this by selecting the frequency that maximizes Thompson''s F-statistic above a threshold of p=%0.4g.\n',alpha);
        end
    else
        fprintf('You didn''t specify any lines (Hz) to remove, so I''ll try to find them using Thompson''s F-statistic.\n');
        fprintf('I''ll use a p-value threshold of %0.4g.\n',alpha)
    end
    fprintf('\nOK, now stand back and let The Maid show you how it''s done!\n\n');
end

EEGLAB_backcolor = getbackcolor;

if plotfigures
    % plot the overlap smoothing function
    overlap = winsize-winstep;
    toverlap = -overlap/2:(1/EEG.srate):overlap/2;

    % specify the smoothing function
    foverlap = 1-1./(1+exp(-tau.*toverlap/overlap));

    % define some colours
    yellow  = [255, 255, 25]/255;
    red     = [255 0 0]/255;

    % plot the figure
    figure('color',EEGLAB_backcolor);
    axis([-winsize+overlap/2 winsize-overlap/2 0 1]); set(gca,'ColorOrder',[0 0 0; 0.7 0 0.8; 0 0 1],'fontsize',11);
    hold on
    h(1)=hlp_vrect([-winsize+overlap/2 -overlap/2], 'yscale',[0 1],'patchProperties',{'FaceColor',yellow,        'FaceAlpha',1,'EdgeColor','none','EdgeAlpha',0.5}); 
    h(2)=hlp_vrect([overlap/2 winsize-overlap/2],   'yscale',[0 1],'patchProperties',{'FaceColor',red,           'FaceAlpha',1,'EdgeColor','none','EdgeAlpha',0.5});
    h(3)=hlp_vrect([-overlap/2 overlap/2],          'yscale',[0 1],'patchProperties',{'FaceColor',(yellow+red)/2,'FaceAlpha',1,'EdgeColor','none','EdgeAlpha',0.5});
    plot(toverlap,foverlap,'linewidth',2);
    plot(toverlap,1-foverlap,'linewidth',1,'linestyle','--');
    hold off;
    xlabel('Time (sec)'); ylabel('Smoothing weight'); 
    title({'Plot of window overlap smoothing function vs. time',['Smoothing factor is \tau = ' num2str(tau)]});
    legend(h,{'Window 1','Window 2','Overlap'});
end

if hasica && isempty(EEG.icaact)
    EEG = eeg_checkset(EEG,'ica');
end

k=0;

%--
try
    mobilab = evalin('base','mobilab');
    mobilab.initStatusbar(1,length(chanlist),'Cleaning...');
catch %#ok
    mobilab = [];
end

%--

for ch=chanlist
    
    
    if ~isempty(mobilab)
        mobilab.statusbar(ch);
    else
        if verb, fprintf('Cleaning %s %d...\n',fastif(strcmpi(sigtype,'Components'),'IC','Chan'),ch);end
    end 
    
    
    % extract data as [chans x frames*trials]
    if strcmpi(sigtype,'components')
        data = squeeze(EEG.icaact(ch,:));
    else
        data = squeeze(EEG.data(ch,:));
    end
    
    
    if plotfigures
        % estimate the sinusoidal lines
        [Fval sig f] = ftestmovingwinc(data,movingwin,params,0.01);
        
        % plot the F-statistics
        [F T] = meshgrid(f,1:size(Fval,1));
        figure('color',EEGLAB_backcolor);
        subplot(311);
        surf(F,T,Fval); shading interp; caxis([0 prctile(Fval(:),99)]); axis tight
        sigplane = ones(size(Fval))*sig;
        hold on; surf(F,T,sigplane,'FaceColor','b','FaceAlpha',0.5);
        xlabel('Frequency'); ylabel('Window'); zlabel('F-value');
        title({[sprintf('%s %d: ',fastif(strcmpi(sigtype,'components'),'IC ','Chan '), ch) 'Thompson F-statistic for sinusoid'],sprintf('Black plane is p<%0.4g thresh',alpha)});
        shadowplot x
        shadowplot y
        axcopy(gca);
        
        subplot(312);
        plot(F,mean(Fval,1),'k');
        axis tight
        hold on
        plot(get(gca,'xlim'),[sig sig],'r:','linewidth',2);
        xlabel('Frequency');
        ylabel('Thompson F-stat');
        title('F-statistic averaged over windows');
        legend('F-val',sprintf('p=%0.4g',alpha));
        hold off
        axcopy(gca);
    end
  
    
    if plotfigures
        subplot(313)
    end
    
    % DO THE MAGIC!
    [datac,datafit,amps,freqs]=rmlinesmovingwinc(data,movingwin,tau,params,alpha,fastif(plotfigures,'y','n'),linefreqs,fastif(scanforlines,params.tapers(1),[]));   
    
    % append to clean dataset any remaining samples that were not cleaned 
    % due to sliding window and step size not dividing the data length
    ndiff = length(data)-length(datac);
    if ndiff>0
        datac(end:end+ndiff) = data(end-ndiff:end);
    end
        
    if plotfigures
        axis tight
        legend('original','cleaned');
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('Power spectrum for %s %d',fastif(strcmpi(sigtype,'components'),'IC','Chan'),ch));
        axcopy(gca);
    end
    
    
    if computepower
        k = k+1;
        if verb, fprintf('Computing spectral power...\n'); end
        
        [Sorig(k,:)  f] = mtspectrumsegc(data,movingwin(1),params);
        [Sclean(k,:) f] = mtspectrumsegc(datac,movingwin(1),params);
     
        if verb && ~isempty(linefreqs)
            fprintf('Average noise reduction: ');
            for fk=1:length(linefreqs)
                [dummy fidx] = min(abs(f-linefreqs(fk)));
                fprintf('%0.4g Hz: %0.4g dB %s ',f(fidx),10*log10(Sorig(k,fidx))-10*log10(Sclean(k,fidx)),fastif(fk<length(linefreqs),'|',''));
            end
            fprintf('\n');
        end
            
        if ch==chanlist(1)
            % First run, so allocate memory for remaining spectra in 
            % Nchans x Nfreqs spectral matrix
            Sorig = cat(1,Sorig,zeros(length(chanlist)-1,length(f)));
            Sclean = cat(1,Sclean,zeros(length(chanlist)-1,length(f)));
        end
    end
    
    
    
    if strcmpi(sigtype,'components')
        EEG.icaact(ch,:) = datac';
    else
        EEG.data(ch,:) = datac';
    end
    
end

if computepower
    
    if verb, fprintf('Converting spectra to dB...\n'); end
    
    % convert to log spectrum
    Sorig  = 10*log10(Sorig);
    Sclean = 10*log10(Sclean);
    
    
    if normSpectrum
        if verb, fprintf('Normalizing log spectra...\n'); end
        
        % normalize spectrum by standarization
        %         Sorig = (Sorig-repmat(mean(Sorig,2),1,size(Sorig,2)))./repmat(std(Sorig,[],2),1,size(Sorig,2));
        %         Sclean = (Sclean-repmat(mean(Sclean,2),1,size(Sclean,2)))./repmat(std(Sclean,[],2),1,size(Sclean,2));
        
        % normalize the spectrum by detrending
        Sorig = detrend(Sorig')';
        Sclean = detrend(Sclean')';
    end
    
end

if strcmpi(sigtype,'components')
    if verb, fprintf('Backprojecting cleaned components to channels...\n'); end
    try
        EEG.data = EEG.icawinv*EEG.icaact(1:end,:);
    catch e
        % low memory, so back-project channels one by one
        EEG.data = zeros(size(EEG.icaact));
        for k=1:size(EEG.icaact,1)
            EEG.data(k,:) = EEG.icawinv(:,k)*EEG.icaact(k,:);
        end
    end
    EEG.data = reshape(EEG.data,EEG.nbchan,EEG.pnts*EEG.trials);
elseif hasica
    if verb, fprintf('Recomputing component activations from cleaned channel data...\n'); end
    EEG.icaact = [];
    EEG = eeg_checkset(EEG,'ica');
end



function BACKCOLOR = getbackcolor

BACKCOLOR = 'w';

try, icadefs; catch, end;



