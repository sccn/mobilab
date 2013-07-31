function [outdata,outstate_ord0,outstate_ord1,outstate_ord2,outstate_buffer] = asr_process(data,srate,window_len,processing_delay,stddev_cutoff,block_size,max_dimensions,max_memory,covariance_matrix,mixing_matrix,filter_b,filter_a,state_ord0,state_ord1,state_ord2,state_buffer)
% Clean a chunk of EEG data.
% [Data,StateOrd0,StateOrd1,StateOrd2,StateBuffer] = asr_process(Data,SamplingRate,WindowLength,LookAhead,StandardDevCutoff,BlockSize,MaxDimensions,MaxMemory,CovarianceMatrix,MixingMatrix,ShapingFilter,StateOrd0,StateOrd1,StateOrd2,StateBuffer)
%
% In:
%   --- data-specific parameters ---
%
%   Data : Chunk of data to process [#channels x #samples]. This is a chunk of data, assumed to be
%          a continuation of the data that was passed in during the last call to asr_process.
%          The data should be high-pass filtered (the same way as for asr_calibrate) and of type double.
%          Also, if bad channels were removed before calling asr_calibrate, the same channels should 
%          be removed from data.
%   
%   SamplingRate : sampling rate of the data in Hz (e.g., 250.0)
%
%
%   --- processing parameters ---
%
%   WindowLength : Length of the statistcs window, in seconds (e.g., 0.5). This should not be much
%                  longer than the time scale over which artifacts persist, but the number of samples 
%                  in the window should not be smaller than 1.5x the number of channels.
%
%   LookAhead : Amount of look-ahead that the algorithm should use. Since the processing is causal,
%               the output signal will be delayed by this amount. This value is in seconds and should
%               be between 0 (no lookahead) and WindowLength/2 (optimal lookahead). The recommended
%               value is WindowLength/2.  This should be the same as what was used in the call to 
%               asr_calibrate.
%
%   StandardDevCutoff: Standard deviation cutoff for rejection. Data portions whose variance is larger
%                      than this threshold relative to the calibration data is considered missing data
%                      and will be removed. The recommended value is 3; the most aggressive value 
%                      that can be used without losing too much EEG is 2.5. A very conservative value 
%                      would be 4.
%
%   BlockSize       : The statistics will be updated every this many samples. The larger this is, 
%                     the faster the algorithm will be. The value must not be larger than 
%                     WindowLength*SamplingRate. The minimum value is 1 (update for every sample)
%                     while the recommended value is 32. 
%
%   MaxDimensions   : Maximum dimensionality of artifacts to remove. Up to this many dimensions (or 
%                     up to this fraction of dimensions) can be removed for a given data segment.
%                     The recommended value is 0.5. If the algorithm needs to tolerate extreme 
%                     artifacts you might increase this to 0.75 (the maximum fraction is 1.0).
%
%   MaxMemory : The maximum amount of memory used by the algorithm when processing a long chunk with
%               many channels, in MB. The recommended value is 256.
%
%   FilterB, FilterA : Coefficients of an IIR filter that is used to shape the spectrum of the signal
%                      when calculating artifact statistics. Must be the same as in asr_calibrate.
%
%   --- calibration parameters (this comes from asr_calibrate) ---
%
%   CovarianceMatrix : the covariance matrix C of the data
%
%   MixingMatrix : the mixing matrix M of the data
%
%   --- filter state (this is passed back in from the last call to asr_process or asr_calibrate) ---
%
%   StateOrd0 : zeroth-order initial filter state
%
%   StateOrd1 : first-order filter state
%
%   StateOrd2 : second-order filter state
%
%   StateBuffer : buffer state
%
%
% Out:
%   StateOrd0 : zeroth-order initial filter state
%
%   StateOrd1 : first-order updated filter state
%
%   StateOrd2 : second-order updated filter state
%
%   StateBuffer : updated buffer state
%
% Example:
%     %% load raw calibration data (EEGLAB function)
%     EEG = pop_loadset('mycalibration.set');
% 
%     % design a highpass filter that we can use offline and online (FIR, 0.5Hz-1Hz transition band, minimum phase)
%     params = firpmord([0.5 1], [0 1], [0.001 0.01], EEG.srate, 'cell');
%     kernel = firpm(max(3,params{1}),params{2:end});
%     kernel = kernel+randn(1,length(kernel))*0.00001;
%     [dummy,kernel] = rceps(kernel);
%     kernel = kernel/abs(max(freqz(kernel)));
% 
%     % filter the calibration data
%     [EEG.data,state] = filter(kernel,1,double(EEG.data),[],2);
% 
%     % calibrate on the calibration data
%     [b,a] = yulewalk(6, [2*[0 2 3 13 14]/EEG.srate 1], [1 0.75 0.3 0.3 1 1]);
%     [cov_mat,mix_mat,shaper,state_ord0,state_ord1,state_ord2,state_buffer] = asr_calibrate(EEG.data, EEG.srate, 0.5, 0.25, b, a);
% 
%     %% do simulated online processing
%     EEG = pop_loadset('mydata.set');
%     scale = 75; duration=5; rawdata = []; fltdata = [];
%     figure('Position',[0 0 1000,500],'Tag','FigVis');
%     t0 = tic(); last_pos = 0;
%     while ~isempty(findobj('Tag','FigVis'))
%
%         % 1. get a new chunk from the simulated device
%         pos = round(1+(toc(t0)+duration)*EEG.srate);
%         rawchunk = double(EEG.data(:,(last_pos+1):pos)); 
%         last_pos = pos;
%
%         % 2. apply a high-pass filter
%         [rawchunk,state] = filter(kernel,1,rawchunk,state,2);
%
%         % 3. apply the artifact removal
%         [fltchunk,state_ord0,state_ord1,state_ord2,state_buffer] = asr_process(rawchunk, EEG.srate, 0.5, 0.25, 32, 32, 0665, 256, cov_mat, mix_mat, shaper, state_ord0, state_ord1, state_ord2, state_buffer);
% 
%         % display the raw and filtered data
%         rawdata = [rawdata rawchunk]; rawdata(:,1:(size(rawdata,2)-EEG.srate*duration)) = [];
%         fltdata = [fltdata fltchunk]; fltdata(:,1:(size(fltdata,2)-EEG.srate*duration)) = [];
%         try
%             for k=1:length(p{1})
%                 set(p{1}(k),'Ydata',rawdata(k,:)+k*scale); end
%             for k=1:length(p{2})
%                 set(p{2}(k),'Ydata',fltdata(k,:)+k*scale); end
%         catch
%             p{1}=plot(EEG.srate*0.25 + (1:size(rawdata,2)),bsxfun(@plus,(1:EEG.nbchan)*scale,rawdata'),'r'); hold on; p{2} = plot((1:size(rawdata,2)),bsxfun(@plus,(1:EEG.nbchan)*scale,fltdata'),'b'); hold off;
%         end
%         axis([EEG.srate*0.25,size(rawdata,2),0,EEG.nbchan*scale+scale]);
%         drawnow;
%     end
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-08-31

coeff = [0.1314,-0.2877,-0.0104];    % coefficients that model the degrees of freedom in the EEG covariance chi^2 distribution
                                     % depending on the window length over which the covariances are calculated (these numbers 
                                     % come from a fit to clean EEG)
                                    
[C,S] = size(data);                  % size of the data
N = round(window_len * srate);       % number of time points in the sliding window
P = round(processing_delay * srate); % number of time points of look-ahead

% get rid of NaN's and Inf's
data(~isfinite(data(:))) = 0;

% prepend the state buffer
if P > 0
    data = [state_buffer data]; end

% the recovery mask protects lower eigenvectors from being reconstructed
if max_dimensions < 1
    max_dimensions = round(C*max_dimensions); end
recovery_mask = (1:C)' < (C-max_dimensions);

% split up the total sample range into k chunks that will fit in memory
numsplits = ceil((C^2*S*8*5) / (max_memory*1024*1024));

eye_mat = eye(size(data,1));
for i=0:numsplits-1
    range = 1+floor(i*S/numsplits) : min(S,floor((i+1)*S/numsplits));
    if ~isempty(range)
        % get spectrally shaped data X for statistics computation (range shifted into the future by P)
        [X,state_ord0] = filter(filter_b,filter_a,double(data(:,range + P)),state_ord0,2);
        % ... and compute running mean E[X]
        [Xmean,state_ord1] = moving_average(N,X,state_ord1);
        % get unfolded cross-terms tensor X*X'
        [m,n] = size(X); X2 = reshape(bsxfun(@times,reshape(X,1,m,n),reshape(X,m,1,n)),m*m,n);
        % ... and running mean of that E[X*X']
        [X2mean,state_ord2] = moving_average(N,X2,state_ord2);
        % compute running covariance E[X*X'] - E[X]*E[X]'
        Xcov = X2mean - reshape(bsxfun(@times,reshape(Xmean,1,m,n),reshape(Xmean,m,1,n)),m*m,n);
    
        last_n = 0;
        last_matrix = eye_mat;
        last_flagged = false;
        % for each position in the statistics buffer for which we update the re-projection matrix...
        for j = 1:block_size:(size(Xcov,2)+block_size-1)
            n = min(j,size(Xcov,2));
            % calculate an eigenvector decomposition
            [V,D] = eig(reshape(Xcov(:,n),m,m));
            % sort by ascending order
            [D,order] = sort(reshape(diag(D),C,1)); V = V(:,order);
            % get the median of the segment amplitudes for each eigenvector in V
            amp_median = sqrt(reshape(diag(V'*covariance_matrix*V),C,1));
            % and estimate their median absolute deviation based on the median (depends on the # of
            % degrees of freedom of the chi-squared distribution underlying the amplitudes)
            amp_mad = amp_median .* (coeff(1)*window_len^coeff(2) + coeff(3));
            % retain only those spatial components whose amplitude is within k standard deviations
            mask = (sqrt(D)-amp_median < amp_mad * stddev_cutoff) | recovery_mask;
            if ~all(mask)
                % we have some rejections; first rotate the mixing matrix into the eigenspace
                mixing_eig = (V'*mixing_matrix);
                % generate a re-construction matrix with this subspace interpolated from the rest...
                mixing_eig_trunc = mixing_eig; mixing_eig_trunc(~mask,:) = 0;
                % 3. rotate back <-- 2. apply reconstruction in this space based on the mixing matrix when transformed into this space <-- 1. rotate data into eigenspace
                reconstruct = real(V * (mixing_eig*pinv(mixing_eig_trunc)) * V');
                flagged = true;
            else
                % nothing to do
                reconstruct = eye_mat;
                flagged = false;
            end
            if flagged || last_flagged
                % apply the reconstruction to the samples range since the last update position
                update_range = (last_n+1):n;
                % ... using the following Hann window blend weights
                wts = (update_range - last_n) / (n - last_n);
                wts = 0.5*(1-cos(pi*wts));
                for k=1:length(update_range)
                    data(:,range(1,last_n+k)) = ((1-wts(k))*last_matrix + wts(k)*reconstruct) * data(:,range(1,last_n+k)); end
            end
            last_matrix = reconstruct;
            last_flagged = flagged;
            last_n = n;
        end
    end
end

if P > 0
    % the end of the signal gets appended to the buffer
    state_buffer = [state_buffer data(:,(end-P+1):end)];
    % and the buffer is trimmed to length P
    state_buffer = state_buffer(:,(end-P+1):end);
    % ... and will not be returned (yet)
    data = data(:,1:(end-P));
end

outdata = data;
outstate_ord0 = state_ord0;
outstate_ord1 = state_ord1;
outstate_ord2 = state_ord2;
outstate_buffer = state_buffer;



function [X,Zf] = moving_average(N,X,Zi)
% Run a moving-average filter along the second dimension of the data.
% [X,Zf] = moving_average(N,X,Zi)
%
% In:
%   N : filter length in samples
%   X : data matrix [#Channels x #Samples]
%   Zi : initial filter conditions (default: [])
%
% Out:
%   X : the filtered data
%   Zf : final filter conditions
%
%                           Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                           2012-01-10

if nargin <= 2
    Zi = []; end

if isempty(Zi)
    % zero initial state
    Zi = zeros(size(X,1),N);
elseif ~isequal(size(Zi),[size(X,1),N])
    error('These initial conditions do not have the correct format.');
end

% pre-pend initial state & get dimensions
Y = [Zi X]; M = size(Y,2);
% get alternating index vector (for additions & subtractions)
I = [1:M-N; 1+N:M];
% get sign vector (also alternating, and includes the scaling)
S = [-ones(1,M-N); ones(1,M-N)]/N;
% run moving average
X = cumsum(bsxfun(@times,Y(:,I(:)),S(:)'),2);
% read out result
X = X(:,2:2:end);

% construct final state
if nargout > 1
    Zf = [-(X(:,end)*N-Y(:,end-N+1)) Y(:,end-N+2:end)]; end
