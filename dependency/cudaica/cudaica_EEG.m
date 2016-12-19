function EEG = cudaica_EEG(EEG, verbose)
% This function is a minimalistic wrapper for computing the ICA decomposition
% of EEG data on a GPU using cudaica. If no GPU is available it tries to
% run binica.
% 
% Input: 
% EEG: EEGLAB's EEG structure
%
% EEG: same structure but with the ICA fields
% 
% For more information visit http://liaa.dc.uba.ar/?q=node/20
% See also: 
%     Raimondo, F., Kamienkowski, J.E., Sigman, M., and Slezak, D.F., 2012. CUDAICA: GPU Optimization of Infomax-ICA EEG Analysis.
%       Computational Intelligence and Neuroscience Volume 2012 (2012), Article ID 206972, 8 pages doi:10.1155/2012/206972
%       http://www.hindawi.com/journals/cin/2012/206972/
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Mar-2013
if nargin < 2, verbose = 'off';end
X = EEG.data(:,:);
if ~isa(X,'double'), X = double(X);end

r = rank(X);
if r < size(X,1)
    try
        disp('This EEG data is rank deficient. Here is what we will try to do.')
        disp('  First we will remove a few channels (starting from the bottom up) from the data until it ');
        disp('    becomes full-rank and then run ICA.')
        disp('  Second we will use the channel locations to extrapolate the decomposition to the previously');
        disp('     removed channels.')
        disp('  See cudaica_EEG_interp_scalpmaps.m for more information.')
        EEG = cudaica_EEG_interp_scalpmaps(EEG);
        return
    catch ME
        disp(ME.message)
        disp('Removing null subspace from tha data before running ICA using PCA.');
        [U,S,V] = svds(X,r);
        X = V';
        clear V;
        iU = diag(1./diag(S))*U';
        U  = U*S;
    end
else
    U  = 1;
    iU = 1;
end
try
    [wts,sph] = cudaica(X, 'verbose',verbose);
catch ME
    warning(ME.message)
    disp('CUDAICA has failed, trying binica...');
    [wts,sph] = binica(X);
end
% We use the SVD decomposition of elminate the null-space 
% from the data and then perform ICA on a full rank matrix.
% 
% The following transformations are used:
%        X = U*S*V'
%        U = U*S
%   inv(U) = inv(S)*U'
%        S = U* wts*sph * inv(U)*X
%        S = U* wts*sph * inv(U)*U*V'
%        S = U* wts*sph * V'
%   
% We define the new wts and sph as:
%   wts = U* * wts
%   sph = sph * inv(U) 

icawinv = pinv(wts*sph);
wts = U*wts*iU;
sph = U*sph*iU;
icawinv = U*icawinv*iU;

EEG.icawinv = icawinv;
EEG.icasphere = sph;
EEG.icaweights = wts;
EEG.icachansind = 1:EEG.nbchan;
EEG = eeg_checkset(EEG);
