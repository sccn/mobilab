function EEG = cudaica_lowrank(EEG, verbose)
% Perform ICA on rank defficient EEG data using PCA reduction.
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
[n,m] = size(X);
r = rank(X);
if r < n
    disp('Removing null subspace from tha data before running ICA using PCA.');
    Cx = X*X'/m;
    [U,S] = eig(Cx);
    [~,sorting] = sort(diag(S),'descend');
    Xr = U(:,sorting(1:end-(n-r)))'*X;
    try
        [wts,sph] = cudaica(Xr);
    catch
        disp('CUDAICA has failed, trying binica...');
        [wts,sph] = binica(Xr);
    end
    wts = wts*sph*U(:,sorting(1:end-(n-r)))';
    sph = eye(n);
else
    try
        [wts,sph] = cudaica(X, 'verbose',verbose);
    catch ME
        warning(ME.message)
        disp('CUDAICA has failed, trying binica...');
        [wts,sph] = binica(X);
    end    
end
icawinv = pinv(wts*sph);
EEG.icawinv = icawinv;
EEG.icasphere = sph;
EEG.icaweights = wts;
EEG.icachansind = 1:EEG.nbchan;
EEG = eeg_checkset(EEG);

%% Clean-up
rmfiles =  dir('cudaica.*');
for k=1:length(rmfiles), delete(rmfiles(k).name);end
d = fileparts(which('cudaica_lowrank'));
rmfiles = dir([d filesep '*.sc']);
for k=1:length(rmfiles), delete(fullfile(d,rmfiles(k).name));end
rmfiles = dir([fileparts(which('cudaica_lowrank')) filesep '*.sph']);
for k=1:length(rmfiles), delete(fullfile(d,rmfiles(k).name));end
end
