function EEG = cudaica_EEG(EEG)
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

data = EEG.data(:,:);
desc = whos('data');
if desc.bytes > 2^30
    disp('Downsampling...')
    sr = obj.samplingRate;
    while desc.bytes > 2^28
        dim = size(data,2);
        ts = timeseries(data,(0:dim-1)/sr);
        t = ts.Time(1:2:dim);
        ts = resample(ts,t);
        data = squeeze(ts.Data);
        desc = whos('data');
    end
    clear ts;
end
if ~isa(data,'double'), data = double(data);end
try
    r = rank(data);
    if r < size(data,1)
        disp('Removing null subspace from tha data before running ICA.');
        try 
            [U,S,V] = svd(gpuArray(data));
            U = gather(U(1:r,1:r));
            S = gather(S(1:r,:));
            V = gather(V);
        catch %#ok
            [U,S,V] = svds(data,r);
        end
        s = diag(S);
        data = V';
        clear V;
        US  = U*S;
        iUS = diag(1./s)*U';
    else
        US  = 1;
        iUS = 1;
    end
    try
        [wts,sph] = cudaica(data);
    catch ME
        warning(ME.message)
        disp('CUDAICA has failed, trying binica...');
        [wts,sph] = binica(data);
    end
    iWts = US*pinv(wts*sph);
    sph = sph*iUS;
    scaling = repmat(sqrt(mean(iWts.^2))', [1 size(wts,2)]);
    wts = wts.*scaling;
end

EEG.icawinv = pinv(wts*sph);
EEG.icasphere = sph;
EEG.icaweights = wts;
EEG.icachansind = 1:size(data,1);
EEG = eeg_checkset(EEG);
