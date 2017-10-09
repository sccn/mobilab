function EEG = cudaica_EEG_interp_scalpmaps(EEG, channels2remove)
% Perform ICA on rank defficient EEG data. First, some channels are removed
% to perform ICA on the full-rank data. Second, a linear interpolator is
% built to represent the independent scalp maps in the original space.
%
% Input: 
% EEG: EEGLAB's EEG structure
% channels2remove: array of channels to remove in order to make the data
%                  full-rank, if left unset we will start removing channels 
%                  starting from the bottom of the montage up until the 
%                  data becomes full-rank.
%
% EEG: same structure but with the ICA fields
% 
% For more information on the CUDAICA program visit http://liaa.dc.uba.ar/?q=node/20
% See also: 
%     Raimondo, F., Kamienkowski, J.E., Sigman, M., and Slezak, D.F., 2012. CUDAICA: GPU Optimization of Infomax-ICA EEG Analysis.
%       Computational Intelligence and Neuroscience Volume 2012 (2012), Article ID 206972, 8 pages doi:10.1155/2012/206972
%       http://www.hindawi.com/journals/cin/2012/206972/
%
% Author: Alejandro Ojeda, SCCN, INC, UCSD, Mar-2013

X = EEG.data(:,:);
if ~isa(X,'double'), X = double(X);end
[n,p] = size(X);
r = rank(X);

if r < min([n,p])
    if nargin < 2, channels2remove = [];end
    elec = [[EEG.chanlocs.X]' [EEG.chanlocs.Y]' [EEG.chanlocs.Z]'];
    loc_i = channels2remove;
    loc = setdiff(1:n, loc_i);
    [loc_i,r] = get_full_rank(X, elec, loc, loc_i, n);
    loc = setdiff(1:n, loc_i);
    
    interpolator = geometricTools.linearInterpolator(elec(loc,:),elec(loc_i,:));
    [wts,sph] = cudaica(X(loc,:), 'verbose','off');
    icawinv = pinv(wts*sph);
    
    W = zeros(n);   W(loc,loc) = wts;
    S = zeros(n);   S(loc,loc) = sph;
    iW = zeros(n); iW(loc,loc) = icawinv;

    W(loc_i,loc)  = interpolator*wts;
    W(loc,loc_i)  = wts*interpolator';
    
    S(loc_i,loc)  = interpolator*sph;
    S(loc,loc_i)  = sph*interpolator';
    
    iW(loc_i,loc) = interpolator*icawinv;
    iW(loc,loc_i) = icawinv*interpolator';
    
    icawinv = iW;
    wts = W;
    sph = S;
else
    [wts,sph] = cudaica(X, 'verbose','off');
    icawinv = pinv(wts*sph);
end

EEG.icawinv = icawinv;
EEG.icasphere = sph;
EEG.icaweights = wts;
EEG.icachansind = 1:EEG.nbchan;
EEG = eeg_checkset(EEG);
end
function [loc_i, r] = get_full_rank(X, elec, loc, loc_i, n)
r = rank(X(loc,:));
if length(loc_i)+r < n
    [~, locs] = sort(elec(loc,3));
    loc_i = [loc_i loc(locs(1:n-(length(loc_i)+r)))];
end
loc_i = sort(loc_i);
end