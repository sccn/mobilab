function EEG = cudaica_EEG(EEG, verbose)
% This function is a wrapper for CUDAICA for computing the ICA decomposition
% of EEG data using GPU. If no GPU is available it runs binica. We take
% care of rank deficient data by using cudaica_lowrank.
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
warning('cudaica_EEG is been deprecated, next use cudaica_lowrank instead.');
EEG = cudaica_lowrank(EEG, verbose);