function [signal,decomposition] = clean_nonstationary(signal,removedims,blocklen,opmode)
% Remoe a maximally non-stationary subspace of the signal
% [Signal,Projection] = clean_nonstationary(Signal, RemoveDimensions,BlockLength,Operation)
%
% This is an implementation of the Analytical Stationary Subspace Analysis algorithm [1]. Typically,
% it is applied to continuous data with a block length that is large enough to average over changes
% in the experimental conditions (if presented in a randomized bock design), e.g. 30 seconds.
%
% The data usually needs to be high-pass filtered prior to use, since otherwise drifts will dominate
% the results. The filter may also be applied to epoched data, in which case the epochs are taken as
% the blocks.
%
% In:
%   Signal           : continuous data set to be filtered
%
%   BadDimensions    : number of maximally non-stationary dimensions to flag as bad (if negative,
%                      this is the number of maximally stationary dimensions to *keep*) (default: 10)
%
%   BlockLength      : length of the signal blocks across which non-stationarity should be assessed,
%                      in seconds; may also be an [Nx2] array of intervals that should be taken as
%                      blocks (default: 30)
%
%   Operation        : Operation to perform; can be one of the following:
%                      * 'keep_stationary': project the signal onto the stationary components
%                      * 'keep_nonstationary': project the signal onto the non-stationary components
%                      * 'separate': order the signal by increasing non-stationarity of its components
%                      * 'backproject_stationary': back-project the stationary components onto the channels
%                      * 'backproject_nonstationary': back-project the non-stationary components onto the channels
%
% Out:
%   Signal           :  filtered EEGLAB data set
%
%   Projection       :  the projection matrix that was applied to obtain the cleaned signal
%
% References:
%  [1] S. Hara, Y. Kawahara, P. von Buenau, "Stationary Subspace Analysis as a Generalized Eigenvalue Problem",
%      Lecture Notes in Computer Science, 2010, Vol. 6443/2010, 422-429.
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2011-07-28

if ~exist('removedims','var') || isempty(removedims)
    removedims = 10; end
if ~exist('blocklen','var') || isempty(blocklen)
    blocklen = 30; end
if ~exist('opmode','var') || isempty(opmode)
    opmode = 'backproject_stationary'; end

% create the blocks X{i}
if ndims(signal.data) == 3
    % signal is already epoched; ignore blocklen
    X = squeeze(mat2cell(signal.data,signal.nbchan,signal.pnts,ones(1,signal.trials)));
elseif isscalar(blocklen)
    % generate blocklen-sized blocks
    l = blocklen*signal.srate;
    X = mat2cell(signal.data,signal.nbchan,[l*ones(1,floor(signal.pnts/l)), mod(signal.pnts,l)]);
else
    % extract intervals for each row in blocklen...
    for i=1:size(blocklen,1)
        X{i} = signal.data(:,round(blocklen(i,1)*signal.srate):round(blocklen(i,2)*signal.srate)); end
end

N = length(X);
% compute mean and covariance for each block
for i=1:N
    mu{i} = mean(X{i}');
    sig{i} = cov(X{i}');
end

% and compute joint mean and covariance
Mu = mean(vertcat(mu{:}));
Sig = mean(cat(3,sig{:}),3);
invSig = inv(Sig);

% compute the matrix S (Eq. 9)
S = zeros(size(Sig));
for i=1:N
    S = S + mu{i}*mu{i}' + (1/2) * sig{i} * invSig * sig{i}; end
S = S/N - Mu*Mu' - 1/2*Sig;

% solve the generalized eigenvalue problem and sort results
[phi,lambdas] = eig(S,Sig);                          % S*phi = Sig*phi*lambda;
[lambdas,idx] = sort(diag(lambdas),'ascend'); %#ok<ASGLU>
phi = phi(:,idx);

% split into stationary and non-stationary subspaces
if removedims < 0
    retain = -removedims;
else
    retain = signal.nbchan-removedims;
end
stationary = phi(:,1:retain)';
nonstationary = phi(:,retain+1:end)';

switch opmode
    case 'keep_stationary'
        decomposition = stationary;
    case 'keep_nonstationary'
        decomposition = nonstationary;
    case 'separate'
        decomposition = phi';
    case 'backproject_stationary'
        decomposition = stationary' * stationary;
    case 'backproject_nonstationary'
        decomposition = nonstationary' * nonstationary;
    otherwise
        error('Unsupported operation requested.');
end

% project data
[C,S,T] = size(signal.data); %#ok<*NODEF>
signal.data = reshape(decomposition*reshape(signal.data,C,[]),[],S,T);
signal.nbchan = size(signal.data,1);

% rewrite chanlocs if necessary
if ~any(strcmp(opmode,{'backproject_stationary','backproject_nonstationary'}))
    signal.chanlocs = struct('labels',cellfun(@(x)sprintf('StationaryComponent%.0f',x),num2cell(1:size(decomposition,1),1),'UniformOutput',false)); end
