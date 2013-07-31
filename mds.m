function [Y,D] = mds(X,method,p)
%% [Y,D] = mds(X,method,p)
% Multidimensional scaling
% Computes a similarity matrix based on Mahalanobis distance (default), 
% otherwise computes Minkowski distance of order p. Then it calls MATLAB's
% clasical mds function.
%
% Input: 
% X: m x n (m-time points x n-trials) or m x n x ntrials (m-time points x n-variables x trials)
% method: 'mahalanobis' (default) or 'minkowski'
% p=1: Manhattan, p=2: Euclidean (default), p=inf: Chebyshev
%
% Author: Alejandro Ojeda 11-May-2012, SCCN, INC, UCSD

if nargin < 2, method = 'mahalanobis';end
if isempty(method), method = 'minkowski';end
if nargin < 3, p = 2;end
    
isM = strcmp(method,'mahalanobis');
[m,n,ntrials] = size(X);
if ntrials == 1, ntrials = n;n = 1;end
S = 1;
if isM, S = std(X,[],length(size(X)));end
S = S(:);
X = reshape(X,[m*n ntrials]);
D = triu(ones(ntrials),1);
I = find(D);
clear D;
[indI,indJ] = ind2sub([ntrials ntrials],I);
disp('Computing similarity matrix...')
%try
%    Mu = X(:,indI)-X(:,indJ);
%    if isM, Mu = bsxfun(@rdivide,Mu,S(:));end
%    Z = power(sum(Mu.^p),1/p);
%catch %#ok 
    N = ntrials*(ntrials-1)/2;
    Z = zeros(N,1);
    % hwait = waitbar(0,'Out of memmory!!!, this alternative will take some time...');
    for it=1:length(indI)
        Mu = X(:,indI(it))-X(:,indJ(it));
        Mu = Mu./S;
        Z(it) = power(sum(Mu.^p),1/p);
        % waitbar(it/N,hwait);
        if ~mod(100*it/N,5), fprintf(' %i%%',100*it/N);end
    end
    fprintf('\n');
    %close(hwait);
%end
D = zeros(ntrials);
D(I) = Z;
D = D+D';
disp('Now running mdscale...');
try
    [Y,e] = mdscale(D,3); %#ok
catch ME
    disp(ME.message)
    disp('Working with the abs value.')
    [Y,e] = mdscale(abs(D),3); %#ok
end
disp('Done!')
