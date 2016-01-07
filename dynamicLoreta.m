function [J,beta,alpha,T,history] = dynamicLoreta(Y, Ut, s2,iLV,beta,alpha, options,hm,ind,L)

%[J,varargout] = dynamicLoreta(V,varargin)
%
% Computes the posterior distribution of the parameters J given some data V. 
% The program solves levels of inference: 1) optimization of parameters J, and
% 2) optimization of hyperparameters sigma and tau. See references for details.
%
% Ut,s2, and iLV are defined as follows: 
%     Y: Nsensors x time points data matrix
%     K: N x P predictor matrix
%     L: sparse P x P square root of the precision matrix 
%     [U,s,V] = svd( K*\L )
%     iLV = inv(L)*V
%     s2  = s.^2
%
% sigma, tau: hyperparameters
% J: current source density (estimated parameters)
% 
%                     p(V|J,sigma)*P(J|tau)
% p(J|V,sigma,tau) = ---------------------- 
%                        p(V|sigma,tau)
% 
%                     /      
% l(sigma, tau) = log | p(V|J,sigma) *p(J|tau)
%                     /
% References:
%   Trujillo-Barreto, N., Aubert-Vazquez, E., Valdes-Sosa, P.A., 2004.
%       Bayesian model averaging in EEG/MEG imaging. NeuroImage 21, 1300???1319
%
%   Yamashita, O., Galka,A., Ozaki, T., Biscay, R. Valdes-Sosa, P.A., 2004.
%       Recursive Penalized Least Squares Solution for Dynamical Inverse Problems
%       of EEG Generation. Human Brain Mapping 21:221–235
%
% Author: Alejandro Ojeda, Syntrogi Inc., Jan-2014

if nargin < 4, error('Not enough input arguments.');end
if nargin < 5, beta = [];end
if nargin < 6, alpha = [];end
if nargin < 7,
    options = struct('maxTol',1e-6,'maxIter',100,'gridSize',100,'verbose',true,'history',true,'useGPU',false,'initNoiseFactor',0.001);
end
if isempty(options)
    options = struct('maxTol',1e-6,'maxIter',100,'gridSize',100,'verbose',true,'history',true,'useGPU',false,'initNoiseFactor',0.001);
end
[history.beta, history.alpha, history.error, history.logE, history.df] = deal(nan(1,options.maxIter));
error_win = 3;
UtY = Ut*Y;
s = s2.^(0.5);
m = size(iLV,1);
n = size(Ut,1);
t = size(Y,2);

% Initialize hyperparameters
if isempty(beta) || isempty(alpha)
    %lambda2 = gcv_func(UtY,s,m,options.gridSize,0);
    %beta  = options.initNoiseFactor*(Y(:)'*Y(:))/n;
    %alpha    = beta*lambda2;
    beta = rand;
    alpha = rand;
end
history.beta(1) = beta;
history.alpha(1) = alpha;
history.error(1) = inf;
history.logE(1) = -inf;

if options.verbose
    fprintf('Iter\tGamma\tAlpha\t\tBeta\t\tHyperp. Error\tlogE\n');
end
for it=2:options.maxIter
        
    % Updating hyperparameters
    %gamma = sum(beta*s2./(beta*s2+alpha));
    gamma = sum(s./(s+alpha));
    alpha = gamma/mean(sum(bsxfun(@times,beta*s./(beta*s2+alpha),UtY).^2));
    %alpha = gamma./sum(sum(bsxfun(@times,beta*s./(beta*s2+alpha),UtY).^2));
    
    norm_err = sum(bsxfun(@times,alpha./(beta*s2+alpha),UtY).^2); %   ||y-y_hat||^2
    mse = mean(norm_err);
    beta = (n-gamma)/(mse+eps);
    
    
    % Compute the log-evidence
    E = beta/2*mse + 1/2*alpha*sum(sum(bsxfun(@times,UtY,beta*s./(beta*s2+alpha)).^2))/t;
    logE = m/2*log(alpha) + t*n/2*log(beta) -E -t/2*sum(log(alpha+beta*s2)); -t*n/2*log(2*pi);
        
    history.beta(it) = beta;
    history.alpha(it) = alpha;
    history.df(it-1) = gamma;
    history.logE(it) = logE;
    if it-error_win-1 < 1
        err = 0.5*std(history.beta(1:it)) + 0.5*std(history.alpha(1:it));
    else
        err = 0.5*std(history.beta(it-error_win:it)) + 0.5*std(history.alpha(it-error_win:it));
    end
    history.error(it) = err;
    
    if options.verbose
        fprintf('%i\t%5g\t%e\t%e\t%e\t%e\n',it-1,gamma,alpha,beta,err,history.logE(it));
    end
    if err < options.maxTol, break;end
end
history.df(it) = sum(beta*s2./(beta*s2+alpha));
if it == options.maxIter, 
    warning('Maximum iteration reached. Failed to converge.');
    [~,ind] = max(history.logE);
    beta = history.beta(ind);
    alpha = history.alpha(ind);
end
if options.verbose
    fprintf('\n')
end
if ~options.history
    history = [];
end
% [logE,ind] = max(history.logE);
% beta = history.beta(ind);
% alpha = history.alpha(ind);

% parameters's estimation
%T = iLV*bsxfun(@times,s./(s2+lambda2),Ut);
T = iLV*bsxfun(@times,beta*s./(beta*s2+alpha),Ut);
J = T*Y;
end