function [J,alpha,beta,T] = dynamicLoreta(Ut,Y,s2,V,iLV,L,alpha,beta,options)

%[J,varargout] = dynamicLoreta(V,varargin)
%
% Computes the posterior distribution of the parameters J given some data V. 
% The program solves levels of inference: 1) optimization of parameters J, and
% 2) optimization of hyperparameters alpha and beta. See Trujillo-Barreto
% et. al. (2004) for details.
%
% Ut,s2, and iLV are defined as follows: 
%     Y: Nsensors x time points data matrix
%     K: N x P predictor matrix
%     L: sparse P x P square root of the precision matrix 
%     [U,s,V] = svd( K*inv(L) )
%     iLV = inv(L)*V
%     s2  = s.^2
%
% alpha, beta: hyperparameters
% J: estimated parapeters
% 
%                     P(V|J,alpha)*P(J|beta)
% P(J|V,alpha,beta) = ---------------------- 
%                        P(V|alpha,beta)
% 
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Jan-2013
%
% References:
%   Trujillo-Barreto, N., Aubert-Vazquez, E., Valdes-Sosa, P.A., 2004.
%     Bayesian model averaging in EEG/MEG imaging. NeuroImage 21, 1300???1319



if nargin < 6, error('Not enough input arguments.');end
if nargin < 9
    options.maxTol = 1e-3;
    options.maxIter = 100;
    options.verbose = true;
    options.gridSize = 100;
end


s = s2.^(0.5);
n = length(s);
p = size(L,1);

% Initialize hyperparameters
if nargin < 8
    UtY = Ut*Y;
    tol = max([n p])*eps(max(s));
    lambda2 = logspace(log10(tol),log10(max(s)),options.gridSize);
    gcv = zeros(options.gridSize,1);
    parfor k=1:options.gridSize
        d = lambda2(k)./(s2+lambda2(k));
        f = diag(d)*UtY;
        gcv(k) = dot(f,f,1)/sum(d)^2;
    end
    loc = getMinima(gcv);
    if isempty(loc), loc = 1;end
    loc = loc(end);
    lambda2 = lambda2(loc);
     
    alpha = 0.001*(Y(:)'*Y(:))/n;
    beta = alpha*lambda2;
end
err       = inf;
aic_old   = inf;
alpha_old = alpha;
beta_old  = beta;
for it=1:options.maxIter
    if err < options.maxTol, break;end
        
    % computing sigma
    H = Ut'*diag(alpha*s2./(alpha*s2+beta))*Ut;    
    SSE = mean(sum((Y - H*Y).^2));
    JtJ = mean(Y'*Ut*diag(((alpha*s)./(alpha*s2+beta)).^2)*Ut'*Y);
    
    % computing gamma
    gamma = sum((alpha*s2)./(alpha*s2+beta));
    
    % computing AIC
    aic = -2*log(SSE) + 2*gamma;
    if aic > aic_old,
        alpha = alpha_old;
        beta = beta_old;
        break;
    end
    
    % updating hyperparameters
    alpha = 1/(SSE/(n-gamma));
    beta  = gamma/JtJ;
    
    err = 0.5*abs(alpha_old-alpha) + 0.5*abs(beta_old-beta);
    if options.verbose
        disp([num2str(it) ' => alpha: ' num2str(alpha) '  beta: ' num2str(beta) ' df: ' num2str(gamma) ' hyperp. error: ' num2str(err) ' sse: ' num2str(SSE) ' aic: ' num2str(aic)]);
    end
    aic_old = aic;
    alpha_old = alpha;
    beta_old  = beta;
    
    
end
if it == options.maxIter, warning('Maximum iteration reached. Failed to converge.');end

% parameters's estimation
T = iLV*diag(alpha.*s./(alpha.*s2+beta))*Ut;
J = T*Y;

% standardized Loreta
E = sum(Y-H*Y,2);
sigma = E'*E/(n-trace(H));
dT = 1./sqrt(dot(T,T,2));
S = 1./sigma*dT;
S = S./std(eps+S);
T = bsxfun(@times,T,S);%sqrt(p)*
J = bsxfun(@times,J,S);%sqrt(p)*
end


%---
function indmin = getMinima(x)
fminor = diff(x)>=0;
fminor = ~fminor(1:end-1, :) & fminor(2:end, :);
fminor = [0; fminor; 0];
indmin = find(fminor);
end

function T = standT(T,H,Y,n)
E = Y-H*Y;
df = (n-trace(H));
sigma = E'*E/df;
dT = 1./sqrt(dot(T,T,2));
S = 1./sigma*dT;
T = bsxfun(@times,S,T);
end