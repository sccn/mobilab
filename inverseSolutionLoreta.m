function [J,lambdaOpt] = inverseSolutionLoreta(Y,K,L,nlambda,plotGCV,threshold)
%[J,lambdaOpt] = inverseSolutionLoreta(Y,K,L,nlambda,plotGCV,threshold)
%
% Estimates the Primary Current Density of a given topographic map using
% a laplacian penalty. The estimated parametric map representing the PCD 
% is formaly equivalent to the sLORETA solution (Pascual-Marki, 2002;
% Trujillo-Barreto, 2004).
%
% Y: voltage on the sensors (Nsensors X Nsamples)
% K: lead field matrix (Nsensors X Nsources)
% L: spatial laplacian operator (Nsources X Nsources)
% Jest: estimated PCD (Nsources X Nsamples)
% 
% Jest = argmin(J) ||Y-K*J||^2 + lambda*||L*J||^2
% with lambda > 0
%
% This code is based on a previous implementation used in Valdes-Hernandez 
% et al. (2009), written by Alejandro Ojeda and Pedro Valdez-Hernandez at 
% the Cuban Neuroscience Center in 2009.
% 
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Jul-2012
%
% References:
%   Trujillo-Barreto, N., Aubert-Vazquez, E., Valdes-Sosa, P., 2004. Bayesian
%       model averaging. NeuroImage 21, 1300–1319.
%
%   Pascual-Marqui, R.D., 2002. Standardized low-resolution brain electromagnetic
%      tomography (sLORETA): technical details. Methods Find.
%       Exp. Clin. Pharmacol. 24 (Suppl. D), 5–12
%
%   Pedro A. Valdés-Hernández, Alejandro Ojeda, Eduardo Martínez-Montes, Agustín
%       Lage-Castellanos, Trinidad Virués-Alba, Lourdes Valdés-Urrutia, Pedro A.
%       Valdes-Sosa, 2009. White matter White matter architecture rather than 
%       cortical surface area correlates with the EEG alpha rhythm. NeuroImage 49
%       (2010) 2328–2339

if nargin < 3, error('Not enough input arguments.');end
if nargin < 4, nlambda = 100;end
if nargin < 5, plotGCV = false;end
if nargin < 6, threshold = [5 95];end

[n,p] = size(K);
[U,S,V] = svd(K/L,'econ');
V = L\V;
s = diag(S);
s2 = s.^2;
UtY = U'*Y;

tol = max([n p])*eps(max(s));
lambda2 = logspace(log10(tol),log10(max(s)),nlambda);
gcv = zeros(nlambda,1);
for it=1:nlambda
    d = lambda2(it)./(s2+lambda2(it));
    %f = diag(d)*UtY(:,1);
    f = mean(diag(d)*UtY,2);
    gcv(it) = dot(f,f,1)/sum(d)^2;
end
loc = getMinima(gcv);
if isempty(loc), loc = 1;end
loc = loc(end);
lambdaOpt = lambda2(loc);

T = V*diag(s./(s2+lambdaOpt))*U';
J = T*Y;                            % J = (K'*K+lambda*L'*L)\K'*Y

H = K*T;
E = sum(Y-H*Y,2);
sigma = E'*E/(n-trace(H));
dT = 1./sqrt(dot(T,T,2));
S = 1./sigma*dT;
J = bsxfun(@times,J,S);

J = bsxfun(@minus,J,median(J));
J = bsxfun(@rdivide,J,std(J));

try %#ok
    th = prctile(J,threshold);
    th = mean(th,2);
    %if any(threshold > 1), threshold = threshold/100;end
    %th = tinv(threshold,length(J)-1);
    if length(th) == 2
        J(J>th(1) & J<th(2)) = 0;
    else
        J(J<th) = 0;
    end
end

if plotGCV
    figure('Color',[0.93 0.96 1]);
    semilogx(lambda2,gcv)
    %plot(lambda2,gcv)
    xlabel('log-lambda');
    ylabel('GCV');
    hold on;
    plot(lambdaOpt,gcv(loc),'rx','linewidth',2)
    grid on;
end



%---
function indmin = getMinima(x)
fminor = diff(x)>=0;
fminor = ~fminor(1:end-1, :) & fminor(2:end, :);
fminor = [0; fminor; 0];
indmin = find(fminor);