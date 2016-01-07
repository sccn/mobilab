function [J,lambdaOpt,Yhat,T] = ridgeGCV(Y,K,L,nlambda,plotGCV)
%[J,lambdaOpt] = ridgeGCV(Y,K,L,nlambda,plotGCV)
%
% Estimates a ridge regression model, also know as Tikhonov regularization, 
% or minimum norm with L2 prior (or Loreta in the EEG inverse solution literature). 
% For an implementation of sLORETA model see the function inverseSolutionLoreta.
%
% Y: measurements (Nsensors X 1)
% K: N X P predictor matrix
% L: P X P prior covariance matrix (sparse matrix is recomended)
% Jest: estimated parapeters
% nlambda: maximum size of the grid for the hyperparameter lambda, default: 100
% plotGCV: plot the GCV curve (true/false), default: false
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
%   Pedro A. Valdés-Hernández, Alejandro Ojeda, Eduardo Martínez-Montes, Agustín
%       Lage-Castellanos, Trinidad Virués-Alba, Lourdes Valdés-Urrutia, Pedro A.
%       Valdes-Sosa, 2009. White matter White matter architecture rather than 
%       cortical surface area correlates with the EEG alpha rhythm. NeuroImage 49
%       (2010) 2328–2339

if nargin < 2, error('Not enough input arguments.');end
[n,p] = size(K);
if nargin < 3, L = speye(p);end
if nargin < 4, nlambda = 100;end
if nargin < 5, plotGCV = false;end

[U,S,V] = svd(K/L,'econ');
Ut = U';
V = L\V;
s = diag(S);
s2 = s.^2;
[J,lambdaOpt,T] = ridgeSVD(Y,Ut, s2,V,nlambda,plotGCV);
if nargout > 2
    Yhat = K*J;
end