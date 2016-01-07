function [J,lambda2,T] = ridgeSVD(Y,Ut, s2,V,nlambda,plotGCV)
%[J,lambdaOpt,T] = ridgeSVD(Y,Ut, s2,V,nlambda,plotGCV)
%
% Estimates a ridge regression model, also know as Tikhonov regularization, 
% or minimum norm with L2 prior (or Loreta in the EEG inverse solution literature). 
% For an implementation of sLORETA model see the function inverseSolutionLoreta.
%
% Y: measurements (Nsensors X 1)
% Ut, s2,V are defined as the SVD decomposition of the standardized lead field matrix
% nlambda: maximum size of the grid for the hyperparameter lambda, default: 100
% plotGCV: plot the GCV curve (true/false), default: false
% Jest: estimated parapeters
% T: estimated inverse operatormaximum size of the grid for the hyperparameter lambda, default: 100
% 
% Jest = argmin(J) ||Y-K*J||^2 + lambda*||L*J||^2 == argmin(J) ||Y-K/L*Jst||^2 + lambda*||I||^2, s.t. J = L/Jst 
% and lambda > 0
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

if nargin < 4, error('Not enough input arguments.');end
if nargin < 5, nlambda = 100;end
if nargin < 6, plotGCV = false;end

p = size(V,1);
s = sqrt(s2);
UtY = Ut*Y;
lambda2 = gcv_func(UtY,s,p,nlambda,plotGCV);
T = V*diag(s./(s2+lambda2))*Ut;
J = T*Y;                            % J = (K'*K+lambda*L'*L)\K'*Y