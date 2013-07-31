function [J,hViewer] = scalpmap2pcd(V,independentSamples,plotFlag, templateModelFile)
% This function is a wrapper of variationalDynLoreta.m for localizing the cortical
% Primary Source Density of EEG voltage topographies (or ICA scalp-maps).
% This function uses a head model template constructed in MoBILAB using MNI
% Colin27, it has a suorce space of 3751 vertices and the orientations.
%
% Foe more details visit: http://sccn.ucsd.edu/wiki/Mobilab_software#eeg
%
% See also: variationalDynLoreta
%
% V:        Nsensors x Nscalpmaps to localize
% dependentSamples: if true, each column of V is a dependent sample e.g., V(t), V(t+1),..., V(t+n), then, 
%                   updates the hyper-parameters in a 64 samples sliding window, otherwise assumes that each
%                   sample is independent, therefore the hyper-parameters are optimized in a pointwise manner,
%                   in this case the solution is equivalen to sLoreta
% plotFlag:         flag, if true plots the estimated Primary Source Density
%
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Mar-2013



if nargin < 2, independentSamples = true;end
if nargin < 3, plotFlag = false;end
if nargin < 4
    path = fileparts(which('variationalDynLoreta.m'));
    templateModelFile = pickfiles(path,'head_model_hmObject_Colin27_3751_with_orientations.mat');
    if isempty(templateModelFile), error('Cannot find the template.');end
    templateModelFile = deblank(templateModelFile(1,:));
end

if ~exist(templateModelFile,'file'), error('Cannot find the template.');end
hmObj = headModel.loadFromFile(templateModelFile);

if size(hmObj.channelSpace,1) ~= size(V,1), error('Channel space don''t math the template. Run >> corregisterScalpmaps first.');end
if ~exist(hmObj.surfaces,'file'), error('Surfaces are missing in the head model.');end
load(hmObj.surfaces);
n = size(surfData(end).vertices,1); %#ok

% loading the lead field K
load(hmObj.leadFieldFile);
if ~exist('K','var'), disp('Lead field matrix is missing. Run >> hmObj.computeLeadFieldBEM first.');end

% opening the surfaces by the Thalamus
structName = {'Thalamus_L' 'Thalamus_R'};
[~,K,L,rmIndices] = getSourceSpace4PEB(hmObj,structName);
ind = setdiff(1:n,rmIndices);

% removing the average reference
Y = double(V);
Ns = size(V,2);
%[ny,Ns] = size(V);
%H = eye(ny) - ones(ny)/n;
%Y = H*Y;
%K = H*K;
%Y(end,:) = [];
%K(end,:) = [];
dim = size(K);

hasDirection = n == dim(2)/3+length(rmIndices);
if hasDirection
    J = zeros(n*3,Ns);
    tmp = zeros(n,3);
    tmp(ind,:) = 1;
    tmp = tmp(:);
    ind = find(tmp);
else
    J = zeros(n,Ns);
end

%--
[U,S,V] = svd(K/L,'econ');
Ut = U';
s2 = diag(S).^2;
iLV = L\V;
%--
hwait = waitbar(0,'Estimating PCD...','Color',[0.93 0.96 1]);
if independentSamples
    for it=1:Ns
        J(ind,it) = variationalDynLoreta(Ut,Y(:,it),s2,iLV,L);
        waitbar(it/Ns,hwait);
    end
else
    delta = 32;
    delta(delta>Ns) = Ns;
    
    drawnow;
    [J(ind,1:delta),alpha,beta] = variationalDynLoreta(Ut,Y(:,1:delta),s2,iLV,L);
    for it=delta+1:delta:Ns
        if it+delta <= Ns
             [J(ind,it-delta:it+delta-1),alpha,beta] = variationalDynLoreta(Ut,Y(:,it-delta:it+delta-1),s2,iLV,L,alpha,beta);
        else [J(ind,it-delta:end),       alpha,beta] = variationalDynLoreta(Ut,Y(:,it-delta:end),       s2,iLV,L,alpha,beta);
        end
        waitbar(it/Ns,hwait);
    end
end
close(hwait);

if plotFlag, hViewer = hmObj.plotOnModel(J,Y,'PCD');end