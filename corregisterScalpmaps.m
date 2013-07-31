function [scalpmaps,xyzSensors,sensorLabels] = corregisterScalpmaps(scalpmaps,xyzSensors,landmarkCoordinates,landmarkLabels,templateModelFile)
% [scalpmaps,xyzSensors] = corregisterScalpmaps(scalpmaps,xyzSensors,landmarkCoordinates,...
%    landmarkLabels,templateModelFile)
%
% Automatic corregistration of an individual montage to a template, it uses
% a common set of landmarks to estimate the affine mapping from the
% individual space to the template.
% 
% Input:
%  scalpmaps:           Nsensors x Nscalpmaps to be corregistered
%  xyzSensors:          Nsensors x 3 with the sensor locations in world coordinates
%  landmarkCoordinates: Nlandmarks x 3 coordinates of the points used for matching individual and template spaces
%  landmarkLabels:      Nlandmarks x 1 cell array of strings with the label of each landmar
%
% Output:
%  scalpmaps:  spatialy resampled scalpmaps (if is needed)
%  xyzSensors: warped individual montage
%
% See also: headModel, geometricTools, variationalDynLoreta
% For more details visit https://code.google.com/p/mobilab/
%
% Author: Alejandro Ojeda, SCCN/INC/UCSD, Mar-2013


if nargin < 2, error('Not enough input parameters.');end
if nargin < 5
    path = fileparts(which('headModel.m'));
    templateModelFile = pickfiles(path,'head_model_hmObject_Colin27_3751_with_orientations.mat');
    if isempty(templateModelFile), error('Cannot find the head model template.');end
    templateModelFile = deblank(templateModelFile(1,:));
end
    
if ~exist(templateModelFile,'file'), error('Cannot find the head model template.');end

[~,tname] = fileparts(templateModelFile);
disp(['Loading template: ' tname ' ...'])
hmObj = headModel.loadFromFile(templateModelFile);
sensorLabels = hmObj.getChannelLabels;
classScalpmaps = class(scalpmaps);

% target space
T = [hmObj.fiducials.nasion;...
    hmObj.fiducials.lpa;...
    hmObj.fiducials.rpa];

% source space
S = landmarkCoordinates;

[~,loc1,loc2] = intersect({'nasion';'lpa';'rpa'},landmarkLabels,'stable');
if ~isempty(loc1)
    T = T(loc1,:);
    S = S(loc2,:);
else
    [~,loc1,loc2] = intersect(hmObj.getChannelLabels,landmarkLabels,'stable');
    if isempty(loc1), error('Cannot find a match between the your channel labels and the 10/20 system.');end
    T = hmObj.channelSpace(loc1,:);
    S = landmarkCoordinates(loc2,:);
end

% affine coregistration of the head
disp('Estimating the affine mapping...');
Aff = geometricTools.affineMapping(S,T);
xyzSensors = geometricTools.applyAffineMapping(xyzSensors, Aff);
                
if ~isempty(which('bspline_trans_points_double'))
    disp('B-Spline corregistration...');
    S = xyzSensors;
    T = hmObj.channelSpace;
    [S,d] = geometricTools.nearestNeighbor(S, T);
    z = zscore(d);
    th = norminv(0.90);
    S(abs(z)>th,:) = [];
    T(abs(z)>th,:) = [];
    options.Verbose = true;
    options.MaxRef = 2;
    [Def,spacing,offset] = geometricTools.bSplineMapping(S,T,S,options);
    xyzSensors = geometricTools.applyBSplineMapping(Def,spacing,offset,xyzSensors);
else
    disp('Affine corregistration is not always enough, for a second step bspline corregistration add ''.../mobilab/dependency/nonrigid_version23'' to the path');
    disp('or you can download nonrigid_version23 from: http://www.mathworks.com/matlabcentral/fileexchange/20057-b-spline-grid-image-and-point-based-registration');
end

% spatial resampling if needed
if size(hmObj.channelSpace,1) ~= size(xyzSensors,1)
    
    % Gaussian kernel interpolation
    W = geometricTools.localGaussianInterpolator(xyzSensors,hmObj.channelSpace,3);
    scalpmaps = W*double(scalpmaps); %#ok
    scalpmaps = eval([classScalpmaps '(scalpmaps);']);
end
