% In case your channel labels don't follow the 10/20 convention use
% fiducial landmaks as starting point for the co-registration
eegfile           = '357_epoch4study_E.set';
montageFile          = '357_04112012_56A_CEScap_conducting.sfp';
templateModelFile = 'head_model_hmObject_Colin27_3751_with_orientations';

% load the head model template
hmObj = headModel.loadFromFile(templateModelFile);

% load the EEGLAB .set file
EEG = pop_loadset(eegfile);

% load the sensor positions and fiducials
[xyz, fiducials, fiducials_xyz, fiducials_labels] = readMontage(EEG,montageFile);

% keep only the channels used for ICA
xyz = xyz(EEG.icachansind);


% scalp maps to invert
ICs = [1 2 3 4 5 6 7 8 9 10 11 12 13 17 30];
pop_topoplot(EEG,0,ICs);
V = EEG.icawinv(:,ICs);


% warp scalp topographies to template's head
Vi = corregisterScalpmaps(hmObj, V, xyz);


% estimating the primary current density
areDependentSamples = true;
J = scalpmap2pcd(headModelObj, scalpmaps_i, areDependentSamples);








template = '/data/common/matlab/eeglab/plugins/mobilab/data/head_model_hmObject_Colin27_11997_with_no_orientations.mat';
[scalpmaps_i,xyzSensors_i] = corregisterScalpmaps(scalpmaps,xyzSensors,landmarkCoordinates,landmarkLabels,template);



[J,hViewer] = scalpmap2pcd(scalpmaps_i,true,true);

[J,hViewer] = scalpmap2pcd(scalpmaps_i,true,true, template);