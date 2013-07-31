% Example of 3D affine registration using the fminunc optimizer.

% clean
clear all; close all; clc;

% Add all function paths
addpaths

% Get the volume data
[I1,I2]=get_example_data;

% Convert all volumes from single to double
I1=double(I1); I2=double(I2);

% First resize volume I1 to match size of volume I2
I1=imresize3d(I1,[],size(I2),'linear');

% Show the volume data
showcs3(I1);
showcs3(I2);

% Type of registration error used see registration_error.m
type='sd';

% Smooth both images for faster registration
I1s=imgaussian(I1,1.3,[6 6 6]);
I2s=imgaussian(I2,1.3,[6 6 6]);

% Parameter scaling of  translateX translateY translateZ, rotateX rotateY rotateZ 
% resizeX resizeY resizeZ, shearXY, shearXZ, shearYX, shearYZ, shearZX, shearZY
% (Rotation made zero, because is already "included" in the shear transformations)
scale=[1 1 1 0 0 0 0.01 0.01 0.01 1 1 1 1 1 1];

[x]=fminlbfgs(@(x)affine_registration_error(x,scale,I1s,I2s,type),[0 0 0 0 0 0 100 100 100 0 0 0 0 0 0],struct('GradObj','on','Display','iter','MaxIter',100,'HessUpdate','lbfgs'));
%[x]=lsqnonlin(@(x)affine_registration_image(x,scale,I1s,I2s,type),[0 0 0 0 0 0 100 100 100 0 0 0 0 0 0],[],[],optimset('Display','iter','MaxIter',100));

% Scale the translation, resize and rotation parameters to the real values
x=x.*scale;

% Make the affine transformation matrix
M=make_transformation_matrix(x(1:3),x(4:6),x(7:9),x(10:15));
    
% Transform the input volume
Icor=affine_transform(I1,M);

% Show the registration results
figure,
showcs3(Icor);
