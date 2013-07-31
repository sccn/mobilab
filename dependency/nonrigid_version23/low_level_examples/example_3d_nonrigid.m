% Example of 3D non-rigid registration
% using steepest gradient optimizer and grid refinement.

% clean
clear all; close all; clc;

% Add all function paths
addpaths

% Get the volume data
[I1,I2]=get_example_data;

% Show the volume data
showcs3(I1);
showcs3(I2);

% Start b-spline grid dimensions (in the image)
Spacing=[26 26 12];

% Type of registration error used see registration_error.m
options.type='sd';
% Fast forward error gradient instead of central gradient.
options.centralgrad='false';


% Convert all values tot type double
I1=double(I1); I2=double(I2); 

% Resize I1 to fit I2
I1=imresize3d(I1,[],size(I2),'linear');

% Make the Initial b-spline registration grid
O_trans=make_init_grid(Spacing,size(I1));
O_trans=double(O_trans); 

% Smooth both images for faster registration
I1s=imgaussian(I1,1.3,[6 6 6]);
I2s=imgaussian(I2,1.3,[6 6 6]);

% Optimizer parameters
optim=struct('Display','iter','GradObj','on','HessUpdate','lbfgs','MaxIter',5,'DiffMinChange',0.03,'DiffMaxChange',1,'TolFun',1e-14,'StoreN',5);

% Reshape O_trans from a matrix to a vector.
sizes=size(O_trans); O_trans=O_trans(:);

% Start the b-spline nonrigid registration optimizer
O_trans = fminlbfgs(@(x)bspline_registration_gradient(x,sizes,Spacing,I1s,I2s,options),O_trans,optim);

% Reshape O_trans from a vector to a matrix
O_trans=reshape(O_trans,sizes);

% Refine the b-spline grid
[O_trans,Spacing]=refine_grid(O_trans,Spacing,size(I1s));

% Reshape O_trans from a matrix to a vector.
sizes=size(O_trans); O_trans=O_trans(:);

% Start the b-spline nonrigid registration optimizer
O_trans = fminlbfgs(@(x)bspline_registration_gradient(x,sizes,Spacing,I1s,I2s,options),O_trans,optim);

% Reshape O_trans from a vector to a matrix
O_trans=reshape(O_trans,sizes);

% Transform the input image with the found optimal grid.
Icor=bspline_transform(O_trans,I1,Spacing); 

% Make a (transformed) grid image
Igrid=make_grid_image(Spacing,size(I1));
Igrid=bspline_transform(O_trans,Igrid,Spacing); 

% Show the registration results
figure,
showcs3(Icor);
showcs3(Igrid);
