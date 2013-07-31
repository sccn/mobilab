% Example of nonrigid registration using steepest gradient optimizer
%
% Example is written by D.Kroon University of Twente (October 2008)

% clean
clear all; 
close all; 

% Add all function paths
addpaths

% Read two greyscale images of Lena
I1=im2double(imread('lenag1.png'));
I2=im2double(imread('lenag2.png'));

% Type of registration error used see registration_error.m
options.type='sd';

% Use fast forward instead of central error gradient
options.centralgrad=false;

% Use cubic interpolation
options.interpolation='cubic';

% b-spline grid spacing in x and y direction
Spacing=[32 32];

% Make the Initial b-spline registration grid
[O_trans]=make_init_grid(Spacing,size(I1));

% Convert all values tot type double
I1=double(I1); I2=double(I2); O_trans=double(O_trans); 

% Smooth both images for faster registration
I1s=imfilter(I1,fspecial('gaussian'))+0.001;
I2s=imfilter(I2,fspecial('gaussian'))+0.001;

% Optimizer parameters
optim=struct('Display','iter','GradObj','on','HessUpdate','lbfgs','MaxIter',30,'DiffMinChange',0.03,'DiffMaxChange',1,'TolFun',1e-14,'StoreN',5,'GoalsExactAchieve',0);

% Reshape O_trans from a matrix to a vector.
sizes=size(O_trans); O_trans=O_trans(:);

% Start the b-spline nonrigid registration optimizer
O_trans = fminlbfgs(@(x)bspline_registration_gradient(x,sizes,Spacing,I1s,I2s,options),O_trans,optim);

% Reshape O_trans from a vector to a matrix
O_trans=reshape(O_trans,sizes);

% Transform the input image with the found optimal grid.
Icor=bspline_transform(O_trans,I1,Spacing,3); 

% Make a (transformed) grid image
Igrid=make_grid_image(Spacing,size(I1));
[Igrid,Tx,Ty]=bspline_transform(O_trans,Igrid,Spacing); 

% Show the registration results
figure,
subplot(2,2,1), imshow(I1); title('input image 1');
subplot(2,2,2), imshow(I2); title('input image 2');
subplot(2,2,3), imshow(Icor); title('transformed image 1');
subplot(2,2,4), imshow(Igrid); title('grid');

