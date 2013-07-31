% Exampleusing fminunc optimizer and mutual information and grid
% refinement
%
% Example is written by D.Kroon University of Twente (OCtober 2008)

clear all; close all; clc

% Add all function paths
addpaths

% Read two greyscale images of Lena
I1=im2double(imread('brain3.png'));
I2=im2double(imread('brain2.png'));

% Type of registration error used see registration_error.m
options.type='mi';

% No smooth registration grid penalty
options.penaltypercentage=0;

% b-spline grid spacing in x and y direction
Spacing=[60 60];

% Make the Initial b-spline registration grid
[O_trans]=make_init_grid(Spacing,size(I1));

% Convert all values tot type double
I1=double(I1); I2=double(I2); O_trans=double(O_trans); 

% Smooth both images for faster registration
I1s=imfilter(I1,fspecial('gaussian',[3 3],0.5));
I2s=imfilter(I2,fspecial('gaussian',[3 3],0.5));

% Optimizer parameters
optim=optimset('GradObj','on','LargeScale','off','Display','iter','MaxIter',100,'DiffMinChange',0.01,'DiffMaxChange',1,'MaxFunEvals',1000);

% Reshape O_trans from a matrix to a vector.
sizes=size(O_trans); O_trans=O_trans(:);

% Start the b-spline nonrigid registration optimizer
O_trans = fminunc(@(x)bspline_registration_gradient(x,sizes,Spacing,I1s,I2s,options),O_trans,optim);

% Reshape O_trans from a vector to a matrix
O_trans=reshape(O_trans,sizes);


% Refine the b-spline grid
[O_trans,Spacing]=refine_grid(O_trans,Spacing,size(I1));
		
% Reshape O_trans from a matrix to a vector.
sizes=size(O_trans); O_trans=O_trans(:);

% Start the b-spline nonrigid registration optimizer
O_trans = fminunc(@(x)bspline_registration_gradient(x,sizes,Spacing,I1s,I2s,options),O_trans,optim);

% Reshape O_trans from a vector to a matrix
O_trans=reshape(O_trans,sizes);
    
% Transform the input image with the found optimal grid.
[Icor,Tx,Ty]=bspline_transform(O_trans,I1,Spacing); 


% Show the registration results
figure,
subplot(2,2,1), imshow(I1); title('input image 1');
subplot(2,2,2), imshow(I2); title('input image 2');
subplot(2,2,3), imshow(Icor); title('transformed image 1');
subplot(2,2,4), imshow(Tx,[]); title('Tranform in x direction');


