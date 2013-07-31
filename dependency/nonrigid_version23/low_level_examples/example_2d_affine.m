% example of 2D affine registration using the lsqnonlin optimizer.

% clean
clear all; close all; clc;

% Add all function paths
addpaths

% Read two greyscale images of Lena
I1=im2double(imread('lenag3.png')); 
I2=im2double(imread('lenag2.png'));

% Type of registration error used see registration_error.m
type='d';

% Smooth both images for faster registration
I1s=imfilter(I1,fspecial('gaussian'));
I2s=imfilter(I2,fspecial('gaussian'));

% Parameter scaling of translateX translateY rotate resizeX resizeY 
% (ShearXY, ShearYX can also be included add the end of the vector, but 
%   because rotation can be seen as a type of shear transform, the scaling 
%       of the rotation or shear transform must be very small)
scale=[1 1 1 0.01 0.01];

[x]=lsqnonlin(@(x)affine_registration_image(x,scale,I1s,I2s,type),[0 0 0 100 100],[],[],optimset('Display','iter','MaxIter',100));

% Scale the translation, resize and rotation parameters to the real values
x=x.*scale;

% Make the affine transformation matrix
M=make_transformation_matrix(x(1:2),x(3),x(4:5));

Icor=affine_transform(I1,M,3); % 3 stands for cubic interpolation

% Show the registration results
figure,
subplot(1,3,1), imshow(I1);
subplot(1,3,2), imshow(I2);
subplot(1,3,3), imshow(Icor);
