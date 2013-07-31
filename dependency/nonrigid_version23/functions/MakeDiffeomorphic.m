function [O_trans,Spacing]=MakeDiffeomorphic(O_trans,Spacing,sizeI)
% This function MakeDiffeomorphic will make the b-spline grid diffeomorphic
% thus will regularize the grid to have a Jacobian larger then zero
% for the whole b-spline grid.
%
% [Grid,Spacing]=point_registration_diffeomorphic(Grid,Spacing,sizeI)
%
%  Grid: The b-spline controlpoints, can be used to transform another
%        image in the same way: I=bspline_transform(Grid,I,Spacing);
%  Spacing: The uniform b-spline knot spacing
%  sizeI : Size of the image (volume)
%
% Example, 2D Diffeomorphic Warp
%    Xstatic=[1 1;
%      1 128;
%      64+32 64
%      64-32 64
%      128 1;
%      128 128];
% 
%   Xmoving=[1 1;
%      1 128;
%      64-32 64
%      64+32 64
%      128 1;
%      128 128];
%   option=struct; options.MaxRef=4;
%   sizeI=[128 128]; 
%   [O_trans,Spacing]=point_registration(sizeI,Xstatic,Xmoving,options);
%   [O_trans,Spacing]=MakeDiffeomorphic(O_trans,Spacing,sizeI);
%
%   Igrid=make_grid_image(Spacing*2,sizeI);
%   [Ireg,B]=bspline_transform(O_trans,Igrid,Spacing,3);
%   figure, imshow(Ireg)
%
% Example, 3D Diffeomorphic Warp
%   Xstatic=[1 1 64;
%      1 128 64;
%      64+32 64 64
%      64-32 64 64
%      128 1 64;
%      128 128 64];
% 
%   Xmoving=[1 1 64;
%      1 128 64;
%      64-32 64 64
%      64+32 64 64
%      128 1 64;
%      128 128 64];
%   option=struct; options.MaxRef=3;
%   sizeI=[128 128 128]; 
%   [O_trans,Spacing]=point_registration(sizeI,Xstatic,Xmoving,options);
%   [O_trans,Spacing]=MakeDiffeomorphic(O_trans,Spacing,sizeI);
%
%   Igrid=make_grid_image(Spacing,sizeI);
%   [Ireg,B]=bspline_transform(O_trans,Igrid,Spacing,3);
%   showcs3(Ireg);
%
%   Function is written by D.Kroon University of Twente (March 2011)

% Make Diffeomorphic
optim=struct('GradObj','on','GoalsExactAchieve',1,'StoreN',10,'HessUpdate','lbfgs','Display','iter','MaxIter',25,'DiffMinChange',0.001,'DiffMaxChange',1,'MaxFunEvals',1000,'TolX',0.005,'TolFun',1e-8);
O_trans = fminlbfgs(@(x)jacobiandet_cost_gradient(x,size(O_trans),O_trans,sizeI,Spacing),make_init_grid(Spacing,sizeI),optim);
    