function [s_Trans, s_Rotation, s_Scale, s_Shear]=affine_parameter_scaling(sizeI)
% This function gives scaling values for different parameters, such as
% rotation and shear of an affine matrix. This scaling make the influence on
% pixel changes in registration of for instance a rotation value
% approximately equal to a small step in translation.  This improves the 
% result and speed of image registration with (steepest descent based) optimizers.
% 
% [s_Trans, s_Rotation, s_Scale, s_Shear] = affine_parameter_scaling(sizeI)
%
% Literature :
%   Colin studholme et Al. "Automated Multimodality Registration Using the 
%   Full Affine Transformation: Application to MR and CT Guided Skull 
%   Base Surgery"
%
% Note!! : We used the literature above, but had to introduce some magic
% constant of value 58 which is multiplied with the rotation-scaling-value
% to get results which are equal to the scaling found with a numerical
% test on an volume with random noise. Test code:
%   I1=rand(round(rand(1,3)*400));
%   t=[1e-5 0 0]; r=[0 0 0]; s=[1 1 1]; h=[0 0 0 0 0 0];
%   M1=make_transformation_matrix(t,r,s,h);
%   t=[0 0 0]; r=[1e-5 0 0]; s=[1 1 1]; h=[0 0 0 0 0 0];
%   M2=make_transformation_matrix(t,r,s,h);
%   Iout1 = affine_transform(I1,M1,2);
%   Iout2 = affine_transform(I1,M2,2);
%   err1=sum(abs(Iout1(:)-I1(:)))/step;
%   err2=sum(abs(Iout2(:)-I1(:)))/step;
%   scale=err1/err2;
%
%  
% Function is written by D.Kroon University of Twente (August 2010)

dt=1; c=58;

Fx=sizeI(1);
Fy=sizeI(2);
Fz=sizeI(3);

axy = atan(Fy/Fx); bxy = pi/2 - axy;
ayz = atan(Fz/Fy); byz = pi/2 - ayz;
axz = atan(Fz/Fx); bxz=  pi/2 - axz;

Kxy  = Fx^3*(sin(axy)/cos(axy)^2+log(abs(tan((pi/4)+(axy/2)))))+ Fy^3*(sin(bxy)/cos(bxy)^2+log(abs(tan((pi/2)-(axy/2)))));
Kxz  = Fx^3*(sin(axz)/cos(axz)^2+log(abs(tan((pi/4)+(axz/2)))))+ Fz^3*(sin(bxz)/cos(bxz)^2+log(abs(tan((pi/2)-(axz/2)))));
Kyz  = Fy^3*(sin(ayz)/cos(ayz)^2+log(abs(tan((pi/4)+(ayz/2)))))+ Fz^3*(sin(byz)/cos(byz)^2+log(abs(tan((pi/2)-(ayz/2)))));

s_Trans(1) = dt;
s_Trans(2) = dt;
s_Trans(3) = dt;

s_Rotation(1) = abs( 2 * asin( (6 * dt *Fy*Fz )/ Kyz) )*c;
s_Rotation(2) = abs( 2 * asin( (6 * dt *Fx*Fz )/ Kxz) )*c;
s_Rotation(3) = abs( 2 * asin( (6 * dt *Fx*Fy )/ Kxy) )*c;

s_Scale(1) = 4 * dt / Fx;
s_Scale(2) = 4 * dt / Fy;
s_Scale(3) = 4 * dt / Fz;

s_Shear(1) = 4 *dt / Fy;
s_Shear(2) = 4 *dt / Fz;
s_Shear(3) = 4 *dt / Fx;
s_Shear(4) = 4 *dt / Fz;
s_Shear(5) = 4 *dt / Fx;
s_Shear(6) = 4 *dt / Fy;
