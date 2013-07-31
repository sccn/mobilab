function [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Iin,dx,dy,mode)
% Bspline transformation grid function
% 
% [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Iin,dx,dy,mode)
%
% Inputs,
%   Ox, Oy : are the grid points coordinates
%   Iin : is input image, Iout the transformed output image
%   dx and dy :  are the spacing of the b-spline knots
%   mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            2: cubic interpolation and outside pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
%
% Outputs,
%   Iout: The transformed image
%   Tx: The transformation field in x direction
%   Ty: The transformation field in y direction
%
% This function is an implementation of the b-spline registration
% algorithm in "D. Rueckert et al. : Nonrigid Registration Using Free-Form 
% Deformations: Application to Breast MR Images".
% 
% We used "Fumihiko Ino et al. : a data distrubted parallel algortihm for 
%  nonrigid image registration" for the correct formula's, because 
% (most) other papers contain errors. 
%
% Function is written by D.Kroon University of Twente (June 2009)

% Make all x,y indices
[x,y]=ndgrid(0:size(Iin,1)-1,0:size(Iin,2)-1);

% Calulate the transformation of all image coordinates by the b-sline grid
O_trans(:,:,1)=Ox; O_trans(:,:,2)=Oy;
Tlocal=bspline_trans_points_double(O_trans,[dx dy],[x(:) y(:)],false);

switch(mode)
	case 0
		Interpolation='bilinear';
		Boundary='replicate';
	case 1
		Interpolation='bilinear';
		Boundary='zero';
	case 2
		Interpolation='bicubic';
		Boundary='replicate';
	otherwise
		Interpolation='bicubic';
		Boundary='zero';
end

if(nargout>1)
    % Store transformation fields
    Tx=reshape(Tlocal(:,1),[size(Iin,1) size(Iin,2)])-x;  
    Ty=reshape(Tlocal(:,2),[size(Iin,1) size(Iin,2)])-y; 
end
Iout=image_interpolation(Iin,Tlocal(:,1),Tlocal(:,2),Interpolation,Boundary);
