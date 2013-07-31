function Dlocal=jacobiandet_transform_2d_double(Ox,Oy,Isize,dx,dy)
% Determinant of the Jacobian of the  transformation grid 
% 
% [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Isize,dx,dy)
%
% Inputs,
%   Ox, Oy : are the grid points coordinates
%   Isize : The size of the input image [m n] or in 3d [m n k]
%   dx and dy :  are the spacing of the b-spline knots

% Make all x,y indices
[x,y]=ndgrid(0:Isize(1)-1,0:Isize(2)-1);

% Calulate the transformation of all image coordinates by the b-sline grid
O_trans(:,:,1)=Ox; O_trans(:,:,2)=Oy;

[Tlocal,Dlocal]=bspline_trans_points_double(O_trans,[dx dy],[x(:) y(:)],false);
Dlocal=reshape(Dlocal,Isize);