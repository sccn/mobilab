function [V1,V2]=get_example_data
% get_example_data is used to make some rigid and nonrigid transformed
% 3D MRI example data for the  registration examples.
%
% example,
% [V1,V2]=get_example_data;
%
% Function is written by D.Kroon University of Twente (October 2008)
addpaths;

load mri; D=squeeze(D); V2=single(D)./single(max(D(:))); clear D;
V2=imresize3d(V2,[],[128 128 64]);

Spacing=[20 20 20];
O_trans=make_init_grid(Spacing,[128 128 64]);
O_trans(2:7,2:7,2:6,:)=O_trans(2:7,2:7,2:6,:)+6*rand([6 6 5 3])-3;
V1=bspline_transform(O_trans,V2,Spacing);
V1=(V1<0.9).*V1;
t=clock; t=rand(1,round(t(6))+1); clear t;
par=(rand(1,9)-0.5)*5;
M=mat_tra([par(1) par(2) par(3)])*mat_siz([1+par(4)*0.01 1+par(5)*0.01 1+par(6)*0.01])*mat_rot([par(7) par(8) par(9)]); 
V1=affine_transform(single(V1),single(M));
V1=imresize3d(V1,[],[80 80 40]);

function M=mat_rot(r)
    r=r*(pi/180);
    Rx=[1 0 0 0;
        0 cos(r(1)) -sin(r(1)) 0;
        0 sin(r(1)) cos(r(1)) 0;
        0 0 0 1];

    Ry=[cos(r(2)) 0 sin(r(2)) 0;
        0 1 0 0;
        -sin(r(2)) 0 cos(r(2)) 0;
        0 0 0 1];

    Rz=[cos(r(3)) -sin(r(3)) 0 0;
        sin(r(3)) cos(r(3)) 0 0;
        0 0 1 0;
        0 0 0 1];
    M=Rx*Ry*Rz;

function M=mat_siz(s)
	M=[s(1) 0 0 0;
	   0 s(2) 0 0;
	   0 0 s(3) 0;
	   0 0 0 1];

function M=mat_tra(t)
	M=[1 0 0 t(1);
	   0 1 0 t(2);
	   0 0 1 t(3);
	   0 0 0 1];
   