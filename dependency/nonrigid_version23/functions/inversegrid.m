function [O_trans2,Spacing2]=inversegrid(O_trans,Spacing,sizeI,MaxRef)
% Calculates the (numeric) inverse of the current b-spline grid
%
% [O_trans2,Spacing2]=inversegrid(O_trans,Spacing,sizeI,MaxRef);
%
% Example,
%
% % Read two greyscale images of Lena
% cd ..
% Imoving=imread('images/lenag1.png');
% Istatic=imread('images/lenag2.png');
%  
% % Register the images
% [Ireg,O_trans,Spacing,M,B,F] = image_registration(Imoving,Istatic);
%
% % Show the registration result
% figure,
% subplot(2,2,1), imshow(Imoving); title('moving image');
% subplot(2,2,2), imshow(Istatic); title('static image');
% subplot(2,2,3), imshow(Ireg); title('registerd moving image');
%
% % Show also the static image transformed to the moving image
% [O_trans2,Spacing2]=inversegrid(O_trans,Spacing,size(Imoving),4);
% Ireg2=bspline_transform(O_trans2,Istatic,Spacing2,3);
% subplot(2,2,4), imshow(Ireg2); title('registerd static image');
%
O_ref = make_init_grid(Spacing/8,[sizeI(1) sizeI(2)]);
x=O_ref(:,:,1); 
y=O_ref(:,:,2); 
x=x(:);
y=y(:);
check=(x<1)|(x>sizeI(1))|(y<1)|(y>sizeI(2));
x(check)=[]; y(check)=[];
X2(:,1) = x(:); X2(:,2) = y(:);
X1 = bspline_trans_points_double(O_trans,Spacing,X2);
Options.MaxRef=MaxRef;
[O_trans2,Spacing2]=point_registration([sizeI(1) sizeI(2)],X1,X2,Options);
