function F=backwards2forwards(B)
% This function will turn a backward transformation field in to a 
% forward transformation field, using gaussian kernel splatting.
%
% Thus when you have transformed image1 into image2 with the Rueckert
% Registration, this function can reverse the vectorField allowing the
% transformation of image2 into image 1.
% 
% Note: Some small interpolation artifacts will be present, thus derivatives
%  of the transformation field used for strain measurements are less reliable.
%
%
% F=backwards2forwards(B); 
%
% inputs,
%   B: The backward transformation fields
%
% outputs,
%   F: The forward transformation fields
%
% Function is written by D.Kroon University of Twente (November 2008)

if(ndims(B)==3) % Detect if 2D or 3D
    % Gaussian kernel (Must be symetric and odd in size)
    H=fspecial('gaussian',[7 7],5/6);
    Bx=2*imresize(B(:,:,1),2,'bilinear');  
    By=2*imresize(B(:,:,2),2,'bilinear');
    [Fx,Fy]=backwards2forwards_2d_double(double(Bx),double(By),double(H));
    F(:,:,1)=(1/2)*imresize(Fx,1/2,'bilinear'); 
    F(:,:,2)=(1/2)*imresize(Fy,1/2,'bilinear');
else
    sigma=9/6;
    [x,y,z] = ndgrid(-3:3,-3:3,-3:3);
    H = exp(-(x.*x/2/sigma^2 + y.*y/2/sigma^2 + z.*z/2/sigma^2)); H = H/sum(H(:));
    if(isa(B,'double'))
        [Fx,Fy,Fz]=backwards2forwards_3d_double(double(B(:,:,:,1)),double(B(:,:,:,2)),double(B(:,:,:,3)),double(H));
    else
        [Fx,Fy,Fz]=backwards2forwards_3d_single(single(B(:,:,:,1)),single(B(:,:,:,2)),single(B(:,:,:,3)),single(H));
    end
    F(:,:,:,1)=Fx;
    F(:,:,:,2)=Fy;
    F(:,:,:,3)=Fz;
end

        