function I3=movepixels(I1,T,mode)
% This function movepixels, will (backwards) translate the pixels 
% of an 2D/3D image according to x, y (and z) translation images 
% (bilinear interpolated).
% The function is a wrapper around mex files movepixels_2d_double.c and
% movepixels_3d_double.c and movepixels_3d_single.c
%
% J = movepixels(I,T,mode);
%
% Inputs;
%   T : The transformation images, describing the
%             (backwards) translation of every pixel in x,y and z direction.
%   mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            2: cubic interpolation and outsite pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
%
%
% Outputs,
%   Iout : The transformed image
%
% Function is written by D.Kroon University of Twente (March 2009)

if(~exist('mode','var')), mode=0; end

if(size(I1,3)<4)
    I3=movepixels_2d_double(double(I1),double(T(:,:,1)),double(T(:,:,2)),double(mode));
else
    if(isa(I1,'double'))
        I3=movepixels_3d_double(double(I1),double(T(:,:,:,1)),double(T(:,:,:,2)),double(T(:,:,:,3)),double(mode));
    else
        I3=movepixels_3d_single(single(I1),single(T(:,:,:,1)),single(T(:,:,:,2)),single(T(:,:,:,3)),single(mode));
    end
end
if(~isa(I1,'double')&&~isa(I1,'single'))
    if(isa(I1,'uint8')), I3=uint8(I3); end
    if(isa(I1,'uint16')),I3=uint16(I3); end
    if(isa(I1,'uint32')),I3=uint32(I3); end
    if(isa(I1,'int8')),  I3=int8(I3); end
    if(isa(I1,'int16')), I3=int16(I3); end
    if(isa(I1,'int32')), I3=int32(I3); end
end


