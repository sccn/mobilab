function [e,egrad]=affine_registration_error(par,scale,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode)
% This function affine_registration_error, uses affine transfomation of the
% 3D input volume and calculates the registration error after transformation.
%
% [e,egrad]=affine_registration_error(parameters,scale,I1,I2,type,Grid,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode);
%
% input,
%   parameters (in 2D) : Rigid vector of length 3 -> [translateX translateY rotate]
%                        or Affine vector of length 7 -> [translateX translateY
%                                           rotate resizeX resizeY shearXY shearYX]
%
%   parameters (in 3D) : Rigid vector of length 6 : [translateX translateY translateZ
%                                           rotateX rotateY rotateZ]
%                       or Affine vector of length 15 : [translateX translateY translateZ,
%                             rotateX rotateY rotateZ resizeX resizeY
%                             resizeZ,
%                             shearXY, shearXZ, shearYX, shearYZ, shearZX, shearZY]
%
%   scale: Vector with Scaling of the input parameters with the same lenght
%               as the parameter vector.
%   I1: The 2D/3D image which is rigid or affine transformed
%   I2: The second 2D/3D image which is used to calculate the
%       registration error
%   type: The type of registration error used see registration_error.m
% (optional)
%   Grid: B-spline controlpoints grid. When defined, the Grid describing the
%       nonrigid-transformation is rigid transformed and then used to
%       nonrigid-transform the image or volume.
%   Spacing: The spacing of the B-spline grid.
%   MaskI1: Image/volume which affine transformed in the same way as I1 and
%           is multiplied with the individual pixel errors
%         before calculation of the te total (mean) similarity error.
%   MaskI2: Also a Mask but is used  for I2
%   Points1: List N x 2 of landmarks x,y in Imoving image
%   Points2: List N x 2 of landmarks x,y in Istatic image, in which
%                     every row correspond to the same row with landmarks
%                     in Points1.
%   PStrength: List Nx1 with the error strength used between the
%                     corresponding points, (lower the strenght of the landmarks
%                     if less sure of point correspondence).
%   mode: If 0: linear interpolation and outside pixels set to nearest pixel
%            1: linear interpolation and outside pixels set to zero
%            2: cubic interpolation and outsite pixels set to nearest pixel
%            3: cubic interpolation and outside pixels set to zero
%             
% outputs,
%   e: registration error between I1 and I2
%   egrad: error gradient of input parameters
% example,
%   see example_3d_affine.m
%
% Function is written by D.Kroon University of Twente (April 2009)

if(~exist('O_trans','var')), O_trans=[]; end
if(~exist('MaskI1','var')), MaskI1=[]; end
if(~exist('MaskI2','var')), MaskI2=[]; end
if(~exist('Points1','var')), Points1=[]; end
if(~exist('Points2','var')), Points2=[]; end
if(~exist('PStrength','var')), PStrength=[]; end
if(~exist('mode','var')), mode=0; end

% Change mutual information to local mutual information error
if( strcmp( type,'mi')), type='mip'; end

% Scale the inputs
par=par.*scale;

% Delta
delta=1e-5;

% Special case for simple squared difference (speed optimized code)
if((size(I1,3)>3)&&(isempty(O_trans))&&(isempty(MaskI1))&&(strcmp(type,'sd')))
    M=getransformation_matrix(par);
    if(isa(I1,'double'))
        if(nargout>1)
            [e,mgrad]=affine_error_3d_double(double(I1),double(I2),double(M),double(mode));
            Me=[mgrad(1) mgrad(2) mgrad(3) mgrad(4);
                mgrad(5) mgrad(6) mgrad(7) mgrad(8);
                mgrad(9) mgrad(10) mgrad(11) mgrad(12);
                0        0        0         0];
            egrad=zeros(1,length(par));
            for i=1:length(par)
                par2=par;
                par2(i)=par(i)+delta*scale(i);
                Mg=getransformation_matrix(par2);
                diffM=(Mg-M).*Me;
                egrad(i)=sum(diffM(:))/delta;
            end
        else
            e=affine_error_3d_double(double(I1),double(I2),double(M),double(mode));
        end
    else
        if(nargout>1)
            [e,mgrad]=affine_error_3d_single(single(I1),single(I2),single(M),single(mode));
            Me=[mgrad(1) mgrad(2) mgrad(3) mgrad(4);
                mgrad(5) mgrad(6) mgrad(7) mgrad(8);
                mgrad(9) mgrad(10) mgrad(11) mgrad(12);
                0        0        0         0];
            egrad=zeros(1,length(par));
            for i=1:length(par)
                par2=par;
                par2(i)=par(i)+delta*scale(i);
                Mg=getransformation_matrix(par2);
                diffM=(Mg-M).*Me;
                egrad(i)=sum(diffM(:))/delta;
            end
        else
            e=affine_error_3d_single(single(I1),single(I2),single(M),single(mode));
        end
    end
    return;
end

% Normal error calculation between the two images, and error gradient if needed
% by final differences
if(size(I1,3)<4)
    e=affine_registration_error_2d(par,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode);
    if(nargout>1)
        egrad=zeros(1,length(par));
        for i=1:length(par)
            par2=par; par2(i)=par(i)+delta*scale(i);
            egrad(i)=(affine_registration_error_2d(par2,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode)-e)/delta;
        end
    end
else
    e=affine_registration_error_3d(par,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode);
    if(nargout>1)
        egrad=zeros(1,length(par));
        for i=1:length(par)
            par2=par; par2(i)=par(i)+delta*scale(i);
            egrad(i)=(affine_registration_error_3d(par2,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode)-e)/delta;
        end
    end
end
 

function e=affine_registration_error_2d(par,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode)
M=getransformation_matrix(par);
if(isempty(O_trans)),
    % Affine transform the image
    I3=affine_transform(I1,M,mode);
        
    % Transform MaskI1 with the affine transformation.
    if(~isempty(MaskI1)), MaskI1=affine_transform(MaskI1,M); end
    % Calculate a combine masked for excluding some regions from image
    if(~isempty(MaskI1)), if(~isempty(MaskI2)), Mask=MaskI1.*MaskI2; else Mask=MaskI1; end
    elseif(~isempty(MaskI2)), Mask=MaskI2;
    else Mask=[];
    end
    
    
else
    % Calculate center of the image
    mean=size(I1)/2;
    % Make center of the image coordinates 0,0
    xd=O_trans(:,:,1)-mean(1); yd=O_trans(:,:,2)-mean(2);
    % Calculate the rigid transformed coordinates
    O_trans(:,:,1) = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) * 1;
    O_trans(:,:,2) = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) * 1;
    
    % Perform the affine transformation
    I3=bspline_transform(O_trans,I1,Spacing);
    
    % Calculate a combine masked for excluding some regions from image
    % Transform MaskI1 with the affine transformation.
    if(~isempty(MaskI1)), MaskI1=bspline_transform(O_trans,MaskI1,Spacing);  end
    % Calculate a combine masked for excluding some regions from image
    if(~isempty(MaskI1)), if(~isempty(MaskI2)), Mask=MaskI1.*MaskI2; else Mask=MaskI1; end
    elseif(~isempty(MaskI2)), Mask=MaskI2;
    else Mask=[];
    end
    
end

% registration error calculation.
e = image_difference(I3,I2,type,Mask);

if(~isempty( Points1))
    % Add landmark error
    e=e+affine_point_error(M,I1,Points1, Points2, PStrength);
end

function e=affine_registration_error_3d(par,I1,I2,type,O_trans,Spacing,MaskI1,MaskI2,Points1,Points2,PStrength,mode)
M=getransformation_matrix(par);

if(isempty(O_trans)),
    % Calculate transformed image volume, or in case of squared
    % difference and no mask directly the error
    I3=affine_transform(I1,M,mode);
    
    % Transform MaskI1 with the affine transformation.
    if(~isempty(MaskI1)), MaskI1=affine_transform(MaskI1,M); end
    % Calculate a combine masked for excluding some regions from image
    if(~isempty(MaskI1)), if(~isempty(MaskI2)), Mask=MaskI1.*MaskI2; else Mask=MaskI1; end
    elseif(~isempty(MaskI2)), Mask=MaskI2;
    else Mask=[];
    end
else
    % Calculate center of the image
    mean=size(I1)/2;
    % Make center of the image coordinates 0,0
    xd=O_trans(:,:,:,1)-mean(1); yd=O_trans(:,:,:,2)-mean(2); zd=O_trans(:,:,:,3)-mean(3);
    % Calculate the rigid transformed coordinates
    O_trans(:,:,:,1) = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) *zd + M(1,4)* 1;
    O_trans(:,:,:,2) = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) *zd + M(2,4)* 1;
    O_trans(:,:,:,3) = mean(3) + M(3,1) * xd + M(3,2) *yd + M(3,3) *zd + M(3,4)* 1;
    
    % Perform the nonrigid transformation
    I3=bspline_transform(O_trans,I1,Spacing);
    
    % Calculate a combine masked for excluding some regions from image
    % Transform MaskI1 with the affine transformation.
    if(~isempty(MaskI1)), MaskI1=bspline_transform(O_trans,MaskI1,Spacing);  end
    % Calculate a combine masked for excluding some regions from image
    if(~isempty(MaskI1)), if(~isempty(MaskI2)), Mask=MaskI1.*MaskI2; else Mask=MaskI1; end
    elseif(~isempty(MaskI2)), Mask=MaskI2;
    else Mask=[];
    end
end

% registration error calculation.
e = image_difference(I3,I2,type,Mask);

if(~isempty( Points1))
    % Add landmark error
    e=e+affine_point_error(M,I1,Points1, Points2, PStrength);
end

function M=getransformation_matrix(par)
switch(length(par))
    case 6  %3d
        M=make_transformation_matrix(par(1:3),par(4:6));
    case 9  %3d
        M=make_transformation_matrix(par(1:3),par(4:6),par(7:9));
    case 15 %3d
        M=make_transformation_matrix(par(1:3),par(4:6),par(7:9),par(10:15));
    case 3 % 2d
        M=make_transformation_matrix(par(1:2),par(3));
    case 5 % 2d
        M=make_transformation_matrix(par(1:2),par(3),par(4:5));
    case 7 % 2d
        M=make_transformation_matrix(par(1:2),par(3),par(4:5),par(6:7));
end


function errordis=affine_point_error(M,I,Points1, Points2, PStrength)
% This function AFFINE_POINT_ERROR, calculates the squared distance
% between two lists of corresponding points.
%
% The squared distance is normalized by dividing with (width^2+height^2+depth^2),
% and error at every point is distance x Pstrength. All point errors
% are added together to one error value.
%
% error=affine_point_error(M,I,Points1, Points2, PStrength);
%

if(size(I,3)<4)
    % The points
    x1=Points2(:,1); y1=Points2(:,2);
    
    % Calculate center of the image
    mean=size(I)/2;
    
    % Make center of the image coordinates 0,0
    xd=x1-mean(1);
    yd=y1-mean(2);
    
    % Calculate the Transformed coordinates
    x_trans = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) * 1;
    y_trans = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) * 1;
    
    % Normalized distance between transformed and static points
    x2=Points1(:,1); y2=Points1(:,2);
    distance=((x_trans-x2).^2+(y_trans-y2).^2)/(size(I,1).^2+size(I,2).^2);
    
    % The total distance point error, weighted with point strength.
    errordis=sum(distance.*PStrength);
else
    % The points
    x1=Points2(:,1); y1=Points2(:,2); z1=Points2(:,3);
    
    % Calculate center of the image
    mean=size(I)/2;
    
    % Make center of the image coordinates 0,0
    xd=x1-mean(1);
    yd=y1-mean(2);
    zd=z1-mean(3);
    
    
    % Calculate the Transformed coordinates
    x_trans = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) *zd + M(1,4) * 1;
    y_trans = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) *zd + M(2,4) * 1;
    z_trans = mean(3) + M(3,1) * xd + M(3,2) *yd + M(3,3) *zd + M(3,4) * 1;
    
    % Normalized distance between transformed and static points
    x2=Points1(:,1); y2=Points1(:,2); z2=Points1(:,3);
    
    distance=((x_trans-x2).^2+(y_trans-y2).^2+(z_trans-z2).^2)/(size(I,1).^2+size(I,2).^2+size(I,3).^2);
    
    % The total distance point error, weighted with point strength.
    errordis=sum(distance.*PStrength);
end





