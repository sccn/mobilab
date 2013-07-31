function O_trans=make_init_grid(Spacing,sizeI,M)
% This function creates a uniform 2d or 3D b-spline control grid
% O=make_init_grid(Spacing,sizeI,M)
% 
%  inputs,
%    Spacing: Spacing of the b-spline grid knot vector
%    sizeI: vector with the sizes of the image which will be transformed
%    M : The (inverse) transformation from rigid registration.
%  
%  outputs,
%    O: Uniform b-splinecontrol grid
%
%  example,
%
%    I1 = im2double(imread('lenag1.png')); 
%    O = make_init_grid([8 8],size(I1));
%
%  Function is written by D.Kroon University of Twente (September 2008)
  
if(length(Spacing)==2)
    % Determine grid spacing
    dx=Spacing(1); dy=Spacing(2);

    % Calculate te grid coordinates (make the grid)
    [X,Y]=ndgrid(-dx:dx:(sizeI(1)+(dx*2)),-dy:dy:(sizeI(2)+(dy*2)));
    O_trans=ones(size(X,1),size(X,2),2);
    O_trans(:,:,1)=X;
    O_trans(:,:,2)=Y;
    
    if(exist('M','var')),
        % Calculate center of the image
        mean=sizeI/2;

        % Make center of the image coordinates 0,0
        xd=O_trans(:,:,1)-mean(1); 
        yd=O_trans(:,:,2)-mean(2);

        % Calculate the rigid transformed coordinates
        O_trans(:,:,1) = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) * 1;
        O_trans(:,:,2) =  mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) * 1;
    end
else
    % Determine grid spacing
    dx=Spacing(1); dy=Spacing(2); dz=Spacing(3);
    
    % Calculate te grid coordinates (make the grid)
    [X,Y,Z]=ndgrid(-dx:dx:(sizeI(1)+(dx*2)),-dy:dy:(sizeI(2)+(dy*2)),-dz:dz:(sizeI(3)+(dz*2)));
    O_trans=ones(size(X,1),size(X,2),size(X,3),3);
    O_trans(:,:,:,1)=X;
    O_trans(:,:,:,2)=Y;
    O_trans(:,:,:,3)=Z;
    
    if(exist('M','var')),
        % Calculate center of the image
        mean=sizeI/2;

        % Make center of the image coordinates 0,0
        xd=O_trans(:,:,:,1)-mean(1); 
        yd=O_trans(:,:,:,2)-mean(2);
        zd=O_trans(:,:,:,3)-mean(3);

        % Calculate the rigid transformed coordinates
        O_trans(:,:,:,1) = mean(1) + M(1,1) * xd + M(1,2) *yd + M(1,3) *zd + M(1,4)* 1;
        O_trans(:,:,:,2) = mean(2) + M(2,1) * xd + M(2,2) *yd + M(2,3) *zd + M(2,4)* 1;
        O_trans(:,:,:,3) = mean(3) + M(3,1) * xd + M(3,2) *yd + M(3,3) *zd + M(3,4)* 1;
    end
end
    
    