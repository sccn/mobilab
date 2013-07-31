function [O_error, O_grad]=bspline_registration_gradient(O_grid,sizes,Spacing,I1,I2,options,MaskI1,MaskI2,Points1,Points2,PStrength)
% Function registration_gradient. This function will calculate a registration
% error value and gradient after b-spline non-rigid registration
% of two images / volumes.
%
% [error, errorgrad]=bspline_registration_gradient(grid,sizes,Spacing,I1,I2,options,MaskI1,MaskI2)
%
% inputs,
%   grid: b-spline control grid created by make_init_grid.m, (and reshaped
%         to one long vector.)
%   sizes: sizes need tot reshape the grid to matrix format.
%   Spacing: The spacing in x,y (and z) direction of the b-spline grid
%           knots
%   I1 and I2: The input images, of which I1 is transformed.
% (optional)
%   options: Struct with options
%       ->type: Type of image similarity(error) measure used
%               (see image_difference.m) (default 'sd')
%       ->penaltypercentage Percentage of penalty smoothness (bending energy
%               of thin sheet of metal) (default 0.01)
%       ->step: Delta step used for error gradient (default 0.01)
%       ->centralgrad: Calculate derivatives with central instead of forward
%               gradient (default true)
%       ->interpolation Type of interpolation used, linear (default) or
%               cubic.
%       ->scaling: Scaling of dimensions 1x2 or 1x3 (like mm/px), to
%               be used in smoothness penalty
%       ->verbose : Display information (default false)
%   MaskI1: Image/volume which transformed in the same way as I1 and
%           is multiplied with the individual pixel errors
%         before calculation of the te total (mean) similarity error.
%   MaskI2: Also a Mask but is used  for I2
%
% outputs,
%   error: The registration error value
%   errorgrad: The registration error gradient
%
% Note :
%   Control points only influence their neighbor region. Thus when calculating
%   the gradient with finited difference, multiple control points (with a spacing
%   between them of 4) can be moved in one finite difference deformation step, and
%   only the pixels in the neigbourhood of a control point  are used to calculate
%   the finite difference.
%
% example,
%   I1=im2double(imread('lenag1.png'));
%   I2=im2double(imread('lenag2.png'));
%   O_trans=make_init_grid([32 32],size(I1)); sizes=size(O_trans);
%   O_trans = fminsd(@(x)bspline_registration_gradient(x,sizes,[32 32],I1,I2,'sd'), O_trans(:), ...
%               optimset('Display','iter','GradObj','on','MaxIter',20,'DiffMinChange',0.1));
%   Icor=bspline_transform(reshape(O_trans,sizes),I1,Spacing);
%   figure, imshow(I1), figure, imshow(I2), figure, imshow(Icor)
%
%
% This function is written by D.Kroon University of Twente (April 2009)

% Check/set input options
defaultoptions=struct('type','sd', 'penaltypercentage',0.01,'step',0.01,'centralgrad',true,'interpolation','linear','scaling',[1 1 1],'verbose',false);
if(~exist('options','var')),
    options=defaultoptions;
else
    tags = fieldnames(defaultoptions);
    for i=1:length(tags)
        if(~isfield(options,tags{i})),  options.(tags{i})=defaultoptions.(tags{i}); end
    end
    if(length(tags)~=length(fieldnames(options))),
        warning('BsplineRegistrationGradient:unknownoption','unknown options found');
    end
end
% Check if there is any Mask input
if(~exist('MaskI1','var')), MaskI1=[]; end
if(~exist('MaskI2','var')), MaskI2=[]; end
% Check on points input
if(~exist('Points1','var')), Points1=[]; end
if(~exist('Points2','var')), Points2=[]; end
if(~exist('PStrength','var')), PStrength=[]; end

% Type of image similarity(error) measure used
type=options.type;
% Percentage of penalty smoothness (bending energy of thin sheet of metal)
penaltypercentage=options.penaltypercentage;
% Delta step used for error gradient
step=options.step;
% Central gradient or (faster) forward gradient
centralgrad=options.centralgrad;
% Interpolation used linear or cubic
if(strcmpi(options.interpolation(1),'l')), interpolation_mode=0; else interpolation_mode=2; end
% Convert Grid vector to grid matrix
O_grid=reshape(O_grid,sizes);

if(penaltypercentage>0)
    % Calculate penalty smoothness (bending energy of thin sheet of metal)
    if ( nargout > 1 )
        % If penalty gradient needed, also determine the gradient.
        [SO_error, SO_grad]=penalties_smoothness(O_grid,size(I1),options.scaling);
    else
        SO_error=penalties_smoothness(O_grid,size(I1),options.scaling);
    end
end

% Transform MaskI1 with the b-spline transformation.
if(~isempty(MaskI1)), MaskI1=bspline_transform(O_grid,MaskI1,Spacing); end
% Calculate a combine masked for excluding some regions from image
if(~isempty(MaskI1)), if(~isempty(MaskI2)), Mask=MaskI1.*MaskI2; else Mask=MaskI1; end
elseif(~isempty(MaskI2)), Mask=MaskI2;
else Mask=[];
end

% Quick SSD error and error gradient calculation
if(strcmpi(type,'sd')&&isempty(Points1)&&isempty(Mask));
    try
        if ( nargout == 1 )
            if(size(I1,3)<4)
                O_error=bspline_error_2d_double(double(O_grid(:,:,1)),double(O_grid(:,:,2)),double(I1),double(I2),double(Spacing(1)),double(Spacing(2)));
            else
                if(isa(I1,'double'))
                    O_error=bspline_error_3d_double(double(O_grid(:,:,:,1)),double(O_grid(:,:,:,2)),double(O_grid(:,:,:,3)),double(I1),double(I2),double(Spacing(1)),double(Spacing(2)),double(Spacing(3)));
                else
                    O_error=bspline_error_3d_single(single(O_grid(:,:,:,1)),single(O_grid(:,:,:,2)),single(O_grid(:,:,:,3)),single(I1),single(I2),single(Spacing(1)),single(Spacing(2)),single(Spacing(3)));
                end
            end
            % Add penalty to total error
            if(penaltypercentage>0), O_error=O_error+SO_error*penaltypercentage; end
        else
            if(size(I1,3)<4)
                [O_error,O_grad]=bspline_error_2d_double(double(O_grid(:,:,1)),double(O_grid(:,:,2)),double(I1),double(I2),double(Spacing(1)),double(Spacing(2)));
            else
                if(isa(I1,'double'))
                    [O_error,O_grad]=bspline_error_3d_double(double(O_grid(:,:,:,1)),double(O_grid(:,:,:,2)),double(O_grid(:,:,:,3)),double(I1),double(I2),double(Spacing(1)),double(Spacing(2)),double(Spacing(3)));
                else
                    [O_error,O_grad]=bspline_error_3d_single(single(O_grid(:,:,:,1)),single(O_grid(:,:,:,2)),single(O_grid(:,:,:,3)),single(I1),single(I2),single(Spacing(1)),single(Spacing(2)),single(Spacing(3)));
                end
            end
            % Add penalty to total error
            if(options.verbose)
                disp(['Error' num2str(O_error)]);
                disp(['Smoothness Error' num2str(SO_error)]);
                disp(['Abs Mean Error Gradient' num2str(mean(abs(O_grad(:))))]);
                disp(['Abs Max  Error Gradient' num2str(max(abs(O_grad(:))))]);
                disp(['Abs Max  Smoothness Gradient' num2str(max(abs(SO_grad(:))))]);
                disp(['Abs Mean Smoothness Gradient' num2str(mean(abs(SO_grad(:))))]);
            end


            if(penaltypercentage>0), O_error=O_error+SO_error*penaltypercentage;
                O_grad=O_grad+SO_grad*penaltypercentage;
            end
        end
        O_grad(isnan(O_grad))=0;
        return
    catch
    end
end

% Transform the image with the B-spline grid
if(~isempty(Points1))
    % In this case we need the transformation fields to get the movements
    % of the landmarks
    if(size(I1,3)<4)
        [I_init,B]=bspline_transform(O_grid,I1,Spacing,interpolation_mode);
        Bx=B(:,:,1);
        By=B(:,:,2);
    else
        [I_init,B]=bspline_transform(O_grid,I1,Spacing,interpolation_mode);
        Bx=B(:,:,:,1);
        By=B(:,:,:,2);
        Bz=B(:,:,:,3);
    end
else
    Bx=[]; By=[]; Bz=[];
    I_init=bspline_transform(O_grid,I1,Spacing,interpolation_mode);
end

% Calculate the current registration error
if(strcmpi(type,'mi'))
    O_uniform=make_init_grid(Spacing,size(I1));
    O_error=0;
    MaskNum=numel(I1);
    if(size(I1,3)<4)
        for i=1:size(O_uniform,1)
            for j=1:size(O_uniform,2)
                [regAx,regAy,regBx,regBy]=regioninfluenced2D(i,j,O_uniform,size(I1));
                % Mask some regions from the registration error measure
                if(~isempty(Mask)), MaskRegion=Mask(regAx:regBx,regAy:regBy); else MaskRegion=[]; end
                % Determine the registration error in the region
                O_error=O_error+image_difference(I_init(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
            end
        end
    else
        for i=1:size(O_uniform,1)
            for j=1:size(O_uniform,2)
                for k=1:size(O_uniform,3)
                    [regAx,regAy,regAz,regBx,regBy,regBz]=regioninfluenced3D(i,j,k,O_uniform,size(I1));
                    % Mask some regions from the registration error measure
                    if(~isempty(Mask)), MaskRegion=Mask(regAx:regBx,regAy:regBy,regAz:regBz); else MaskRegion=[]; end
                    % Determine the registration error in the region
                    O_error=O_error+image_difference(I_init(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                end
            end
        end
    end
    O_error=O_error/numel(O_grid);
else
    O_error=image_difference(I_init,I2,type,Mask);
end

if(options.verbose&&(nargout > 1 ))
    disp(['Error' num2str(O_error)]);
    disp(['Smoothness Error' num2str(SO_error)]);
end

% Add penalty to total error
if(penaltypercentage>0), O_error=O_error+SO_error*penaltypercentage; end

% Add the Point error
if(~isempty(Points1))
    reg_start=[1 1 1]; reg_end=size(I1);
    if(size(I1,3)<4)
        O_error=O_error+point_error(I1,reg_start,reg_end,Points1,Points2,PStrength,Bx,By);
    else
        O_error=O_error+point_error(I1,reg_start,reg_end,Points1,Points2,PStrength,Bx,By,Bz);
    end
end

%
% Code below is only needed, when the error gradient is asked by the optimizer
%

% If gradient needed, also determine the gradient.
if ( nargout > 1 )
    if(size(I1,3)<4)
        O_grad=error_gradient2d(Spacing,I1,I2,I_init,O_grid,Mask,Points1,Points2,PStrength,Bx,By,interpolation_mode,step,centralgrad,type);
    else
        O_grad=error_gradient3d(Spacing,I1,I2,I_init,O_grid,Mask,Points1,Points2,PStrength,Bx,By,Bz,interpolation_mode,step,centralgrad,type);
    end
    
    if(strcmpi(type,'mi'))
        O_grad=O_grad/numel(O_grad);
    end
    
    if(options.verbose)
        disp(['Abs Mean Error Gradient' num2str(mean(abs(O_grad(:))))]);
        disp(['Abs Max  Error Gradient' num2str(max(abs(O_grad(:))))]);
        disp(['Abs Max  Smoothness Gradient' num2str(max(abs(SO_grad(:))))]);
        disp(['Abs Mean Smoothness Gradient' num2str(mean(abs(SO_grad(:))))]);
    end

    % Add smoothness penalty gradient
    if(penaltypercentage>0), O_grad=O_grad+SO_grad*penaltypercentage; end
    O_grad=O_grad(:);
end

function [regAx,regAy,regBx,regBy]=regioninfluenced2D(i,j,O_uniform,sizeI)
% Calculate pixel region influenced by a grid node
irm=i-2; irp=i+2;
jrm=j-2; jrp=j+2;
irm=max(irm,1); jrm=max(jrm,1);
irp=min(irp,size(O_uniform,1)); jrp=min(jrp,size(O_uniform,2));

regAx=O_uniform(irm,jrm,1); regAy=O_uniform(irm,jrm,2);
regBx=O_uniform(irp,jrp,1); regBy=O_uniform(irp,jrp,2);

if(regAx>regBx), regAxt=regAx; regAx=regBx; regBx=regAxt; end
if(regAy>regBy), regAyt=regAy; regAy=regBy; regBy=regAyt; end

regAx=max(regAx,1); regAy=max(regAy,1);
regBx=max(regBx,1); regBy=max(regBy,1);
regAx=min(regAx,sizeI(1)); regAy=min(regAy,sizeI(2));
regBx=min(regBx,sizeI(1)); regBy=min(regBy,sizeI(2));

function [regAx,regAy,regAz,regBx,regBy,regBz]=regioninfluenced3D(i,j,k,O_uniform,sizeI)
% Calculate pixel region influenced by a grid node
irm=i-2; irp=i+2;
jrm=j-2; jrp=j+2;
krm=k-2; krp=k+2;
irm=max(irm,1); jrm=max(jrm,1); krm=max(krm,1);
irp=min(irp,size(O_uniform,1)); jrp=min(jrp,size(O_uniform,2)); krp=min(krp,size(O_uniform,3));

regAx=O_uniform(irm,jrm,krm,1);
regAy=O_uniform(irm,jrm,krm,2);
regAz=O_uniform(irm,jrm,krm,3);
regBx=O_uniform(irp,jrp,krp,1);
regBy=O_uniform(irp,jrp,krp,2);
regBz=O_uniform(irp,jrp,krp,3);

if(regAx>regBx), regAxt=regAx; regAx=regBx; regBx=regAxt; end
if(regAy>regBy), regAyt=regAy; regAy=regBy; regBy=regAyt; end
if(regAz>regBz), regAzt=regAz; regAz=regBz; regBz=regAzt; end

regAx=max(regAx,1); regAy=max(regAy,1); regAz=max(regAz,1);
regBx=max(regBx,1); regBy=max(regBy,1); regBz=max(regBz,1);
regAx=min(regAx,sizeI(1)); regAy=min(regAy,sizeI(2)); regAz=min(regAz,sizeI(3));
regBx=min(regBx,sizeI(1)); regBy=min(regBy,sizeI(2)); regBz=min(regBz,sizeI(3));



function O_grad=error_gradient2d(Spacing,I1,I2,I_init,O_grid,Mask,Points1,Points2,PStrength,Bx,By,interpolation_mode,step,centralgrad,type)
O_uniform=make_init_grid(Spacing,size(I1));
O_grad=zeros(size(O_grid));
for zi=0:3,
    for zj=0:3,
        % The variables which will contain the controlpoints for
        % determining a central registration error gradient
        O_gradpx=O_grid; O_gradpy=O_grid;
        if(centralgrad)
            O_gradmx=O_grid; O_gradmy=O_grid;
        end
        
        %Set grid movements of every fourth grid node.
        for i=(1+zi):4:size(O_grid,1),
            for j=(1+zj):4:size(O_grid,2),
                O_gradpx(i,j,1)=O_gradpx(i,j,1)+step;
                O_gradpy(i,j,2)=O_gradpy(i,j,2)+step;
                if(centralgrad)
                    O_gradmx(i,j,1)=O_gradmx(i,j,1)-step;
                    O_gradmy(i,j,2)=O_gradmy(i,j,2)-step;
                end
            end
        end
        
        % Do the grid b-spline transformation for movement of nodes to
        % left right top and bottem.
        if(~isempty(Points1))
            [I_gradpx,B]=bspline_transform(O_gradpx,I1,Spacing,interpolation_mode);
            Bx_px=B(:,:,1); By_px=B(:,:,2);
            [I_gradpy,B]=bspline_transform(O_gradpy,I1,Spacing,interpolation_mode);
            Bx_py=B(:,:,1); By_py=B(:,:,2);
            if(centralgrad)
                [I_gradmx,B]=bspline_transform(O_gradmx,I1,Spacing,interpolation_mode);
                Bx_mx=B(:,:,1); By_mx=B(:,:,2);
                [I_gradmy,B]=bspline_transform(O_gradmy,I1,Spacing,interpolation_mode);
                Bx_my=B(:,:,1); By_my=B(:,:,2);
            end
        else
            I_gradpx=bspline_transform(O_gradpx,I1,Spacing,interpolation_mode);
            I_gradpy=bspline_transform(O_gradpy,I1,Spacing,interpolation_mode);
            if(centralgrad)
                I_gradmx=bspline_transform(O_gradmx,I1,Spacing,interpolation_mode);
                I_gradmy=bspline_transform(O_gradmy,I1,Spacing,interpolation_mode);
            end
        end
        
        for i=(1+zi):4:size(O_grid,1),
            for j=(1+zj):4:size(O_grid,2),
                
                % Calculate pixel region influenced by a grid node
                [regAx,regAy,regBx,regBy]=regioninfluenced2D(i,j,O_uniform,size(I1));
                MaskNum=numel(I1);
                % Mask some regions from the registration error
                % measure
                if(~isempty(Mask))
                    MaskRegion=Mask(regAx:regBx,regAy:regBy);
                else
                    MaskRegion=[];
                end
                
                % Determine the registration error in the region
                E_gradpx=image_difference(I_gradpx(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
                E_gradpy=image_difference(I_gradpy(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
                if(centralgrad)
                    E_gradmx=image_difference(I_gradmx(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
                    E_gradmy=image_difference(I_gradmy(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
                else
                    E_grid=image_difference(I_init(regAx:regBx,regAy:regBy),I2(regAx:regBx,regAy:regBy),type,MaskRegion,MaskNum);
                end
                
                if(~isempty(Points1))
                    E_gradpx=E_gradpx+point_error(I1,[regAx regAy],[regBx regBy],Points1,Points2,PStrength,Bx_px,By_px);
                    E_gradpy=E_gradpy+point_error(I1,[regAx regAy],[regBx regBy],Points1,Points2,PStrength,Bx_py,By_py);
                    if(centralgrad)
                        E_gradmx=E_gradmx+point_error(I1,[regAx regAy],[regBx regBy],Points1,Points2,PStrength,Bx_mx,By_mx);
                        E_gradmy=E_gradmy+point_error(I1,[regAx regAy],[regBx regBy],Points1,Points2,PStrength,Bx_my,By_my);
                    else
                        E_grid=E_grid+point_error(I1,[regAx regAy],[regBx regBy],Points1,Points2,PStrength,Bx,By);
                    end
                end
                
                
                % Calculate the error registration gradient.
                if(centralgrad)
                    O_grad(i,j,1)=(E_gradpx-E_gradmx)/(step*2);
                    O_grad(i,j,2)=(E_gradpy-E_gradmy)/(step*2);
                else
                    O_grad(i,j,1)=(E_gradpx-E_grid)/step;
                    O_grad(i,j,2)=(E_gradpy-E_grid)/step;
                end
            end
        end
    end
end

function O_grad=error_gradient3d(Spacing,I1,I2,I_init,O_grid,Mask,Points1,Points2,PStrength,Bx,By,Bz,interpolation_mode,step,centralgrad,type)
O_uniform=make_init_grid(Spacing,size(I1));
O_grad=zeros(size(O_grid));
for zi=0:(4-1),
    for zj=0:(4-1),
        for zk=0:(4-1),
            O_gradpx=O_grid; O_gradpy=O_grid; O_gradpz=O_grid;
            if(centralgrad)
                O_gradmx=O_grid; O_gradmy=O_grid; O_gradmz=O_grid;
            end
            
            %Set grid movements of every fourth grid node.
            for i=(1+zi):4:size(O_grid,1),
                for j=(1+zj):4:size(O_grid,2),
                    for k=(1+zk):4:size(O_grid,3),
                        O_gradpx(i,j,k,1)=O_gradpx(i,j,k,1)+step;
                        O_gradpy(i,j,k,2)=O_gradpy(i,j,k,2)+step;
                        O_gradpz(i,j,k,3)=O_gradpz(i,j,k,3)+step;
                        if(centralgrad)
                            O_gradmx(i,j,k,1)=O_gradmx(i,j,k,1)-step;
                            O_gradmy(i,j,k,2)=O_gradmy(i,j,k,2)-step;
                            O_gradmz(i,j,k,3)=O_gradmz(i,j,k,3)-step;
                        end
                    end
                end
            end
            
            % Do the grid b-spline transformation for movement of nodes to
            % left right top and bottem.
            if(~isempty(Points1))
                [I_gradpx,B]=bspline_transform(O_gradpx,I1,Spacing,interpolation_mode);
                Bx_px=B(:,:,:,1); By_px=B(:,:,:,2);  Bz_px=B(:,:,:,3);
                [I_gradpy,B]=bspline_transform(O_gradpy,I1,Spacing,interpolation_mode);
                Bx_py=B(:,:,:,1); By_py=B(:,:,:,2);  Bz_py=B(:,:,:,3);
                [I_gradpz,B]=bspline_transform(O_gradpz,I1,Spacing,interpolation_mode);
                Bx_pz=B(:,:,:,1); By_pz=B(:,:,:,2);  Bz_pz=B(:,:,:,3);
                if(centralgrad)
                    [I_gradmx,B]=bspline_transform(O_gradmx,I1,Spacing,interpolation_mode);
                    Bx_mx=B(:,:,:,1); By_mx=B(:,:,:,2);  Bz_mx=B(:,:,:,3);
                    [I_gradmy,B]=bspline_transform(O_gradmy,I1,Spacing,interpolation_mode);
                    Bx_my=B(:,:,:,1); By_my=B(:,:,:,2);  Bz_my=B(:,:,:,3);
                    [I_gradmz,B]=bspline_transform(O_gradmz,I1,Spacing,interpolation_mode);
                    Bx_mz=B(:,:,:,1); By_mz=B(:,:,:,2);  Bz_mz=B(:,:,:,3);
                end
            else
                I_gradpx=bspline_transform(O_gradpx,I1,Spacing,interpolation_mode);
                I_gradpy=bspline_transform(O_gradpy,I1,Spacing,interpolation_mode);
                I_gradpz=bspline_transform(O_gradpz,I1,Spacing,interpolation_mode);
                if(centralgrad)
                    I_gradmx=bspline_transform(O_gradmx,I1,Spacing,interpolation_mode);
                    I_gradmy=bspline_transform(O_gradmy,I1,Spacing,interpolation_mode);
                    I_gradmz=bspline_transform(O_gradmz,I1,Spacing,interpolation_mode);
                end
            end
            
            for i=(1+zi):4:size(O_grid,1),
                for j=(1+zj):4:size(O_grid,2),
                    for k=(1+zk):4:size(O_grid,3),
                        
                        [regAx,regAy,regAz,regBx,regBy,regBz]=regioninfluenced3D(i,j,k,O_uniform,size(I1));
                        MaskNum=numel(I1); %(regBx-regAx+1) * (regBy-regAy+1) * (regBz-regAz+1);
                        
                        % Mask some regions from the registration error
                        % measure
                        if(~isempty(Mask))
                            MaskRegion=Mask(regAx:regBx,regAy:regBy,regAz:regBz);
                        else
                            MaskRegion=[];
                        end
                        
                        % Determine the registration error in the region
                        E_gradpx=image_difference(I_gradpx(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                        E_gradpy=image_difference(I_gradpy(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                        E_gradpz=image_difference(I_gradpz(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                        if(centralgrad)
                            E_gradmx=image_difference(I_gradmx(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                            E_gradmy=image_difference(I_gradmy(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                            E_gradmz=image_difference(I_gradmz(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                        else
                            E_grid=image_difference(I_init(regAx:regBx,regAy:regBy,regAz:regBz),I2(regAx:regBx,regAy:regBy,regAz:regBz),type,MaskRegion,MaskNum);
                        end
                        
                        % Add Error for distance between
                        % corresponding points
                        if(~isempty(Points1))
                            E_gradpx=E_gradpx+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_px,By_px,Bz_px);
                            E_gradpy=E_gradpy+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_py,By_py,Bz_py);
                            E_gradpz=E_gradpz+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_pz,By_pz,Bz_pz);
                            if(centralgrad)
                                E_gradmx=E_gradmx+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_mx,By_mx,Bz_mx);
                                E_gradmy=E_gradmy+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_my,By_my,Bz_my);
                                E_gradmz=E_gradmz+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx_mz,By_mz,Bz_mz);
                            else
                                E_grid=E_grid+point_error(I1,[regAx regAy regAz],[regBx regBy regBz],Points1,Points2,PStrength,Bx,By,Bz);
                            end
                        end
                                                
                        % Calculate the error registration gradient.
                        if(centralgrad)
                            O_grad(i,j,k,1)=(E_gradpx-E_gradmx)/(step*2);
                            O_grad(i,j,k,2)=(E_gradpy-E_gradmy)/(step*2);
                            O_grad(i,j,k,3)=(E_gradpz-E_gradmz)/(step*2);
                        else
                            O_grad(i,j,k,1)=(E_gradpx-E_grid)/step;
                            O_grad(i,j,k,2)=(E_gradpy-E_grid)/step;
                            O_grad(i,j,k,3)=(E_gradpz-E_grid)/step;
                        end
                    end
                end
            end
        end
    end
end


function errordis=point_error(I,reg_start,reg_end,Points1,Points2,PStrength,Bx,By,Bz)
if(size(I,3)<4)
    % Points must be inside image
    x1=round(Points2(:,1)); y1=round(Points2(:,2));
    
    % Remove points outside asked region
    check=(x1<reg_start(1))|(x1>reg_end(1))|(y1<reg_start(2))|(y1>reg_end(2));
    x1(check)=[]; y1(check)=[];
    
    % Find transformation of points in transformation field
    x_trans=Bx(sub2ind(size(I),x1,y1))+x1;
    y_trans=By(sub2ind(size(I),x1,y1))+y1;
    
    x2=Points1(:,1); y2=Points1(:,2);
    % Remove points outside asked region
    x2(check)=[]; y2(check)=[]; PStrength(check)=[];
    
    % Normalized distance between transformed and static points
    distance=((x_trans-x2).^2+(y_trans-y2).^2)/(size(I,1).^2+size(I,2).^2);
    
    % The total distance point error, weighted with point strength.
    errordis=sum(distance.*PStrength);
else
    % Points must be inside image
    x1=round(Points2(:,1)); y1=round(Points2(:,2));  z1=round(Points2(:,3));
    % Remove points outside asked region
    check=(x1<reg_start(1))|(x1>reg_end(1))|(y1<reg_start(2))|(y1>reg_end(2))|(z1<reg_start(3))|(z1>reg_end(3));
    x1(check)=[]; y1(check)=[]; z1(check)=[];
    
    % Find transformation of points in transformation field
    x_trans=Bx(sub2ind(size(I),x1,y1,z1))+x1;
    y_trans=By(sub2ind(size(I),x1,y1,z1))+y1;
    z_trans=Bz(sub2ind(size(I),x1,y1,z1))+z1;
    
    x2=Points1(:,1); y2=Points1(:,2); z2=Points1(:,3);
    % Remove points outside asked region
    x2(check)=[]; y2(check)=[]; z2(check)=[]; PStrength(check)=[];
    
    % Normalized distance between transformed and static points
    distance=((x_trans-x2).^2+(y_trans-y2).^2+(z_trans-z2).^2)/(size(I,1).^2+size(I,2).^2+size(I,3).^2);
    
    % The total distance point error, weighted with point strength.
    errordis=sum(distance.*PStrength);
end




