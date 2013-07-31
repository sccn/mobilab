function [O_error, O_grad]=jacobiandet_error_2d_double(Ox,Oy,sizeI,dx,dy)
% Gradient and error of the Jacobian of the  transformation grid 
% 
% [O_error, O_grad]=jacobiandet_error_2d_double(Ox,Oy,sizeI,dx,dy)
%
% Inputs,
%   Ox, Oy : are the grid points coordinates
%   Isize : The size of the input image [m n] or in 3d [m n k]
%   dx and dy :  are the spacing of the b-spline knots
% 
% Outputs,
%   O_error : abs(log(max(D,eps))./max(D,eps)); with D the determinant
%               of the Jacobian
%   O_grad : Error Gradient of the control points


step=0.001;
O_grid(:,:,1)=Ox; O_grid(:,:,2)=Oy;
Spacing=[dx dy];

C_init=jacobiandet_transform_2d_double(O_grid(:,:,1),O_grid(:,:,2),sizeI,Spacing(1),Spacing(2));
C_init=abs(log(max(C_init,eps))./max(C_init,eps));
O_error=mean(C_init(:));
O_uniform=make_init_grid(Spacing,sizeI);
    
if(nargout>1)
    O_grad=zeros(size(O_grid));
    for zi=0:3,
        for zj=0:3,
            % The variables which will contain the controlpoints for
            % determining a central registration error gradient
            O_gradpx=O_grid; O_gradpy=O_grid;

            %Set grid movements of every fourth grid node.
            for i=(1+zi):4:size(O_grid,1),
                for j=(1+zj):4:size(O_grid,2),
                    O_gradpx(i,j,1)=O_gradpx(i,j,1)+step;
                    O_gradpy(i,j,2)=O_gradpy(i,j,2)+step;
                end
            end

            % Do the grid b-spline transformation for movement of nodes to
            % left right top and bottem.
            C_gradpx=jacobiandet_transform_2d_double(O_gradpx(:,:,1),O_gradpx(:,:,2),sizeI,Spacing(1),Spacing(2));
            C_gradpy=jacobiandet_transform_2d_double(O_gradpy(:,:,1),O_gradpy(:,:,2),sizeI,Spacing(1),Spacing(2));
            C_gradpx=abs(log(max(C_gradpx,eps))./max(C_gradpx,eps));
            C_gradpy=abs(log(max(C_gradpy,eps))./max(C_gradpy,eps));

            for i=(1+zi):4:size(O_grid,1),
                for j=(1+zj):4:size(O_grid,2),

                    % Calculate pixel region influenced by a grid node
                    [regAx,regAy,regBx,regBy]=regioninfluenced2D(i,j,O_uniform,sizeI);
                    sr=(regBx-regAx+1)*(regBy-regAy+1);
                    % Determine the registration error in the region
                    E_grid=C_init(regAx:regBx,regAy:regBy);
                    E_gradpx=C_gradpx(regAx:regBx,regAy:regBy)-E_grid;
                    E_gradpy=C_gradpy(regAx:regBx,regAy:regBy)-E_grid;

                    O_grad(i,j,1)=sum(E_gradpx(:))/(step*sr);
                    O_grad(i,j,2)=sum(E_gradpy(:))/(step*sr);
                end
            end
        end
    end
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
