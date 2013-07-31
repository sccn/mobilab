function [O_penalty, O_grad]=penalties_smoothness(O,sizeI,Scaling)
% This function calculates the 2D or 3D equivalent of the 2D 
% bending energy of thin sheet of metal, for a 2D or 3D transformation 
% grid of cubic spline control points.
%
% [O_penalty, O_derivative]=penalties_smoothness(O,sizeI,Scaling);
% 
% inputs,
%   sizeI : The dimensions of the image or volume
%   O : The b-spline controlpoint grid
%   Scaling : Scaling of dimensions 1x2 or 1x3 (like mm/px)
%
% outputs,
%   O_penalty : The bending energy penalty for the b-spline control grid.
%   O_derivative : The penalty derivates of the control points.
%
% example,
%
% % Make the Initial b-spline registration grid
% sizeI=[256 256 256]; spacing=[16 16 16];
% O=make_init_grid(spacing,sizeI);
% % Calculate penalty of initial grid.
% [O_penalty, O_grad]=penalties_smoothness(O,sizeI);
%
% Function is written by D.Kroon University of Twente (October 2008)

global penalties;
% Load penalties constants
if(isempty(penalties)), 
    load penalty_matrix; 
end

% Check inputs
if(~exist('Scaling','var')), Scaling=[1 1 1]; end

% Gradient needed as output ?
if ( nargout > 1 ), 
    calgrad=true; 
else
    calgrad=false; 
end

% Norm of image size, to make smoothness penalty (approximately)
% independent of image size
msizeI=sqrt(sum(sizeI.^2));
          
if(length(sizeI)==2||sizeI(3)<4)
    % Make penalty independent of image size
    O(:,:,1)=O(:,:,1)/(msizeI/Scaling(1));
    O(:,:,2)=O(:,:,2)/(msizeI/Scaling(2));

    % Calculate the thin sheet of metal energy
    [PEx,PEgx]=calculate_energy_2d(O(:,:,1),calgrad,penalties.Verror2d,penalties.Vgrad2d);
    [PEy,PEgy]=calculate_energy_2d(O(:,:,2),calgrad,penalties.Verror2d,penalties.Vgrad2d);
    O_penalty=sum(PEx(:))+ sum(PEy(:));
    if(calgrad)
        O_grad=zeros(size(O));
        O_grad(:,:,1)=PEgx/(msizeI/Scaling(1));
        O_grad(:,:,2)=PEgy/(msizeI/Scaling(2));
    end
else
    % Make penalty independent of image size
    O(:,:,:,1)=O(:,:,:,1)/(msizeI/Scaling(1));
    O(:,:,:,2)=O(:,:,:,2)/(msizeI/Scaling(2));
    O(:,:,:,3)=O(:,:,:,3)/(msizeI/Scaling(3));
   
    % Calculate the thin sheet of metal energy
    [PEx,PEgx]=calculate_energy_3d(O(:,:,:,1),calgrad,penalties.Verror3d,penalties.Vgrad3d);
    [PEy,PEgy]=calculate_energy_3d(O(:,:,:,2),calgrad,penalties.Verror3d,penalties.Vgrad3d);
    [PEz,PEgz]=calculate_energy_3d(O(:,:,:,3),calgrad,penalties.Verror3d,penalties.Vgrad3d);

    O_penalty=sum(PEx(:))+ sum(PEy(:))+ sum(PEz(:));
    if(calgrad)
        O_grad=zeros(size(O));
        O_grad(:,:,:,1)=PEgx/(msizeI/Scaling(1));
        O_grad(:,:,:,2)=PEgy/(msizeI/Scaling(2));
        O_grad(:,:,:,3)=PEgz/(msizeI/Scaling(3));
    end
end

% Normalize to compensate grid refinement 
O_penalty=O_penalty*numel(O)/100;
if(calgrad), O_grad=O_grad*numel(O)/100; end

function [PE,PEg]=calculate_energy_2d(O,calgrad,Verror2d,Vgrad2d)
PE=zeros(size(O)-3);
PEg=zeros(size(O));

for i=1:size(O,1)-3;
    for j=1:size(O,2)-3;
       % Get the control points of one grid cell
       P=O(i+(0:3),j+(0:3))'; P=P(:)';
       
       % Calculate the 2D bending energy of thin sheet of metal
       PE(i,j)=sum(sum((P'*P).*Verror2d));
       
       if(calgrad)
           DP=(Vgrad2d*P(:))';
           % Add penalties from this cell to the total penalty gradients of the control points
           PEg(i+(0:3),j+(0:3))=PEg(i+(0:3),j+(0:3))+reshape(DP,[4 4])';
       end
    end
end


function [PE,PEg]=calculate_energy_3d(O,calgrad,Verror3d,Vgrad3d)
PE=zeros(size(O)-3);
PEg=zeros(size(O));
for i=1:size(O,1)-3;
    for j=1:size(O,2)-3;
        for k=1:size(O,3)-3;
            % Get the control points of one grid cell
            P=permute(O(i+(0:3),j+(0:3),k+(0:3)),[3 2 1]); P=P(:)';
      
             % Calculate the 3D equivalent of the 2D bending energy of thin
             % sheet of metal.
               PE(i,j,k)=sum(sum((P'*P).*Verror3d));
            
             if(calgrad)
                 % Calculate all the bending energy control point derivatives.
                 DP=(Vgrad3d*P(:));
                 % Add penalties from this cell to the total penalty gradients of the
                 % control points
                 PEg(i+(0:3),j+(0:3),k+(0:3))=PEg(i+(0:3),j+(0:3),k+(0:3))+permute(reshape(DP,[4 4 4]),[3 2 1]);
            end
        end
    end
end


% This code can be used to test the difference between real derrivatives
% and gradient derrivatives
% 
%     [O_penalty, O_grad]=penalties_smoothness(O,sizeI);
%     step=0.001;
%     O_grad2=zeros(size(O));
%     for i=1:size(O,1)
%         for j=1:size(O,2)
%             for h=1:2
%                 Op=O; Om=O;
%                 Op(i,j,h)=Op(i,j,h)+step;
%                 Om(i,j,h)=Om(i,j,h)-step;
%                 [O_pp]=penalties_smoothness(Op,sizeI);
%                 [O_pm]=penalties_smoothness(Om,sizeI);
%                 O_grad2(i,j,h)=(O_pp-O_pm)/(2*step);
%             end
%         end
%     end
%     sum(abs(O_grad(:)-O_grad2(:)))
% 
% 
%     [O_penalty, O_grad]=penalties_smoothness(O,sizeI);
%     step=0.001;
%     O_grad2=zeros(size(O));
%     for i=1:size(O,1)
%         for j=1:size(O,2)
%             for k=1:size(O,3), 
%               for h=1:3
%                 Op=O; Om=O;
%                 Op(i,j,k,h)=Op(i,j,k,h)+step;
%                 Om(i,j,k,h)=Om(i,j,k,h)-step;
%                 [O_pp]=penalties_smoothness(Op,sizeI);
%                 [O_pm]=penalties_smoothness(Om,sizeI);
%                 O_grad2(i,j,k,h)=(O_pp-O_pm)/(2*step);
%               end;
%            end
%         end
%     end
%     sum(abs(O_grad(:)-O_grad2(:)))
