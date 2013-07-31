function Igrid=make_grid_image(Spacing,sizeI)
% This function creates a uniform 2d or 3D image of grid
% Igrid=make_grid_image(Spacing,sizeI)
%
% inputs,
% Spacing: Spacing of the b-spline grid knot vector
% sizeI: vector with the sizes of the image which will be transformed
% 
% outputs,
% Igrid: Image of uniform control grid
%
% Function is written by D.Kroon University of Twente (October 2008)

if(length(Spacing)==2)
    O=make_init_grid(Spacing,sizeI);

    % Make an image for showing the grid.
    Igrid=zeros([sizeI(1) sizeI(2)]);
    for i=1:size(O,1),
        px=round(O(i,1,1))+1;
        if(px>0&&px<=sizeI(1)),Igrid(px,:)=1; end
    end
    for i=1:size(O,2),
        py=round(O(1,i,2))+1;
        if(py>0&&py<=sizeI(2)), Igrid(:,py)=1; end
    end
else
    O=make_init_grid(Spacing,sizeI);
        
    % Make an image for showing the grid.
    Igrid=zeros([sizeI(1) sizeI(2) sizeI(3)]);
    for j=1:size(O,2),
        for k=1:size(O,3),
            py=round(O(1,j,k,2))+1;
            pz=round(O(1,j,k,3))+1;
            if(py>0&&py<=sizeI(2)&&pz>0&&pz<=sizeI(3)), Igrid(:,py,pz)=1; end
        end
    end
    for i=1:size(O,1),
        for k=1:size(O,3),
            px=round(O(i,1,k,1))+1;
            pz=round(O(i,1,k,3))+1;
            if(px>0&&px<=sizeI(1)&&pz>0&&pz<=sizeI(3)), Igrid(px,:,pz)=1; end
        end
    end
    for i=1:size(O,1),
        for j=1:size(O,2),
            px=round(O(i,j,1,1))+1;
            py=round(O(i,j,1,2))+1;
            if(px>0&&px<=sizeI(1)&&py>0&&py<=sizeI(2)), Igrid(px,py,:)=1; end
        end
    end
end