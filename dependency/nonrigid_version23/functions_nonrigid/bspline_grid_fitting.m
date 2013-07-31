function O_trans=bspline_grid_fitting(O,Spacing,D,X)
switch size(D,2)
    case 2
        O_trans=bspline_grid_fitting_2d(O,Spacing,D,X);
    case 3
        O_trans=bspline_grid_fitting_3d(O,Spacing,D,X);
    otherwise
        error('bspline_grid_fitting:input','unknown input dimmension');
end

function O_trans=bspline_grid_fitting_2d(O,Spacing,D,X)
% calculate which is the closest point on the lattic to the top-left
% corner and find ratio's of influence between lattice point.
gx  = floor(X(:,1)/Spacing(1)); 
gy  = floor(X(:,2)/Spacing(2)); 

% Calculate b-spline coordinate within b-spline cell, range 0..1
ax  = (X(:,1)-gx*Spacing(1))/Spacing(1);
ay  = (X(:,2)-gy*Spacing(2))/Spacing(2); 

gx=gx+2; gy=gy+2; 

if(any(ax<0)||any(ax>1)||any(ay<0)||any(ay>1)), error('grid error'), end;

W=bspline_coefficients(ax,ay);

% Make indices of all neighborh knots to every point
[ix,iy]=ndgrid(-1:2,-1:2); ix=ix(:); iy=iy(:);
indexx=repmat(gx,[1 16])+repmat(ix',[size(X,1)  1]); indexx=indexx(:);
indexy=repmat(gy,[1 16])+repmat(iy',[size(X,1)  1]); indexy=indexy(:);
% Limit to boundaries grid
indexx=min(max(1,indexx),size(O,1));
indexy=min(max(1,indexy),size(O,2));
index=[indexx indexy];
 
% according too Lee et al. we update a numerator and a denumerator for
% each knot. In our case we need two numerators, because our value is a
% vector dy,dx. If we want to be able to add/remove keypoints, we need 
% to store the numerators in seperate arrays.
W2=W.^2; S = sum(W2,2);
WT=W2.*W;
WNx= WT.*repmat(D(:,1)./S,[1 16]); 
WNy= WT.*repmat(D(:,2)./S,[1 16]); 
siz=[size(O,1) size(O,2)];
numx=accumarray(index, WNx(:),siz);
numy=accumarray(index, WNy(:),siz);
dnum=accumarray(index, W2(:),siz);

% calculate actual values of knots from the numerator and denumerator that
% we calculated previously
ux  = numx ./ (dnum+eps);
uy  = numy ./ (dnum+eps);

% Update the b-spline transformation grid
O_trans(:,:,1)=ux+O(:,:,1);
O_trans(:,:,2)=uy+O(:,:,2);

function O_trans=bspline_grid_fitting_3d(O,Spacing,D,X)
% calculate which is the closest point on the lattic to the top-left
% corner and find ratio's of influence between lattice point.
gx  = floor(X(:,1)/Spacing(1)); 
gy  = floor(X(:,2)/Spacing(2)); 
gz  = floor(X(:,3)/Spacing(3)); 

% Calculate b-spline coordinate within b-spline cell, range 0..1
ax  = (X(:,1)-gx*Spacing(1))/Spacing(1);
ay  = (X(:,2)-gy*Spacing(2))/Spacing(2); 
az  = (X(:,3)-gz*Spacing(3))/Spacing(3); 

gx=gx+2; gy=gy+2; gz=gz+2;

if(any(ax<0)||any(ax>1)||any(ay<0)||any(ay>1)||any(az<0)||any(az>1)),
    error('bspline_grid_fitting:grid','grid error');
end

W=bspline_coefficients(ax,ay,az);
 
% Make indices of all neighborh knots to every point
[ix,iy,iz]=ndgrid(-1:2,-1:2,-1:2); ix=ix(:); iy=iy(:); iz=iz(:);
indexx=repmat(gx,[1 64])+repmat(ix',[size(X,1) 1]); indexx=indexx(:);
indexy=repmat(gy,[1 64])+repmat(iy',[size(X,1) 1]); indexy=indexy(:);
indexz=repmat(gz,[1 64])+repmat(iz',[size(X,1) 1]); indexz=indexz(:);
% Limit to boundaries grid
indexx=min(max(1,indexx),size(O,1));
indexy=min(max(1,indexy),size(O,2));
indexz=min(max(1,indexz),size(O,3));
index=[indexx indexy indexz];
 
% according too Lee et al. we update a numerator and a denumerator for
% each knot. In our case we need two numerators, because our value is a
% vector dy,dx. If we want to be able to add/remove keypoints, we need 
% to store the numerators in seperate arrays.
W2=W.^2; S = sum(W2,2);
WT=W2.*W;
WNx= WT.*repmat(D(:,1)./S,[1 64]); 
WNy= WT.*repmat(D(:,2)./S,[1 64]); 
WNz= WT.*repmat(D(:,3)./S,[1 64]); 
siz=[size(O,1) size(O,2) size(O,3)];
numx=accumarray(index, WNx(:),siz);
numy=accumarray(index, WNy(:),siz);
numz=accumarray(index, WNz(:),siz);
dnum=accumarray(index, W2(:),siz);

% calculate actual values of knots from the numerator and denumerator that
% we calculated previously
ux  = numx ./ (dnum+eps);
uy  = numy ./ (dnum+eps);
uz  = numz ./ (dnum+eps);

% Update the b-spline transformation grid
O_trans(:,:,:,1)=ux+O(:,:,:,1);
O_trans(:,:,:,2)=uy+O(:,:,:,2);
O_trans(:,:,:,3)=uz+O(:,:,:,3);
