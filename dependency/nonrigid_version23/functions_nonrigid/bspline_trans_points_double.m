function [Tlocal,Dlocal]=bspline_trans_points_double(O_trans,Spacing,X,check)
% If Spacing and points are not an integer, we can not use fast look up tables,
% but have to calculate the bspline coefficients for every point
if(nargin<4)
    check = any(mod(Spacing,1)>0)|any(mod(X(:),1)>0);
end
switch size(X,2)
    case 2,
        if(check)
            Tlocal=bspline_transform_slow_2d(O_trans,Spacing,X);
        else
            if(nargout>1)
                [Tlocal,Dlocal]=bspline_transform_fast_2d(O_trans,Spacing,X);
            else
                Tlocal=bspline_transform_fast_2d(O_trans,Spacing,X);
            end
        end
    case 3,
        Tlocal=bspline_transform_slow_3d(O_trans,Spacing,X);
end

function [Tlocal,Dlocal]=bspline_transform_fast_2d(O_trans,Spacing,X)
% Make row vectors of input coordinates
x2=X(:,1); y2=X(:,2);

% Make polynomial look up tables
Bu=zeros(4,Spacing(1));
Bv=zeros(4,Spacing(2));
Bdu=zeros(4,Spacing(1));
Bdv=zeros(4,Spacing(2));

x=0:Spacing(1)-1;
u=(x/Spacing(1))-floor(x/Spacing(1));
Bu(0*Spacing(1)+x+1) = (1-u).^3/6;
Bu(1*Spacing(1)+x+1) = ( 3*u.^3 - 6*u.^2 + 4)/6;
Bu(2*Spacing(1)+x+1) = (-3*u.^3 + 3*u.^2 + 3*u + 1)/6;
Bu(3*Spacing(1)+x+1) = u.^3/6;

y=0:Spacing(2)-1;
v=(y/Spacing(2))-floor(y/Spacing(2));
Bv(0*Spacing(2)+y+1) = (1-v).^3/6;
Bv(1*Spacing(2)+y+1) = ( 3*v.^3 - 6*v.^2 + 4)/6;
Bv(2*Spacing(2)+y+1) = (-3*v.^3 + 3*v.^2 + 3*v + 1)/6;
Bv(3*Spacing(2)+y+1) = v.^3/6;

if(nargout>1)
    Bdu(0*Spacing(1)+x+1) = -(u - 1).^2/2;
    Bdu(1*Spacing(1)+x+1) = (3*u.^2)/2 - 2*u;
    Bdu(2*Spacing(1)+x+1) = -(3*u.^2)/2 + u + 1/2;
    Bdu(3*Spacing(1)+x+1) = u.^2/2;
    
    Bdv(0*Spacing(2)+y+1) = -(v - 1).^2/2;
    Bdv(1*Spacing(2)+y+1) = (3*v.^2)/2 - 2*v;
    Bdv(2*Spacing(2)+y+1) = -(3*v.^2)/2 + v + 1/2;
    Bdv(3*Spacing(2)+y+1) = v.^2/2;
    Bdu=Bdu./Spacing(1);
    Bdv=Bdv./Spacing(2);
end

% Calculate the indexes need to loop up the B-spline values.
u_index=mod(x2,Spacing(1));
v_index=mod(y2,Spacing(2));

i=floor(x2/Spacing(1)); % (first row outside image against boundary artefacts)
j=floor(y2/Spacing(2));

% This part calculates the coordinates of the pixel
% which will be transformed to the current x,y pixel.

Ox=O_trans(:,:,1);
Oy=O_trans(:,:,2);
Tlocalx=0; Tlocaly=0;
Tlocaldxx=0;
Tlocaldyy=0;
Tlocaldxy=0;
Tlocaldyx=0;

a=zeros(size(X,1),4);
b=zeros(size(X,1),4);
if(nargout>1)
    ad=zeros(size(X,1),4);
    bd=zeros(size(X,1),4);
end

IndexO1l=zeros(size(X,1),4);
IndexO2l=zeros(size(X,1),4);
Check_bound1=false(size(X,1),4);
Check_bound2=false(size(X,1),4);

for r=0:3,
    a(:,r+1)=Bu(r*Spacing(1)+u_index(:)+1);
    b(:,r+1)=Bv(r*Spacing(2)+v_index(:)+1);
    if(nargout>1)
        ad(:,r+1)=Bdu(r*Spacing(1)+u_index(:)+1);
        bd(:,r+1)=Bdv(r*Spacing(2)+v_index(:)+1);
    end
    IndexO1l(:,r+1)=(i+r);
    IndexO2l(:,r+1)=(j+r);
    Check_bound1(:,r+1)=(IndexO1l(:,r+1)<0)|(IndexO1l(:,r+1)>(size(O_trans,1)-1));
    Check_bound2(:,r+1)=(IndexO2l(:,r+1)<0)|(IndexO2l(:,r+1)>(size(O_trans,2)-1));
end

for l=0:3,
    for m=0:3,
        IndexO1= IndexO1l(:,l+1);
        IndexO2= IndexO2l(:,m+1);
        Check_bound=Check_bound1(:,l+1)|Check_bound1(:,m+1);
        IndexO1(Check_bound)=1;
        IndexO2(Check_bound)=1;
        Check_bound_inv=double(~Check_bound);
        
        ab=a(:,l+1).*b(:,m+1);
        if(nargout>1)
            abx=ad(:,l+1).*b(:,m+1);
            aby=a(:,l+1).*bd(:,m+1);
        end
        
        c=Ox(IndexO1(:)+IndexO2(:)*size(Ox,1)+1);
        Tlocalx=Tlocalx+Check_bound_inv(:).*ab.*c;
        if(nargout>1)
            Tlocaldxx=Tlocaldxx+Check_bound_inv(:).*abx.*c;
            Tlocaldxy=Tlocaldxy+Check_bound_inv(:).*aby.*c;
        end
        
        c=Oy(IndexO1(:)+IndexO2(:)*size(Oy,1)+1);
        Tlocaly=Tlocaly+Check_bound_inv(:).*ab.*c;
        if(nargout>1)
            Tlocaldyx=Tlocaldyx+Check_bound_inv(:).*abx.*c;
            Tlocaldyy=Tlocaldyy+Check_bound_inv(:).*aby.*c;
        end
    end
end
Tlocal(:,1)=Tlocalx(:);
Tlocal(:,2)=Tlocaly(:);
if(nargout>1)
    Dlocal=(Tlocaldxx).*(Tlocaldyy)-Tlocaldyx.*Tlocaldxy;
end

function Tlocal=bspline_transform_slow_2d(O_trans,Spacing,X)

% Make row vectors of input coordinates
x2=X(:,1); y2=X(:,2);

% This code calculates for every coordinate in X, the indices of all
% b-spline knots which have influence on the transformation value of
% this point
[m,l]=ndgrid(0:3,0:3); m=m(:)'; l=l(:)';
ixs=floor(x2/Spacing(1));
iys=floor(y2/Spacing(2));
ix=repmat(ixs,[1 16])+repmat(m,[length(x2) 1]); ix=ix(:);
iy=repmat(iys,[1 16])+repmat(l,[length(y2) 1]); iy=iy(:);

% Size of the b-spline grid
s=size(O_trans);

% Points outside the bspline grid are set to the upper corner
Check_bound=(ix<0)|(ix>(s(1)-1))|(iy<0)|(iy>(s(2)-1));
ix(Check_bound)=1; iy(Check_bound)=1;
Check_bound_inv=double(~Check_bound);

% Look up the b-spline knot values in neighborhood of the points in (x2,y2)
Cx=O_trans(ix+iy*s(1)+1).*Check_bound_inv;
Cx=reshape(Cx,[length(x2) 16]);
Cy=O_trans(ix+iy*s(1)+s(1)*s(2)+1).*Check_bound_inv;
Cy=reshape(Cy,[length(x2) 16]);

% Calculate the b-spline interpolation constants u,v in the center cell
% range between 0 and 1
v  = (x2-ixs*Spacing(1))/Spacing(1);
u  = (y2-iys*Spacing(2))/Spacing(2);

% Get the b-spline coefficients in amatrix W, which contains
% the influence of all knots on the points in (x2,y2)
W=bspline_coefficients(v,u);

% Calculate the transformation of the points in (x2,y2) by the b-spline grid
Tlocal(:,1)=sum(W.*Cx,2);
Tlocal(:,2)=sum(W.*Cy,2);

function Tlocal=bspline_transform_slow_3d(O_trans,Spacing,X)
% Make row vectors of input coordinates
x2=X(:,1); y2=X(:,2); z2=X(:,3);

% This code calculates for every coordinate in X, the indices of all
% b-spline knots which have influence on the transformation value of
% this point
[m,l,k]=ndgrid(0:3,0:3,0:3); m=m(:)'; l=l(:)'; k=k(:)';
ixs=floor(x2/Spacing(1));
iys=floor(y2/Spacing(2));
izs=floor(z2/Spacing(3));
ix=repmat(floor(ixs),[1 64])+repmat(m,[length(x2) 1]); ix=ix(:);
iy=repmat(floor(iys),[1 64])+repmat(l,[length(y2) 1]); iy=iy(:);
iz=repmat(floor(izs),[1 64])+repmat(k,[length(z2) 1]); iz=iz(:);

% Size of the b-spline grid
s=size(O_trans);

% Points outside the bspline grid are set to the upper corner
Check_bound=(ix<0)|(ix>(s(1)-1))|(iy<0)|(iy>(s(2)-1))|(iz<0)|(iz>(s(3)-1));
ix(Check_bound)=1; iy(Check_bound)=1; iz(Check_bound)=1;
Check_bound_inv=double(~Check_bound);

% Look up the b-spline knot values in neighborhood of the points in (x2,y2)
Cx=O_trans(ix+iy*s(1) +iz*s(1)*s(2) +                    1).*Check_bound_inv;
Cy=O_trans(ix+iy*s(1) +iz*s(1)*s(2) + s(1)*s(2)*s(3)   + 1).*Check_bound_inv;
Cz=O_trans(ix+iy*s(1) +iz*s(1)*s(2) + s(1)*s(2)*s(3)*2 + 1).*Check_bound_inv;
Cx=reshape(Cx,[length(x2) 64]);
Cy=reshape(Cy,[length(x2) 64]);
Cz=reshape(Cz,[length(x2) 64]);

% Calculate the b-spline interpolation constants u,v in the center cell
% range between 0 and 1
v  = (x2-ixs*Spacing(1))/Spacing(1);
u  = (y2-iys*Spacing(2))/Spacing(2);
w  = (z2-izs*Spacing(3))/Spacing(3);

% Get the b-spline coefficients in a matrix W, which contains
% the influence of all knots on the points in (x2,y2)
W=bspline_coefficients(v,u,w);

% Calculate the transformation of the points in (x2,y2) by the b-spline grid
Tlocal(:,1)=sum(W.*Cx,2);
Tlocal(:,2)=sum(W.*Cy,2);
Tlocal(:,3)=sum(W.*Cz,2);