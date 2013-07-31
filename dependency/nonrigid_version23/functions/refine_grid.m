function [O_new,Spacing]=refine_grid(O_trans,Spacing,sizeI)
% Refine image transformation grid of 1D b-splines with use of spliting matrix
%                 
% Msplit=(1/8)*[4 4 0 0;
%               1 6 1 0;
%               0 4 4 0;
%               0 1 6 1;
%               0 0 4 4];
%
%     [O_new,Spacing] = refine_grid(O_trans,Spacing,sizeI)
%
% Function is written by D.Kroon University of Twente (August 2010)

Spacing=Spacing/2;

if(ndims(O_trans)==3)
    % Refine B-spline grid in the x-direction
    O_newA=zeros(((size(O_trans,1)-2)*2-1)+2,size(O_trans,2),2);
    i=1:size(O_trans,1)-3; j=1:size(O_trans,2); h=1:2;
    [I,J,H]=ndgrid(i,j,h); 
    I=I(:); J=J(:); H=H(:);
    
    ind=sub2ind(size(O_trans),I,J,H);
    P0=O_trans(ind); 
    P1=O_trans(ind+1);
    P2=O_trans(ind+2); 
    P3=O_trans(ind+3);
    Pnew=split_knots(P0,P1,P2,P3);
     
    ind=sub2ind(size(O_newA),1+((I-1)*2),J,H);
    O_newA(ind)=Pnew(:,1);
    O_newA(ind+1)=Pnew(:,2);
    O_newA(ind+2)=Pnew(:,3);
    O_newA(ind+3)=Pnew(:,4);
    O_newA(ind+4)=Pnew(:,5);

    % Refine B-spline grid in the y-direction
    O_newB=zeros(size(O_newA,1),((size(O_newA,2)-2)*2-1)+2,2);
    i=1:size(O_newA,2)-3; j=1:size(O_newA,1); h=1:2;
    [J,I,H]=ndgrid(j,i,h); 
    I=I(:); J=J(:); H=H(:);
    
    ind=sub2ind(size(O_newA),J,I,H);
    P0=O_newA(ind);
    P1=O_newA(ind+size(O_newA,1)); 
    P2=O_newA(ind+size(O_newA,1)*2); 
    P3=O_newA(ind+size(O_newA,1)*3);
    Pnew=split_knots(P0,P1,P2,P3);
   
    ind=sub2ind(size(O_newB),J,1+((I-1)*2),H);
    O_newB(ind                 )=Pnew(:,1);
    O_newB(ind+  size(O_newB,1))=Pnew(:,2);
    O_newB(ind+2*size(O_newB,1))=Pnew(:,3);
    O_newB(ind+3*size(O_newB,1))=Pnew(:,4);
    O_newB(ind+4*size(O_newB,1))=Pnew(:,5);
    
    % Set the final refined matrix
    O_new=O_newB;
    
    % Make sure a new uniform grid will have the same dimensions (crop)
    dx=Spacing(1); dy=Spacing(2);
    X=ndgrid(-dx:dx:(sizeI(1)+(dx*2)),-dy:dy:(sizeI(2)+(dy*2)));
    O_new=O_new(1:size(X,1),1:size(X,2),1:2);
else
    % Refine B-spline grid in the x-direction
    O_newA=zeros(((size(O_trans,1)-2)*2-1)+2,size(O_trans,2),size(O_trans,3),3);
    k=1:size(O_trans,3); j=1:size(O_trans,2); i=1:size(O_trans,1)-3; h=1:3;
    [I,J,K,H]=ndgrid(i,j,k,h); 
    I=I(:); J=J(:); K=K(:); H=H(:);
    
    ind=sub2ind(size(O_trans),I,J,K,H);
    P0=O_trans(ind); 
    P1=O_trans(ind+1);
    P2=O_trans(ind+2); 
    P3=O_trans(ind+3);
    
    Pnew=split_knots(P0,P1,P2,P3);
    
    ind=sub2ind(size(O_newA),1+((I-1)*2),J,K,H);
    O_newA(ind)=Pnew(:,1);
    O_newA(ind+1)=Pnew(:,2);
    O_newA(ind+2)=Pnew(:,3);
    O_newA(ind+3)=Pnew(:,4);
    O_newA(ind+4)=Pnew(:,5);

    % Refine B-spline grid in the y-direction
    O_newB=zeros(size(O_newA,1),((size(O_newA,2)-2)*2-1)+2,size(O_newA,3),3);
    k=1:size(O_newA,3); j=1:size(O_newA,1); i=1:size(O_newA,2)-3; h=1:3;
    [J,I,K,H]=ndgrid(j,i,k,h); 
    I=I(:); J=J(:); K=K(:); H=H(:);
    
    ind=sub2ind(size(O_newA),J,I,K,H);
    P0=O_newA(ind);
    P1=O_newA(ind+size(O_newA,1)); 
    P2=O_newA(ind+size(O_newA,1)*2); 
    P3=O_newA(ind+size(O_newA,1)*3);
    Pnew=split_knots(P0,P1,P2,P3);
   
    ind=sub2ind(size(O_newB),J,1+((I-1)*2),K,H);
    O_newB(ind                 )=Pnew(:,1);
    O_newB(ind+  size(O_newB,1))=Pnew(:,2);
    O_newB(ind+2*size(O_newB,1))=Pnew(:,3);
    O_newB(ind+3*size(O_newB,1))=Pnew(:,4);
    O_newB(ind+4*size(O_newB,1))=Pnew(:,5);
    
    % Refine B-spline grid in the z-direction
    O_newC=zeros(size(O_newB,1),size(O_newB,2),((size(O_newB,3)-2)*2-1)+2,3);
    k=1:size(O_newB,2); j=1:size(O_newB,1); i=1:size(O_newB,3)-3; h=1:3;
    [J,K,I,H]=ndgrid(j,k,i,h); 
    I=I(:); J=J(:); K=K(:); H=H(:);
    
    ind=sub2ind(size(O_newB),J,K,I,H);
    a=size(O_newB,1)*size(O_newB,2);
    P0=O_newB(ind);
    P1=O_newB(ind+a); 
    P2=O_newB(ind+a*2); 
    P3=O_newB(ind+a*3);
    Pnew=split_knots(P0,P1,P2,P3);
    
    ind=sub2ind(size(O_newC),J,K,1+((I-1)*2),H);
    a=size(O_newC,1)*size(O_newC,2);
    O_newC(ind    )=Pnew(:,1);
    O_newC(ind+  a)=Pnew(:,2);
    O_newC(ind+2*a)=Pnew(:,3);
    O_newC(ind+3*a)=Pnew(:,4);
    O_newC(ind+4*a)=Pnew(:,5);
    
    % Set the final refined matrix
    O_new=O_newC;
    
    % Make sure a new uniform grid will have the same dimensions (crop)
    dx=Spacing(1); dy=Spacing(2); dz=Spacing(3);
    X=ndgrid(-dx:dx:(sizeI(1)+(dx*2)),-dy:dy:(sizeI(2)+(dy*2)),-dz:dz:(sizeI(3)+(dz*2)));
    O_new=O_new(1:size(X,1),1:size(X,2),1:size(X,3),1:3);
end

function Pnew=split_knots(P0,P1,P2,P3)
Pnew(:,1)=(4/8)*(P0+P1);
Pnew(:,2)=(1/8)*(P0+6*P1+P2);
Pnew(:,3)=(4/8)*(P1+P2);
Pnew(:,4)=(1/8)*(P1+6*P2+P3);
Pnew(:,5)=(4/8)*(P2+P3);


          
