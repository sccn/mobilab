function [J,models] = invertParametricEmpiricalBayes(Y,K,Q1,L,eta,Omega,obj,sourceSpace)
if nargin < 5, eta = -4;end
if nargin < 5, Omega = 16;end

[N,~,Nh] = size(Q1);

D = triu(ones(Nh));
ind = find(D);
clear D;
[indI,indJ] = ind2sub([Nh Nh],ind);
% N = 0.5*Nh^2;
Nij = length(indI);
C1 = zeros(Nh);
Pi  = zeros(N,N,Nh);
Pij = zeros(N,N,Nij);
iSl = zeros(Nh);
I1  = zeros(Nh,1);
I2  = zeros(Nh);
%--
% C2 = alpha*L'*L;
% C2 = L'*L;
% sqC2 = L';
%--

%--
% C2 = inv(L'*L);
A = L'*L;
iA = eye(size(L,1))./(A+eps);
sqC2 = chol(iA);
%--

[U,S,V] = svd(K*sqC2,'econ');
s = diag(S);
s2 = s.^2;
US2Ut = U*diag(s2)*U';
Sy = Y*Y';
lambda = random('Normal',eta,sqrt(Omega),Nh,1);
lambda1 = lambda;
lambda2 = random('Normal',eta,sqrt(Omega),1);
Omega = Omega*eye(Nh);
Pl = Omega(ind);
dOmega = diag(Omega)';
I = eye(size(Y,1));

% initializing J with Loreta solution => J = arg min ||v-K*J||^2 + lambda^2*||L*J||^2

plotGCV = true;
nlambda = 500;
[~,lambda2] = inverseSolutionLoreta(Y,K,L,nlambda,plotGCV);

T = V*diag(s./(s2+lambda2))*U';
H = K*T;
% T = V*diag(s./(s2+alpha))*U';
% J = T*Y; % (K'*K+alpha2*L'*L)\K'*Y;
% J = J/(std(J)+eps);
% figure;patch(sourceSpace,'FaceVertexCdata',J,'linestyle','none','FaceColor','interp','FaceLighting','phong','LineStyle','none');camlight

alpha2 = lambda2;
alpha = alpha2*Nh;
MaxError = 1e-3;
err = inf;
% optimizing lambda
while err > MaxError
    
    % computing Sigma
    [C1,iC1,c1] = getC(alpha,C1,Q1,Nh);
    
    % computing hyperprior's covariance  components
    Pi = getPi(alpha,Pi,Q1,iC1,Nh);
    Pij = getPij(Pij,Pi,C1,Nij,indI,indJ);
    
    e = H*Y-Y;
    el = lambda1-eta;
    e_c1 = e.^2 - c1;
    E_C1 = diag(e_c1);
    
    % computing the inverse of Sl
    for it=1:Nij
        iSl(indI(it),indJ(it)) = 0.5*trace( Pij(:,:,it)*E_C1 + Pi(:,:,indI(it))*C1*Pi(:,:,indJ(it))*C1 ) + Pl(it);
        iSl(indJ(it),indI(it)) = iSl(indI(it),indJ(it));
    end
    for it=1:Nh
        I1(it) = -0.5*trace( Pi(:,:,it)*U*diag(e_c1+alpha*s.^2)*U' ) + dOmega*el;
    end
    
    % using the identity trace(A*B) = trace(B*A)
    for it=1:Nij
        I2(indI(it),indJ(it)) = -iSl(it) - 0.5*alpha*trace( US2Ut*Pij(:,:,it) );
        I2(indJ(it),indI(it)) = I2(indI(it),indJ(it));
    end
    dl = -I2\I1;
    alpha = alpha+dl;%log(I1-dl);
    % alpha = alpha/alpha2;
    err = sum(dl.^2);
    
    T = V*diag(s./(s2+alpha*c1))*U';
    H = K*T;
end



F = -trace(U*diag((s./(s+diag(C1))))*U*Sy) - log(det(C)) - (lambda1 - eta)'/Omega*(lambda1 - eta) + log(det(S/Omega));
C1 = diag(exp(lambda1))*Q1;
alpha2 = exp(lambda2);
T = L'*V*diag((alpha2*s./(alpha2^2*s2+diag(C1))))*U';
J = T*Y;



function [C,iC,c,ic] = getC(alpha,C,Q,Nh) %#ok
C = alpha(1)*Q(:,:,1);
for it=2:Nh,
    C = C + alpha(it)*Q(:,:,it);
end
c = diag(C);
th = (max(c)-min(c))/100;
indR = c < th;
c(indR) = th;
ic = 1./c;
ic(indR) = 0;
iC = diag(ic);
    

function Pi = getPi(alpha,Pi,Q,iC,Nh)
for it=1:Nh
    Pi(:,:,it) = -alpha(it)*iC*Q(:,:,it)*iC;
end

function Pij = getPij(Pij,Pi,C,Nij,indI,indJ)
ind = indI == indJ;
for it=1:Nij
    Pij(:,:,it) = 2*Pi(:,:,indI(it))*C*Pi(:,:,indJ(it));
end
for it=1:length(ind) 
    Pij(:,:,ind(it)) = Pij(:,:,ind(it)) + Pi(:,:,ind(it));
end


