function sdata = svdDenoising4ERP(data,n)
if nargin < 2, n = 10;end

N= size(data,2);
[U,S,V] = svds(data, n);
sdata = U * S * V';