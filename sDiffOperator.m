function D = sDiffOperator(dLength, dt, winLength)
% order: differentiation order
% winLength: Window length
if nargin < 2, dt = 1;end
if nargin < 3, winLength = 33;end

order = 4;
[~,g] = sgolay(order,winLength); 
D = sparse(dLength,dLength);
Ones = ones(1,dLength);
D(1:(winLength-1)/2+1,:) = g((winLength-1)/2+1:winLength,2)*Ones;
D(dLength-(winLength-1)/2+1:end,:) = g(1:(winLength-1)/2,2)*Ones;
for it=1:dLength, D(it,:) = circshift(D(it,:),it);end
D = D.'/dt;
