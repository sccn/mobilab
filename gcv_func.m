function lambda2Opt = gcv_func(UtY,s,p,nlambda,plotGCV)
s2 = s.^2;
n = numel(UtY);
if isa(s,'gpuArray')
    s = gather(s);
    gcv = gpuArray.zeros(nlambda,1);
else
    gcv = zeros(nlambda,1);
end
tol = max([n p])*eps(max(s));
lambda2 = logspace(log10(tol),log10(max(s.^2)),nlambda);
for it=1:nlambda
    d = lambda2(it)./(s2+lambda2(it));
    err = bsxfun(@times,d,UtY);  %   y-y_hat
    se  = dot(err,err,1);        % ||y-y_hat||^2
    mse = mean(se);              % sum ||y_k-y_hat_k||^2, k=1,..., number of time points
    gcv(it) = mse/sum(d)^2;
end
loc = getMinima(gcv);
lambda2Opt = lambda2(loc);

if plotGCV
    figure;
    semilogx(lambda2,gcv)
    xlabel('log-lambda');
    ylabel('GCV');
    hold on;
    plot(lambda2Opt,gcv(loc),'rx','linewidth',2)
    title(['\lambda^2=' num2str(lambda2Opt) '  loc=' num2str(loc(end))]);
    grid on;
end
end

%---
function indmin = getMinima(x)
fminor = diff(x)>=0;
fminor = ~fminor(1:end-1, :) & fminor(2:end, :);
fminor = [0; fminor; 0];
indmin = find(fminor);
if isempty(indmin), [~,indmin] = min(x);end
indmin = indmin(end);
end
