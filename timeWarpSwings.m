function [Xtw,labels_tw,mXtw,labels,swingTimeMatrix,xyXtw] = timeWarpSwings(X,samplingRate,correctByLength)
%% time warping (if correctByLength==true it takes into account the length of each trial)

if nargin < 3, correctByLength = true;end

Ns = length(X);
maxLength = zeros(Ns,1);
for jt=1:Ns, maxLength(jt) = X{jt}.maxLength;end
maxLength = max(maxLength);
Xtw = [];
xyXtw = [];
labels_tw = {};
labels = cell(Ns,1);
mXtw = zeros(2*maxLength,5,Ns);
swingTimeMatrix = cell(Ns,1);
for jt=1:Ns
    
    
    nt1 = length(X{jt}.rl.data);
    for k=1:nt1 v1(k,:) = X{jt}.rl.data{k}(end,1:2);end
    
    nt2 = length(X{jt}.lr.data);
    for k=1:nt2 v2(k,:) = X{jt}.lr.data{k}(1,1:2);end
    
    loc2 = zeros(nt1,1);
    for k=1:nt1
        ind = find(all(ismember(v2,v1(k,:)),2));
        if ~isempty(ind), loc2(k) = ind;end
    end
    loc1 = (1:nt1)';
    loc1(loc2==0) = [];
    loc2(loc2==0) = [];
         
    dl1 = diff(X{jt}.rl.latency,[],2);
    rm1 = dl1(loc1) >= 2*median(dl1(loc1));
    
    dl2 = diff(X{jt}.lr.latency,[],2);
    rm2 = dl2(loc2) >= 2*median(dl2(loc2));
    
    rm = rm1 | rm2;
    loc1(rm) = [];
    loc2(rm) = [];
    
    swingTimeMatrix{jt} = [X{jt}.rl.latency(loc1,1) X{jt}.lr.latency(loc2,2)];
      
    ntr = length(X{jt}.rl.data);
    x = zeros(maxLength,2,4,ntr);
    for h=1:ntr
        n1 = size(X{jt}.rl.data{h},1);
        t1 = linspace(1,maxLength,n1)';
        tmp = interp1(t1,X{jt}.rl.data{h},(1:maxLength)');
        x(:,:,:,h) = reshape(tmp,[maxLength 2 4]);
    end
    xrl = reshape(x,[maxLength 8 ntr]);
    kk = squeeze(sqrt(sum(x.^2,2)));
    tmp2.rl = [];
    tmp2.rl(:,1,:) = squeeze(x(:,1,1,:));
    tmp2.rl(:,2,:) = squeeze(x(:,2,1,:));
    tmp2.rl(:,3,:) = kk(:,2,:);
    tmp2.rl(:,4,:) = kk(:,3,:);
    tmp2.rl(:,5,:) = kk(:,4,:);
    
    ntl = length(X{jt}.lr.data);
    x = zeros(maxLength,2,4,ntl);
    for h=1:ntl
        n1 = size(X{jt}.lr.data{h},1);
        t1 = linspace(1,maxLength,n1)';
        tmp = interp1(t1,X{jt}.lr.data{h},(1:maxLength)');
        x(:,:,:,h) = reshape(tmp,[maxLength 2 4]);    
    end
    xlr = reshape(x,[maxLength 8 ntl]);
    kk = squeeze(sqrt(sum(x.^2,2)));
    tmp2.lr = [];
    tmp2.lr(:,1,:) = squeeze(x(:,1,1,:));
    tmp2.lr(:,2,:) = squeeze(x(:,2,1,:));
    tmp2.lr(:,3,:) = kk(:,2,:);
    tmp2.lr(:,4,:) = kk(:,3,:);
    tmp2.lr(:,5,:) = kk(:,4,:);
    
    xax_rl = xrl(:,1,loc1)-min(xrl(:,1,loc1));
    xax_lr = -(xlr(:,1,loc2)-min(xlr(:,1,loc2)));
    xrl(:,1,loc1) = xax_rl;
    xlr(:,1,loc2) = xax_lr;
    kkrllr = cat(1,xrl(:,:,loc1),xlr(:,:,loc2));
    k = (X{jt}.rl.swingLength(loc1) + X{jt}.lr.swingLength(loc2))/(maxLength/samplingRate);
    kk = cat(1,tmp2.rl(:,:,loc1),tmp2.lr(:,:,loc2));
    
    if correctByLength
        kk = permute(kk,[1 3 2]);
        kk(:,:,3) = bsxfun(@times,kk(:,:,3),k');
        kk(:,:,4) = bsxfun(@times,kk(:,:,4),k.^2');
        kk(:,:,5) = bsxfun(@times,kk(:,:,5),k'.^3);
        kk = permute(kk,[1 3 2]);
        
        kkrllr(:,3,:) = bsxfun(@times,squeeze(kkrllr(:,3,:)),k');
        kkrllr(:,4,:) = bsxfun(@times,squeeze(kkrllr(:,4,:)),k');
        kkrllr(:,5,:) = bsxfun(@times,squeeze(kkrllr(:,5,:)),k.^2');
        kkrllr(:,6,:) = bsxfun(@times,squeeze(kkrllr(:,6,:)),k.^2');
        kkrllr(:,7,:) = bsxfun(@times,squeeze(kkrllr(:,7,:)),k'.^3);
        kkrllr(:,8,:) = bsxfun(@times,squeeze(kkrllr(:,8,:)),k'.^3);
    end
    
    %kk(:,1,:) = [];
    ntrKK = size(kk,3);
    
    Xtw(:,:,end+1:end+1+ntrKK-1) = kk;
    labels_tw = cat(1,labels_tw,repmat({X{jt}.label},ntrKK,1));
    
    mXtw(:,:,jt) = mean(kk,3);
    labels{jt} = X{jt}.label;
    
    xyXtw(:,:,end+1:end+1+ntrKK-1) = kkrllr;
end
Xtw(:,:,1) = [];
xyXtw(:,:,1) = [];
end