function [I,onset,offset] = searchInSegment(x,fun,h,alpha)
if nargin < 2, fun = 'maxima';end
if nargin < 3, h = 300;end
if nargin < 4, alpha = 0.05;end
h = round(h);

if strcmp(fun,'zero crossing')
    x_mx = diff(x);
    x_mx(end+1) = x_mx(end);
    x_mx = smooth(x_mx,1024);
    I1 = searchInSegment(x_mx,'maxima',h);
    I2 = searchInSegment(-x_mx,'maxima',h);
    I = [I1 I2];
    return
end
if strcmp(fun,'minima'), x = -x;end
x = x - min(x);
t0 = 1;
t1 = t0 + h-1;
I = [];
n = length(x);
s = [0 0];
h = round(h/2)*2;
%figure;plot(x);
%hold on;
%hwait = waitbar(0,'Searching...','Color',[0.66 0.76 1]);
while t1 <= n
    ind = t0:t1;
    [~,loc] = max(x(t0:t1));
    
    try
        s(1) = sign(x(ind(loc))-x(ind(loc)-h/2));
    catch %#ok
        s(1) = sign(x(ind(loc))-x(1));
    end
    try
        s(2) = sign(x(ind(loc)+h/2)-x(ind(loc)));
    catch %#ok
        s(2) = sign(x(end)-x(ind(loc)));
    end
    %plot(t0:t1,x(t0:t1),'r');
    %plot([t0;t1],x([t0 t1]),'rx');
    
    if all(s == [1 -1])
        I(end+1) = ind(loc); %#ok
        try %#ok
            if abs(I(end)-I(end-1)) < 2*h
                tmp = I([end-1 end]);
                [~,mx] = max(x(tmp));
                I(end-1) = tmp(mx);%#ok
                I(end) = [];
            end
        end
        %plot(I(end),x(I(end)),'kx','linewidth',2);
    end
    t0 = t1;
    t1 = t1+h-1;
    %waitbar(t1/n,hwait);
end
xp = alpha*x(I);
onset = zeros(size(I));
offset = zeros(size(I));

for it=1:length(I)
    try
        [~,loc] = min(abs(xp(it)-x(I(it)-h:I(it))));
    catch %#ok
        [~,loc] = min(abs(xp(it)-x(1:I(it))));
    end
    onset(it) = I(it)-h+loc-1;
    try
        [~,loc] = min(abs(xp(it)-x(I(it):I(it)+h)));
    catch %#ok
        [~,loc] = min(abs(xp(it)-x(I(it):end)));
    end
    offset(it) = I(it)+loc-1;
end
onset(onset==0) = [];
offset(offset==0) = [];
end