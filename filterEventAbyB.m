function [bf,af] = filterEventAbyB(a,b,win)
a = a(:);
b = b(:);

mask = false(size(b));
bf = mask;
af = mask;
a = find(a);
b = find(b);
n = length(b);
for it=1:n
    mask(b(it)-win:b(it)+win) = true;
    [~,~,loc2] = intersect(b(it)-win:b(it)+win,a);
    
    %- before
    loc = a(loc2) <= b(it);
    if any(loc)
        loc = find(loc);
        [~,locBf] = max(a(loc2(loc)));
        bf(a(loc2(loc(locBf)))) = true;
    end
    
    %- after
    loc = a(loc2) >= b(it);
    if any(loc)
        loc = find(loc);
        [~,locAf] = max(a(loc2(loc)));
        af(a(loc2(loc(locAf)))) = true;
    end
end