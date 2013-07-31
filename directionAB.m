function direction = directionAB(a,b)
b = bsxfun(@rdivide,b,eps+sqrt(sum(b.^2,2)));
a = bsxfun(@rdivide,a,eps+sqrt(sum(a.^2,2)));
ang = real(acos(squeeze(dot(a,b,2))))*180/pi;
direction = sign(90-ang);
    