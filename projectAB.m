function ap = projectAB(a,b)
na = sqrt(sum(a.^2,2));
ua = bsxfun(@rdivide,a,eps+na);
ub = bsxfun(@rdivide,b,eps+sqrt(sum(b.^2,2)));
cosAB = dot(ua,ub,2);
ang = squeeze(real(acos(cosAB))*180/pi);
direction = sign(90-ang);

ap = bsxfun(@times,ub,cosAB.*na);
ap = squeeze(sqrt(sum(ap.^2,2)));
ap = ap.*direction;

% a = bsxfun(@rdivide,a,eps+sqrt(sum(a.^2,2)));
% cosAB = squeeze(dot(a,b,2));
% 
% a1 = a;
% a1(:,1) = squeeze(a(:,1,:)).*cosAB;
% a1(:,2) = squeeze(a(:,2,:)).*cosAB;
% ang = mean(real(acos(cosAB))*180/pi,2);
% direction = sign(90-ang);
% a = sqrt(sum(a1.^2,2));
% a = bsxfun(@times,a,direction);
