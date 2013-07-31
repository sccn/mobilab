function t = tStudent2Dmap(x)
n = size(x,1);
t = mean(x)./(std(x)+eps)/sqrt(n);
t = squeeze(t);