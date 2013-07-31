function [hist12, hist1, hist2]=mutual_histogram_double(I1,I2,Imin,Imax,nbins)
% This function makes a 2D joint histogram of 1D,2D...ND images
% and also calculates the seperate histograms of both images.
%
% Note! : For a value of 4.3, the histogram function will put 0.7 in bin 4 
% and 0.3 in bin 5. This makes the histogram less discrete for the use by optimizers
%
% [hist12, hist1, hist2]=mutual_histogram_double(I1,I2,Imin,Imax,nbins)
%
% Function is written by D.Kroon University of Twente (August 2010)
%

% Number of bins must be integer
nbins=round(nbins);

% scaling value 
scav=nbins/(Imax-Imin);

% Indices (1D) of all pixels
index=1:numel(I1);

% Calculate histogram positions
xd=scav*(I1(index)-Imin); xd=xd(:); 
yd=scav*(I2(index)-Imin); yd=yd(:);

% Calculate both neighbors and interpolation percentages
xm=floor(xd);xp=xm+1;
xmd=xp-xd; xpd=xd-xm;
ym=floor(yd); yp=ym+1;
ymd=yp-yd; ypd=yd-ym;

% Fit to range ...
xm(xm<0)=0; xp(xp<0)=0;
xm(xm>(nbins-1))=(nbins-1); xp(xp>(nbins-1))=(nbins-1);
ym(ym<0)=0; yp(yp<0)=0;
ym(ym>(nbins-1))=(nbins-1); yp(yp>(nbins-1))=(nbins-1);

xm=xm+1; ym=ym+1; xp=xp+1; yp=yp+1;

% Make sum of all values in histogram 1 and histrogram 2 equal to 1
xmd=xmd./numel(I1); xpd=xpd./numel(I1); 
ymd=ymd./numel(I1); ypd=ypd./numel(I1); 
 
% Accumulate all histogram values
hist1 = accumarray([xm;xp], [xmd;xpd], [nbins 1]);
hist2 = accumarray([ym;yp], [ymd;ypd], [nbins 1]);
hist12 = accumarray([xm ym;xp ym;xm yp;xp yp],[xmd.*ymd;xpd.*ymd;xmd.*ypd;xpd.*ypd],[nbins nbins]);


  