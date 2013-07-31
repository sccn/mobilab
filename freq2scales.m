function scales = freq2scales(fmin, fmax, fnum, wname, delta)
if nargin < 5, error('Not enough input arguments.');end 
wpool = {'cmor1-1.5','cmor','haar','db','sym','coif','bior','rbio','meyr','dmey','gaus','mexh','morl','cgau','shan','fbsp'};
if ~any(strcmp(wpool,wname)), error('Unknown wavelet.');end

fc = centfrq(wname);
smax = fc/(fmin.*delta);
smin = fc/(fmax.*delta);
scales = logspace(log10(smin),log10(smax), fnum);
scal2frq(scales([1 end]),wname,delta);