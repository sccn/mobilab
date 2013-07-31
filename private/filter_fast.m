function [X,Zf] = filter_fast(B,A,X,Zi,dim)
% [Y,Zf] = filter_fast(B,A,X,Zi,Dim)
% Equivalent to filter(), except for being faster when both the filter and the signal are long, by using FFT convolution (needs fftfilt).
% The function is faster than filter when approx. length(B)>256 and size(X,Dim)>1024, otherwise slower (due size-testing overhead).
% 
% See also:
%   filter()
%
%                           Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                           2010-07-09

persistent has_fftfilt;
if isempty(has_fftfilt)
    has_fftfilt = exist('fftfilt','file'); end
if ~exist('dim','var')
    dim = find(size(X)~=1,1); end
if ~exist('Zi','var')
    Zi = []; end

lenx = size(X,dim);
lenb = length(B);

if lenb < 256 || lenx<1024 || lenx <= lenb || lenx*lenb < 4000000 || ~isequal(A,1) || ~has_fftfilt
    % use the regular filter
    if nargout > 1
        [X,Zf] = filter(B,A,X,Zi,dim);
    else
        X = filter(B,A,X,Zi,dim);
    end
else
    was_single = strcmp(class(X),'single');
	% fftfilt can be used
    if isempty(Zi)
        % no initial conditions to take care of
        if nargout < 2
            % and no final ones
            X = unflip(fftfilt(B,flip(double(X),dim)),dim);
        else
            % final conditions needed
            X = flip(X,dim);
            [dummy,Zf] = filter(B,1,X(end-length(B)+1:end,:),Zi,1); %#ok<ASGLU>
            X = fftfilt(B,double(X));
            X = unflip(X,dim);
        end
    else
        % initial conditions available
        X = flip(X,dim);
        % get a Zi-informed piece
        tmp = filter(B,1,X(1:length(B),:),Zi,1);
        if nargout > 1
            % also need final conditions
            [dummy,Zf] = filter(B,1,X(end-length(B)+1:end,:),Zi,1); %#ok<ASGLU>
        end
        X = fftfilt(B,double(X));
        % incorporate the piece
        X(1:length(B),:) = tmp;
        X = unflip(X,dim);
    end
    if was_single
        X = single(X); end
end

function X = flip(X,dim)
if dim ~= 1
    order = 1:ndims(X);
    order = order([dim 1]);
    X = permute(X,order);
end

function X = unflip(X,dim)
if dim ~= 1
    order = 1:ndims(X);
    order = order([dim 1]);
    X = ipermute(X,order);
end
