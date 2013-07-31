function [geometricMedian convergenceHistory]= geometric_median(x, varargin)
% geometricMedian = geometric_median(x, {key, value pairs})
% Input
%
%   x   is an N x M matrix, representing N observations of a M-dimensional matrix.
%
% Key, value pairs
%
%   initialGuess        is optional an 1 x M matrix, representing the initial guess for the gemetrix median
%
%   tolerance           an scalar value. It is the maximum relative change in geometricMedian vector (size of the change in
%                       the last iteration divided by the size of the geometricMedian vector) that makes the
%                       algorithm to continue to the next iteration. If relative change is less than tolerance, it is assumed
%                       that convergence is achieved.
%                       had a relative change more than tolerance then more iterations are performed.
%                       default = 1e-4.
%
% Output
%   geometricMedian     is an 1 x m matrix.
%   convergenceHistory  shows the value of maximum relative chage, which is compared to tolerance in
%                       each iteration.

% use mean as the median as an initial guess if none is provided.

inputOptions = finputcheck(varargin, ...
    {'initialGuess'         'real'  [] mean(x);...
    'tolerance'             'real' [0 1] 1e-6;...
    'maxNumberOfIterations' 'integer' [1 Inf]  1000;...
    });

geometricMedian = inputOptions.initialGuess;



for i=1:inputOptions.maxNumberOfIterations
    lastGeometricMedian = geometricMedian;
    differenceToEstimatedMedian = bsxfun(@minus, x, geometricMedian);
    sizeOfDifferenceToEstimatedMedian = (sum(differenceToEstimatedMedian .^2, 2) .^ 0.5);
    oneOverSizeOfDifferenceToEstimatedMedian = 1 ./ sizeOfDifferenceToEstimatedMedian;
    
    % to prevent nans
    oneOverSizeOfDifferenceToEstimatedMedian(isinf(oneOverSizeOfDifferenceToEstimatedMedian)) = 1e20;
    
    geometricMedian = sum(bsxfun(@times, x , oneOverSizeOfDifferenceToEstimatedMedian)) / sum(oneOverSizeOfDifferenceToEstimatedMedian);
    %maxRelativeChange = max(max(abs(lastGeometricMedian - geometricMedian)) ./ abs(geometricMedian));
    maxRelativeChange = (sum((lastGeometricMedian - geometricMedian).^2) / sum(geometricMedian.^2)) .^ 0.5;
    
    if nargout > 1
        convergenceHistory(i) = maxRelativeChange;
    end;
    
    if (maxRelativeChange < inputOptions.tolerance || isnan(maxRelativeChange))
        break;
    end;
end;