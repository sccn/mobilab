% Given a data with undesired zero values, this function fills those zero 
% values by using interpolation methods. If there are zero values in the
% beginning or end of the data, it can also extrapolate.
%
% - x is Nxp data, of p signals of length N. Data signals x(:,i) should
% have zero values at the same rows. Otherwise, call the function
% seperately for each signal.
%
% - method is interp1 methods. See interp1.m

function res = interpolateZeroValues(x,method)

% in the case of non-vector x, make sure that a row is completely missing
for i = size(x,2)
    outlierLocations = find(x(:,i) == 0);
    x(outlierLocations,:) = 0;
end
validLocations = setdiff(1:length(x),outlierLocations);

        
res = interp1(validLocations',x(validLocations,:),(1:length(x))',method,'extrap');

