function checkNumberInputArguments(low,high)
narg = evalin('caller','nargin');
if narg < low,  error('Too many input arguments..');end
if narg > high, error('Not enough input arguments.');end