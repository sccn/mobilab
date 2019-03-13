function runmobilab(showGuiFlag)
if nargin < 1
    showGuiFlag = false;
end
try
    mobilab = evalin('base','mobilab');
    if ~isvalid(mobilab), error('MoBILAB:unexpectedError','Unexpected error.\nRestarting MoBILAB...');end
catch
    mobilab = mobilabApplication;
    assignin('base','mobilab',mobilab)
end
if showGuiFlag
    mobilab.gui;
end
