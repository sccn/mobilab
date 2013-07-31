function runmobilab
try
    mobilab = evalin('base','mobilab');
    if ~isvalid(mobilab), error('MoBILAB:unexpectedError','Unexpected error.\nRestarting MoBILAB...');end
catch %#ok
    mobilab = mobilabApplication;
    assignin('base','mobilab',mobilab)
end
mobilab.gui;
