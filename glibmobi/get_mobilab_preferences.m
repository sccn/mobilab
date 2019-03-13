function preferences = get_mobilab_preferences
try 
    mobilab = evalin('base','mobilab');
    preferences = mobilab.preferences;
catch
    try
        load(fullfile(getHomeDir,'.mobilab.mat'));
        preferences = configuration;
    catch
        preferences.gui.backgroundColor = [0.93 0.96 1];
        preferences.gui.buttonColor = [1 1 1];
        preferences.gui.fontColor = [0 0 0.4];
        preferences.mocap.interpolation = 'pchip';
        preferences.mocap.smoothing = 'sgolay';
        preferences.mocap.lowpassCutoff = 6;
        preferences.mocap.derivationOrder = 3;
        preferences.mocap.stickFigure = '';
        preferences.mocap.bodyParts = '';
        preferences.eeg.resampleMethod = 'linear';
        preferences.eeg.filterType = 'bandpass';
        preferences.eeg.cutoff = [1 200];
        preferences.tmpDirectory = tempdir;
    end
end