function eegplotNG(EEG,componentsOrCChannels)
if nargin < 2, componentsOrCChannels = 1;end
if componentsOrCChannels && ~isempty(EEG.icaact)
    EEG.data = EEG.icaact;
end
eegplotObj = eegplotNGHandle(EEG);