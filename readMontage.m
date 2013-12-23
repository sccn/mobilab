function [elec,labels,fiducials] = readMontage(EEG)
try
    file = EEG.chaninfo.filename;
    [eloc, labels] = readlocs(file);
    elec = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
catch
    try
        for it=1:length(EEG.chanlocs)
            if isempty(EEG.chanlocs(it).X)
                EEG.chanlocs(it).X = nan;
                EEG.chanlocs(it).Y = nan;
                EEG.chanlocs(it).Z = nan;
            end
        end
        elec = [cell2mat({EEG.chanlocs.X}'), cell2mat({EEG.chanlocs.Y}'), cell2mat({EEG.chanlocs.Z}')];
        labels = {EEG.chanlocs.labels};
    catch
        disp('No channel information available.')
    end
end
lowerLabels = lower(labels);
Nl = length(labels);
count = 1;
rmThis = false(Nl,1);
for it=1:Nl
    if ~isempty(strfind(lowerLabels{it},'fidnz')) || ~isempty(strfind(lowerLabels{it},'nasion')) || ~isempty(strfind(lowerLabels{it},'Nz'))
        fiducials.nasion = elec(it,:);
        rmThis(it) = true;
        count = count+1;
    elseif ~isempty(strfind(lowerLabels{it},'fidt9')) || ~isempty(strfind(lowerLabels{it},'lpa')) || ~isempty(strfind(lowerLabels{it},'LPA'))
        fiducials.lpa = elec(it,:);  
        rmThis(it) = true;
        count = count+1;
    elseif ~isempty(strfind(lowerLabels{it},'fidt10')) || ~isempty(strfind(lowerLabels{it},'rpa')) || ~isempty(strfind(lowerLabels{it},'RPA'))
        fiducials.rpa = elec(it,:);
        rmThis(it) = true;
        count = count+1;
    elseif ~isempty(strfind(lowerLabels{it},'fidt10')) || ~isempty(strfind(lowerLabels{it},'vertex'))
        fiducials.vertex = elec(it,:);
        rmThis(it) = true;
        count = count+1;
    end
    if count > 4, break;end
end
elec(rmThis,:) = [];
labels(rmThis) = [];
end