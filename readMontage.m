function [elec,labels,fiducials] = readMontage(file)
[eloc, labels] = readlocs(file);
elec = [cell2mat({eloc.X}'), cell2mat({eloc.Y}'), cell2mat({eloc.Z}')];
Nl = length(labels);
count = 1;
for it=1:Nl
    if ~isempty(strfind(labels{it},'fidnz')) || ~isempty(strfind(labels{it},'nasion'))
        fiducials.nasion = elec(it,:);
        count = count+1;
    elseif ~isempty(strfind(labels{it},'fidt9')) || ~isempty(strfind(labels{it},'lpa'))
        fiducials.lpa = elec(it,:);  
        count = count+1;
    elseif ~isempty(strfind(labels{it},'fidt10')) || ~isempty(strfind(labels{it},'rpa'))
        fiducials.rpa = elec(it,:);
        count = count+1;
    elseif ~isempty(strfind(labels{it},'fidt10')) || ~isempty(strfind(labels{it},'vertex'))
        fiducials.vertex = elec(it,:);
        count = count+1;
    end
    if count > 4, break;end
end