function [queryStrings, spaceLocations] = extractQueryStringsFromTextFile(PathName,FileName)

textFile = fullfile(PathName,FileName);
fid = fopen(textFile);
i = 1;
spaceLocations = [];
while 1
    tline = fgetl(fid);
    if strcmpi(tline,'%'),  break; end;
    if isempty(tline) || strcmpi(tline(1),' '),
        spaceLocations = [spaceLocations i-1];
    else
    
        queryStrings{i} = parseStringWithCharacter(tline,';');
        i = i + 1;
    end
    
end
end


function strings = parseStringWithCharacter(line,s)
    locs = strfind(line,s);
    strings = cell(length(locs)+1,1);
    locs = [0 locs length(line)+1];
    for i = 1:length(locs)-1
        strings{i} = line(locs(i)+1:locs(i+1)-1);
    end
end