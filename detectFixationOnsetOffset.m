function [onsets,offsets,fixations] = detectFixationOnsetOffset(hotspotdata,srate,minFixationTime,maxSpatialDeviationForFixation)

%hotspotdata = hotspotdata(1:end,:);

% correct the interpolation in hotspotdata (change it to linear
% interpolation, so downsample first and then linearly interpolate,
% currently assumes 16 times upsampling)

%
%hotspotdata = hotspotdata(1:16:end,:);
hotspotdata1 = hotspotdata(:,1);
hotspotdata2 = hotspotdata(:,2);
%hotspotdataint1 = interp1([1:size(hotspotdata1,1)]',hotspotdata1,[1:1/16:size(hotspotdata1,1)]');
%hotspotdataint2 = interp1([1:size(hotspotdata2,1)]',hotspotdata2,[1:1/16:size(hotspotdata2,1)]');

hotspotdata = [hotspotdata1 hotspotdata2];

%screenLookings = findWhichScreenIsLookedAt(hotspotdata);
%hotspotdata(isnan(screenLookings),:) = NaN;
startOfWindow = 1;
lengthOfWindow = ceil(minFixationTime*srate);
endOfWindow = startOfWindow + lengthOfWindow - 1;

fixations = zeros(size(hotspotdata,1),1);
i = 1;
previousFixation = 0;
fixationMarker = 1;
while endOfWindow <= size(hotspotdata,1)
    %if sum(sum(abs(diff(hotspotdata(startOfWindow:endOfWindow,:)))))/lengthOfWindow < maxSpatialDeviationForFixation
    stds = std(hotspotdata(startOfWindow:endOfWindow,:));
    %stdss(i,:) = stds;
    
    if  ~any(~(stds <  maxSpatialDeviationForFixation))
        if previousFixation 
            fixations(startOfWindow:endOfWindow) = fixationMarker;
        else
            fixationMarker = -fixationMarker;
            fixations(startOfWindow:endOfWindow) = fixationMarker; 
        end
        endOfWindow = endOfWindow + 1;
        previousFixation = 1;
    else
        if previousFixation
            startOfWindow = endOfWindow;
            
        else
            startOfWindow = startOfWindow + 1;
            
        end
        endOfWindow = startOfWindow + lengthOfWindow - 1;
        previousFixation = 0;
    end
    if startOfWindow >= 1470
        5
    end
    i = i + 1;
end


tmp = diff(fixations);

onsets = find(tmp == 1);
offsets = find(tmp == -1);

% if fixations(1) == 1
%     onsets = [1 onsets];
% end

