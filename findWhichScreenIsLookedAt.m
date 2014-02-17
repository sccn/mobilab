%channels are x,y coordinates on each screen.. If looking at screen, values
%should be between 0 and 1 for that screen.
% in case of multiple screen lookings closer screen is selected. NaN means person doesn't look at any screens. 
% size(data,2) = 2*numberOfScreens
function res = findWhichScreenIsLookedAt(data)
    res = zeros(size(data,1),1);
    screenLookings = zeros(size(data,1),size(data,2)/2);
for i = 1:size(data,2)/2
    screenLookings(:,i) = ((data(:,2*(i-1)+1)>=0)&(data(:,2*i)>=0)&(data(:,2*(i-1)+1)<=1)&(data(:,2*i)<=1)); %looking at screen i?
end


res(sum(screenLookings,2) == 0) = NaN;

for i = find(sum(screenLookings,2) > 1)'
    
    competingScreens = find(screenLookings(i,:) == 1);
    
        diff = data(i,vec([2*(competingScreens-1)+1 ;2*competingScreens])) - repmat([0.5 0.5],1,length(competingScreens));
        sums = diff.^2 + circshift(diff.^2,1);
        [~,winningInd] = min(sums(2:2:end));
        res(i) = competingScreens(winningInd);
    
end



onlyOneScreenLookings = find(sum(screenLookings,2) == 1);
res(onlyOneScreenLookings) = sum(screenLookings(onlyOneScreenLookings,:).*repmat(1:size(data,2)/2,length(onlyOneScreenLookings),1),2);

end