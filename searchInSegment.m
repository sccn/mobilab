function [I,J,onset,offset] = searchInSegment(data,fun,h,movementThreshold, movementOnsetThresholdFine, minDuration, alpha)
if nargin < 2, fun = 'maxima';end
if nargin < 3, h = 300;end
if nargin < 4, movementThreshold = 1.2;end
if nargin < 5, movementOnsetThresholdFine = 0.05; end
if nargin < 6, minDuration = 0;end
if nargin < 7, alpha = 0.05;end
h = round(h);

[I,J] = deal(0);

% if strcmp(fun,'zero crossing')
%     x_mx = diff(x);
%     x_mx(end+1) = x_mx(end);
%     x_mx = smooth(x_mx,1024);
%     I1 = searchInSegment(x_mx,'maxima',h);
%     I2 = searchInSegment(-x_mx,'maxima',h);
%     I = [I1 I2];
%     return
% end

if strcmp(fun,'zero crossing')
    
    signsOfSignal = sign(data);
    
    % diff of the signs returns 2 or -2 at positions just before zero is
    % being crossed, find finds those positions in the vector and then 1 is
    % added to have the final positions which are just after zero has been
    % crossed
    I = find(diff(signsOfSignal))+1;
    
    return
end

if strcmp(fun,'movements')
    
    numberOnsets = 0;
    numberOffsets = 0;
    
%     pd = fitdist(data,'tlocationscale');
%     pi = paramci(pd,'alpha',movementThreshold);

%     thresholdData = movementThreshold*pd.sigma; %      % of values are below threshold -> 1.2 seems good (80% of data below)
    
%     thresholdData = movementThreshold * max(abs(data));
    
    sortedData = sort(abs(data));
    thresholdData = sortedData(round(length(sortedData)/100*movementThreshold));
    thresholdData / 0.07

    movement = false;
    positive = false;
    
    timePoint = h+1;
    lastOffsetTimePoint = 0;
    
    while timePoint <= length(data)-h
        step = 1;
        if ~movement
            if abs(data(timePoint)) > thresholdData
%             if data(timePoint) > pi(2,1) ||  data(timePoint) < pi(1,1)
                
                fineAccThreshold = max(abs(data(max(lastOffsetTimePoint+1,timePoint-h):timePoint+h)))*movementOnsetThresholdFine;

                fineTimePoint = timePoint;
                
                while abs(data(fineTimePoint)) > fineAccThreshold && fineTimePoint > lastOffsetTimePoint + 1
                    fineTimePoint = fineTimePoint - 1;
                end
                
                numberOnsets = numberOnsets + 1;
                I(numberOnsets) = fineTimePoint;
                movement = true;
                
                if sign(data(timePoint)) == 1
                    positive = true;
                else
                    positive = false;
                end

                step = fineTimePoint-timePoint+1;

                
            end
        else
            if data(timePoint) < fineAccThreshold && positive || data(timePoint) > -1*fineAccThreshold && ~positive
                numberOffsets = numberOffsets + 1;
                J(numberOffsets) = timePoint-1;
                movement = false;
                lastOffsetTimePoint = timePoint;
                
                if J(numberOffsets) - I(numberOnsets) <= minDuration
                    I(numberOnsets) = [];
                    J(numberOffsets) = [];
                    numberOnsets = numberOnsets - 1;
                    numberOffsets = numberOffsets -1;
                end
            end
        end    
            
        timePoint = timePoint + step;
    end
    
    return
    
end

if strcmp(fun,'sliding window deviation')
    %     SD = sqrt(var(x));
    
    numberEvents = 0;
    clear I
    
    D1 = diff(data);
    D1(end+1) = D1(end);
    
    D2 = diff(D1);
    D2(end+1) = D2(end);
    
    D3 =diff(D2);
    D3(end+1) = D3(end);
    
    D4 = diff(D3);
    D4(end+1) = D4(end);
    
    
    for timePoint = h:length(data)-h
        
        windowBefore = data(timePoint-h+1:timePoint);
        windowAfter = data(timePoint+1:timePoint+h);
        
        rightCurve(timePoint) = windowAfter(end)<windowBefore(end);
        
        %         windowBefore = y(timePoint-h+1:timePoint);
        %         windowAfter = y(timePoint+1:timePoint+h);
        
        varBefore(timePoint)  = var(windowBefore);
        SDbefore(timePoint)  = sqrt(varBefore(timePoint));
        meanBefore(timePoint)  = mean(windowBefore);
        rangeBefore(timePoint) = range(windowBefore);
        
        varAfter(timePoint)  = var(windowAfter);
        SDafter(timePoint)  = sqrt(varAfter(timePoint));
        meanAfter(timePoint)  = mean(windowAfter);
        rangeAfter(timePoint) = range(windowAfter);
        
        rangeDiffs(timePoint) = abs(rangeAfter(timePoint) - rangeBefore(timePoint));
        
        
        differenceStartEndWindow(timePoint)  = abs(data(timePoint-h+1)-data(timePoint));
    end
    
    
    timePoint = h;
    while timePoint < length(data)-h
%         disp(timePoint)
        
        if rangeAfter(timePoint) > rangeBefore(timePoint) * h/10 %(abs(meanAfter)  > abs(meanBefore) *10) % && (abs(meanAfter) > abs(meanBefore))
            
            
            
            %I(numberEvents+1) = timePoint + find(abs(jerk(timePoint+1:timePoint+h))==max(abs(jerk(timePoint+1:timePoint+h))));
            
            
            %                if rightCurve
            %
            %                    I(numberEvents+1) = timePoint + find(jerk(timePoint+1:timePoint+h)==min(jerk(timePoint+1:timePoint+h)));
            %
            %                else
            %
            %                    I(numberEvents+1) = timePoint + find(jerk(timePoint+1:timePoint+h)==max(jerk(timePoint+1:timePoint+h)));
            %
            %                end
            %
            
            %I(numberEvents+1) = timePoint + find(diff(sign(jerk(timePoint:timePoint+h))))+1;
            
%             I(numberEvents+1) = timePoint + find(rangeDiffs(timePoint+1:timePoint+h)==max(rangeDiffs(timePoint+1:timePoint+h)),1);

            windowSize = round(h/3); %round(h/2);

            for fineTimePoint = 1:h
                
%                 timePoint
%                 fineTimePoint
%                 windowSize
                dataToRegress = data(timePoint + fineTimePoint - windowSize+1 : timePoint + fineTimePoint + 2*windowSize-1);
                ratio(fineTimePoint) = ratioOutliersOfRegression(dataToRegress);


%                 dataToTimepoint = data(timePoint+fineTimePoint-windowSize+1:timePoint+fineTimePoint);
%                 xToTimePoint = 1:windowSize;
%                 xToTimePoint = xToTimePoint';
%                 X = [ones(size(xToTimePoint)) xToTimePoint];
%                 [B,BINT,R,RINT,stats] = regress(dataToTimepoint,X,0.001);
%                 timePoint
%                 RINT

%                 if rightCurve(timePoint) && RINT(windowSize,2) < 0
%                     
%                     ~(RINT(windowSize,1) < 0 && RINT(windowSize,2) > 0)
% %                     figure; plot(xToTimePoint,dataToTimepoint,'*');lsline
% %                     timePoint
% 
%                     numberEvents = numberEvents + 1;
%                     
%                     
%                     I(numberEvents) = timePoint + fineTimePoint;
%                     break
%                     
%                 else  
%                 end
            end

%             for fineTimePoint = 5:h
% 
%                 dataToTimepoint = data(timePoint+1:timePoint+fineTimePoint);
%                 xToTimePoint = 1:fineTimePoint;
%                 xToTimePoint = xToTimePoint';
%                 X = [ones(size(xToTimePoint)) xToTimePoint];
%                 [B,BINT,R,RINT] = regress(dataToTimepoint,X,0.001);
% 
%                 if ~(RINT(fineTimePoint,1) < 0 && RINT(fineTimePoint,2) > 0)
%                     
%                     numberEvents = numberEvents + 1;
%                     
%                     
%                     I(numberEvents) = timePoint + fineTimePoint;
%                     break
%                 end
%             end

            
            numberEvents = numberEvents + 1;
            I(numberEvents) = timePoint + find(ratio==max(ratio));
            
            clear ratio
            timePoint = timePoint + h;
        else
            timePoint = timePoint + 1;
            
        end
    end
%     I
    numberEvents
    
    return
end

if strcmp(fun,'minima'), data = -data;end
data = data - min(data);
t0 = 1;
t1 = t0 + h-1;
I = [];
n = length(data);
s = [0 0];
h = round(h/2)*2;
%figure;plot(x);
%hold on;
%hwait = waitbar(0,'Searching...','Color',[0.66 0.76 1]);
while t1 <= n
    ind = t0:t1;
    [~,loc] = max(data(t0:t1));
    
    try
        s(1) = sign(data(ind(loc))-data(ind(loc)-h/2));
    catch %#ok
        s(1) = sign(data(ind(loc))-data(1));
    end
    try
        s(2) = sign(data(ind(loc)+h/2)-data(ind(loc)));
    catch %#ok
        s(2) = sign(data(end)-data(ind(loc)));
    end
    %plot(t0:t1,x(t0:t1),'r');
    %plot([t0;t1],x([t0 t1]),'rx');
    
    if all(s == [1 -1])
        I(end+1) = ind(loc); %#ok
        try %#ok
            if abs(I(end)-I(end-1)) < 2*h
                tmp = I([end-1 end]);
                [~,mx] = max(data(tmp));
                I(end-1) = tmp(mx);%#ok
                I(end) = [];
            end
        end
        %plot(I(end),x(I(end)),'kx','linewidth',2);
    end
    t0 = t1;
    t1 = t1+h-1;
    %waitbar(t1/n,hwait);
end
xp = alpha*data(I);
onset = zeros(size(I));
offset = zeros(size(I));

for it=1:length(I)
    try
        [~,loc] = min(abs(xp(it)-data(I(it)-h:I(it))));
    catch %#ok
        [~,loc] = min(abs(xp(it)-data(1:I(it))));
    end
    onset(it) = I(it)-h+loc-1;
    try
        [~,loc] = min(abs(xp(it)-data(I(it):I(it)+h)));
    catch %#ok
        [~,loc] = min(abs(xp(it)-data(I(it):end)));
    end
    offset(it) = I(it)+loc-1;
end
onset(onset==0) = [];
offset(offset==0) = [];
end