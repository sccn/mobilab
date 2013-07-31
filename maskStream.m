function mObj = maskStream(obj,timeMask)
loc = ismember(obj.timeStamp,timeMask);
if ~any(loc)
    error('MoBILAB:maskStream','Time stamps don''t match  at all.');
end
I = diff(loc);
seMatrix = obj.timeStamp([find(I==1)'+1 find(I==-1)'+1]);

bsObj = basicSegment(seMatrix,'msk');
mObj = bsObj.apply(obj);