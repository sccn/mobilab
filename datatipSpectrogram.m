function output_txt = datatipSpectrogram(~,event_obj)
% Display the position of the data cursor
% obj          Currently not used (empty)
% event_obj    Handle to event object
% output_txt   Data cursor text string (string or cell array of strings).

handler = get(get(event_obj.Target,'parent'),'parent');
browserObj = get(handler,'userData');
if strcmp(browserObj.dcmHandle.Enable,'off'),return;end

persistent updateFlag
if isempty(updateFlag)
    
    updateFlag = 1;
    pos = get(event_obj,'Position');
    output_txt = {['latency: ',num2str(pos(1),4) ' sec'],...
        ['frequency: ',num2str(pos(2),4) ' Hz']};
       
    updateCursor(handler,pos);
    
    tobj = timer('TimerFcn',' ','StartDelay',0.2);
    start(tobj);
    wait(tobj);
    stop(tobj);
    
    updateFlag = [];
end