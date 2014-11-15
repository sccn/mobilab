function add2EventsEditor2_Callback(obj)
if isMultipleCall, return;end
try
    EventsEditor2(obj); 
catch ME
    errordlg(ME.message)
end