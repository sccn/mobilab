function addBrowserList_Callback(obj)
if isMultipleCall, return;end
try
    MultiStreamBrowser(obj); 
catch ME
    errordlg(ME.message)
end
