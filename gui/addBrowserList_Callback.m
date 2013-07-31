function addBrowserList_Callback(obj)
if isMultipleCall, return;end
try
    MultiStreamBrowser(obj); 
catch ME
    errordlg2(ME.message)
end
