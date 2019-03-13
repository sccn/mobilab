function varargout = MultiStreamBrowser(varargin)
% MULTISTREAMBROWSER MATLAB code for MultiStreamBrowser.fig
%      MULTISTREAMBROWSER, by itself, creates a new MULTISTREAMBROWSER or raises the existing
%      singleton*.
%
%      H = MULTISTREAMBROWSER returns the handle to a new MULTISTREAMBROWSER or the handle to
%      the existing singleton*.
%
%      MULTISTREAMBROWSER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MULTISTREAMBROWSER.M with the given input arguments.
%
%      MULTISTREAMBROWSER('Property','Value',...) creates a new MULTISTREAMBROWSER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MultiStreamBrowser_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MultiStreamBrowser_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MultiStreamBrowser

% Last Modified by GUIDE v2.5 08-Jun-2012 21:02:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MultiStreamBrowser_OpeningFcn, ...
                   'gui_OutputFcn',  @MultiStreamBrowser_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MultiStreamBrowser is made visible.
function MultiStreamBrowser_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
if isempty(varargin),
    try
        varargin{1} = evalin('base','mobilab');
    catch %#ok
        error('MoBILAB:noRunning','You have to have MoBILAB running. Try ''runmobilab'' first.');
    end
end


try
    if isa(varargin{1},'mobilabApplication') 
        mobilab = varargin{1};
        handles.mobilab = mobilab;
        guidata(hObject, handles);

        browserListObj = browserHandleList(handles.figure1);
        set(handles.figure1,'userData',browserListObj,'Color',mobilab.preferences.gui.backgroundColor);
        set(handles.text21,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text22,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text23,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text27,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text28,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text6,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text7,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.text8,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
        set(handles.uipanel5,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.uipanel6,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.uipanel7,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.open,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.load,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.save,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.play_rev,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.play,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.play_fwd,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.previous,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.next,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.pushbutton34,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.pushbutton36,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.showStreamList,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.createEvent,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.settings,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
        set(handles.edit8,'Value',1,'String',' ');
        
        path = fullfile(mobilab.path,'skin');        
        CData = imread([path filesep '32px-Gnome-document-open.svg.png']);     set(handles.open,'CData',CData);
        CData = imread([path filesep '32px-Gnome-document-save.svg.png']);     set(handles.save,'CData',CData);
        CData = imread([path filesep '32px-Gnome-insert-object.svg.png']);     set(handles.load,'CData',CData);
        CData = imread([path filesep '32px-Gnome-media-seek-backward.svg.png']);set(handles.play_rev,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-media-seek-forward.svg.png']);set(handles.play_fwd,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-preferences-system.svg.png']); set(handles.settings,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-media-seek-backward.svg.png']);set(handles.previous,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-media-seek-forward.svg.png']);set(handles.next,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-media-playback-start.svg.png']);set(handles.play,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
        CData = imread([path filesep '32px-Gnome-preferences-other.svg.png']);set(handles.showStreamList,'CData',CData);
        CData = imread([path filesep '32px-Gnome-colors-alacarte.svg.png']);set(handles.createEvent,'CData',CData);
        set(handles.play,'userData',{imread([path filesep '32px-Gnome-media-playback-start.svg.png']) imread([path filesep '32px-Gnome-media-playback-pause.svg.png'])});
        
        set(handles.slider1,'Max',10000);
        set(handles.slider1,'Min',browserListObj.nowCursor);
        set(handles.slider1,'Value',browserListObj.nowCursor);
        set(handles.slider1,'SliderStep',0.00025*ones(1,2));
        set(handles.text6,'String',num2str(browserListObj.nowCursor));
        set(handles.text7,'String',num2str(browserListObj.endTime));
        set(handles.text8,'String',['Current latency = ' num2str(browserListObj.nowCursor) ' sec']);
        
        set([handles.popupmenu2 handles.edit8 handles.text27 handles.text28 handles.listbox1 handles.previous handles.next],'Visible','off');
        set([handles.popupmenu2 handles.edit8 handles.text27 handles.text28 handles.listbox1 handles.previous handles.next],'Enable','off');
        try
            hListener = handle.listener(handles.slider1,'ActionEvent',@slider1_Callback);
        catch
            hListener = addlistener(handles.slider1,'ContinuousValueChange',@slider1_Callback);
        end
        setappdata(handles.slider1,'sliderListeners',hListener);

        set(handles.load,'Visible','off');
        load_Callback(handles.load, [], handles);
    elseif isa(varargin{1},'coreStreamObject') 
        streamObj = varargin{1};
        browserListObj = get(handles.figure1,'userData');
        browserListObj.addHandle(streamObj);
    elseif isa(varargin{length(varargin)},'coreStreamObject') 
        streamObj = varargin{length(varargin)};
        browserListObj = get(handles.figure1,'userData');
        browserListObj.addHandle(streamObj);
    elseif varargin{1} == -1
        return;
    else
        error('MoBILAB:noRunning','You have to have MoBILAB running. Try ''runmobilab'' first.');
    end
catch ME
    if strcmp(ME.identifier,'MATLAB:nonStrucReference'),return;end
    if strcmp(ME.message,'There are no data to plot.')
        warndlg2(ME.message);
    else
        errordlg(ME.message);
    end
end



% --- Outputs from this function are returned to the command line.
function varargout = MultiStreamBrowser_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;



% --- Executes on button press in load.
function load_Callback(hObject, eventdata, handles)
mobilab = handles.mobilab;
try
    [isactive,hFigure] = mobilab.isGuiActive;
    if isactive
        position = get(hFigure,'Position');
        %close(mobilab.isGuiActive);
        treeHandle = mobilab.gui('add2Browser_Callback');
        %hFigure = mobilab.isGuiActive;
        set(hFigure,'Position',position);
    else
        treeHandle = mobilab.gui('add2Browser_Callback');
    end
    set(treeHandle,'Name','Right click on objects to add them to the Browser');
catch ME
    errordlg(ME.message);
end


% --- Executes on button press in play_rev.
function play_rev_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
    browserListObj.plotStep(-browserListObj.step);
catch ME
    errordlg(ME.message);
end


% --- Executes on button press in play.
function play_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
    if ~isempty(browserListObj.list)
        browserListObj.play;
        browserListObj.state;
        CData = get(handles.play,'UserData');
        if browserListObj.state
            set(handles.play,'CData',CData{2});
            start(browserListObj.timerObj);
        else
            set(handles.play,'CData',CData{1});
            stop(browserListObj.timerObj);  
        end
    end
catch ME
    errordlg(ME.message);
end


    
% --- Executes on button press in play_fwd.
function play_fwd_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
    browserListObj.plotStep(browserListObj.step);
catch ME
    errordlg(ME.message);
end


% --- Executes on button press in showStreamList.
function showStreamList_Callback(hObject, eventdata, handles)

browserListObj = get(handles.figure1,'userData');
if ishandle(browserListObj.slHandle), delete(browserListObj.slHandle);end
if ishandle(browserListObj.slHandle), delete(browserListObj.slHandle);end
browserListObj.slHandle = StreamsList(browserListObj);
set(handles.figure1,'userData',browserListObj);


% --- Executes on button press in settings.
function settings_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
    browserListObj.changeSettings;
catch ME
    errordlg(ME.message);
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
if isMultipleCall, return;  end
value = get(hObject,'Value');
try
    browserListObj = get(get(get(hObject,'parent'),'parent'),'userdata');
    if value + browserListObj.bound <= browserListObj.endTime &&...
            value - browserListObj.bound >= browserListObj.startTime
        newNowCursor = value;
    elseif value + browserListObj.bound > browserListObj.endTime
        newNowCursor = value - browserListObj.bound/2;
    else
        newNowCursor = value + browserListObj.bound/2;
    end
    browserListObj.plotThisTimeStamp(newNowCursor);
catch ME
    errordlg(ME.message);
end

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
browserListObj.delete;
treeHandle = get(handles.load,'userData');
try delete(treeHandle);end %#ok


% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
[FileName,PathName] = uiputfile2({'*.session','MS Bowser session'},'Select the session file');
if any([isnumeric(FileName) isnumeric(PathName)]), return;end
sessionFilename = [PathName FileName];
browserListObj.save(sessionFilename);




% --- Executes on button press in open.
function open_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
[FileName,PathName] = uigetfile2({'*.session','MS Bowser session'},'Select the session file');
if any([isnumeric(FileName) isnumeric(PathName)]), return;end
sessionFilename = [PathName FileName];
browserListObj.load(sessionFilename);


% --- Executes on button press in previous.
function previous_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
if isempty(browserListObj.list), return;end
objName = get(handles.edit8,'String');
if isempty(objName), return;end
itemIndex = get(handles.edit8,'Value');

ind = get(handles.listbox1,'ListboxTop');
if ~isempty(browserListObj.list{itemIndex}.streamHandle.event.label)
    [~,loc] = ismember( browserListObj.list{itemIndex}.streamHandle.event.label, browserListObj.list{itemIndex}.streamHandle.event.uniqueLabel{ind});
    latency = browserListObj.list{itemIndex}.streamHandle.event.latencyInFrame(logical(loc));
    I = browserListObj.nowCursor > browserListObj.list{itemIndex}.streamHandle.timeStamp(browserListObj.list{itemIndex}.streamHandle.event.latencyInFrame(logical(loc)));
    latency(~I) = [];
    if ~isempty(latency)
        latency = sort(latency);
        jumpLatency = latency(end);
        set(handles.slider1,'Value',browserListObj.list{itemIndex}.streamHandle.timeStamp(jumpLatency));
        slider1_Callback(handles.slider1,[],handles); 
    end
end



% --- Executes on button press in next.
function next_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
if isempty(browserListObj.list), return;end
objName = get(handles.edit8,'String');
if isempty(objName), return;end
itemIndex = get(handles.edit8,'Value');

ind = get(handles.listbox1,'ListboxTop');
if ~isempty(browserListObj.list{itemIndex}.streamHandle.event.label)
    [~,loc] = ismember( browserListObj.list{itemIndex}.streamHandle.event.label, browserListObj.list{itemIndex}.streamHandle.event.uniqueLabel{ind});
    latency = browserListObj.list{itemIndex}.streamHandle.event.latencyInFrame(logical(loc));
    I = browserListObj.nowCursor < browserListObj.list{itemIndex}.streamHandle.timeStamp(browserListObj.list{itemIndex}.streamHandle.event.latencyInFrame(logical(loc)));
    latency(~I) = [];
    if ~isempty(latency)
        latency = sort(latency);
        jumpLatency = latency(1);
        set(handles.slider1,'Value',browserListObj.list{itemIndex}.streamHandle.timeStamp(jumpLatency));
        slider1_Callback(handles.slider1,[],handles); 
    end
end


% --- Executes on button press in previous.
function pushbutton28_Callback(hObject, eventdata, handles)
% hObject    handle to previous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in next.
function pushbutton29_Callback(hObject, eventdata, handles)
% hObject    handle to next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
if isempty(browserListObj.list), return;end
index = get(handles.edit8,'Value');
set( handles.listbox1,'String',browserListObj.list{index}.streamHandle.event.uniqueLabel);


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in loadPlus.
function loadPlus_Callback(hObject, eventdata, handles)
% hObject    handle to loadPlus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton34.
function pushbutton34_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton36.
function pushbutton36_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
        
    latency = str2double(get(handles.edit10,'String'));
    if isempty(latency), error('Invalid latency');end
        
    if ~(latency >= 0 && latency <= browserListObj.list{1}.streamHandle.timeStamp(end)), error('This latency is out of range');end
    [~,latency] = min(abs(browserListObj.list{1}.streamHandle.timeStamp-latency));
        
    label = get(handles.edit11,'String');
    if isempty(label), error('You must enter a label (EEG event type).');end

    browserListObj.list{1}.streamHandle.event.addEvent(latency,label);  
    EEG = evalin('base','EEG');
    EEG = eeg_addnewevents(EEG, {latency}, {label});
    % EEG = userData.event.event2eeglab(EEG);
    assignin('base','EEG',EEG);
    evalin('base','eeglab(''redraw'')');
catch ME
    errordlg(ME.message);
end



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in createEvent.
function createEvent_Callback(hObject, eventdata, handles)
browserListObj = get(handles.figure1,'userData');
if isempty(browserListObj.list), errordlg('Load up some data into the browser first.');return;end
browserListObj.ceHandle = CreateEvent(browserListObj);


% --- Executes on button press in play_rev.
function pushbutton39_Callback(hObject, eventdata, handles)
% hObject    handle to play_rev (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in play.
function pushbutton40_Callback(hObject, eventdata, handles)
% hObject    handle to play (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in load.
function pushbutton41_Callback(hObject, eventdata, handles)
% hObject    handle to load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in play_fwd.
function pushbutton42_Callback(hObject, eventdata, handles)
% hObject    handle to play_fwd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in settings.
function pushbutton43_Callback(hObject, eventdata, handles)
% hObject    handle to settings (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in save.
function pushbutton44_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in open.
function pushbutton45_Callback(hObject, eventdata, handles)
% hObject    handle to open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in previous.
function pushbutton46_Callback(hObject, eventdata, handles)
% hObject    handle to previous (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in next.
function pushbutton47_Callback(hObject, eventdata, handles)
% hObject    handle to next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox1.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider5_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function flag = isMultipleCall
flag = false;
% Get the stack
s = dbstack;
if numel(s) <=2
    % Stack too short for a multiple call
    return
end

% How many calls to the calling function are in the stack?
names = {s(:).name};
TF = strcmp(s(2).name,names);
count = sum(TF);
if count>1
    % More than 1
    flag = true;
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
if isvalid(handles.mobilab) && handles.mobilab.isGuiActive 
    handles.mobilab.gui('dispNode_Callback');
end
delete(hObject);
