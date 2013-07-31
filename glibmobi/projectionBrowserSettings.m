function varargout = projectionBrowserSettings(varargin)
% PROJECTIONBROWSERSETTINGS MATLAB code for projectionBrowserSettings.fig
%      PROJECTIONBROWSERSETTINGS, by itself, creates a new PROJECTIONBROWSERSETTINGS or raises the existing
%      singleton*.
%
%      H = PROJECTIONBROWSERSETTINGS returns the handle to a new PROJECTIONBROWSERSETTINGS or the handle to
%      the existing singleton*.
%
%      PROJECTIONBROWSERSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROJECTIONBROWSERSETTINGS.M with the given input arguments.
%
%      PROJECTIONBROWSERSETTINGS('Property','Value',...) creates a new PROJECTIONBROWSERSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before projectionBrowserSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to projectionBrowserSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help projectionBrowserSettings

% Last Modified by GUIDE v2.5 26-Mar-2012 15:32:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @projectionBrowserSettings_OpeningFcn, ...
    'gui_OutputFcn',  @projectionBrowserSettings_OutputFcn, ...
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


% --- Executes just before projectionBrowserSettings is made visible.
function projectionBrowserSettings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};

speedList = get(handles.popupmenu2,'String');
sList = zeros(length(speedList),1);
for it=1:length(speedList), A = speedList{it}; A(A == 'x') = [];sList(it) = str2double(A);end
if browserObj.speed < 1 
     speed = -round(1/browserObj.speed);
else
     speed = browserObj.speed;
end
set(handles.popupmenu2,'Value',find(sList == speed));
if isa(browserObj.master,'browserHandleList')
    set(handles.popupmenu2,'Enable','off');
end
[~,BGobj] = browserObj.streamHandle.container.viewLogicalStructure('',false);
descendants = BGobj.getDescendants(browserObj.streamHandle.container.findItem(browserObj.streamHandle.uuid)+1)-1;
if ~isempty(descendants)
    streamName = cell(length(descendants),1);
    ind = descendants;
    for it=1:length(descendants), streamName{it} =  BGobj.nodeIDs{descendants(it)+1};end
    if length(descendants) > 1
        set(handles.popupmenu3,'String',streamName);
    else
        set(handles.popupmenu3,'String',streamName{1});
    end
    if any(ind==browserObj.colorObjIndex)
        set(handles.popupmenu3,'Value',find(ind==browserObj.colorObjIndex));
    else
        set(handles.popupmenu3,'Value',1);
    end
    set([handles.checkbox6 handles.checkbox7 handles.popupmenu3],'enable','on');
else
    set(handles.popupmenu3,'String',' ');
    browserObj.colorCodeKinematicFlag = false;
    browserObj.showVectorsFlag = false;
    set([handles.checkbox6 handles.checkbox7 handles.popupmenu3],'enable','off');
end

set(handles.checkbox6,'Value',browserObj.colorCodeKinematicFlag);
set(handles.checkbox7,'Value',browserObj.showVectorsFlag);
set(handles.checkbox8,'Value',browserObj.magnitudeOrCurvature);
set(handles.edit4,'String',num2str(browserObj.windowWidth));
set(handles.edit12,'String',num2str(browserObj.channelIndex));
set(handles.checkbox4,'Value',browserObj.onscreenDisplay);
set(handles.figure1,'userData',browserObj);



% --- Outputs from this function are returned to the command line.
function varargout = projectionBrowserSettings_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



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


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1



function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popupmenu3 as text
%        str2double(get(hObject,'String')) returns contents of popupmenu3 as a double


% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popupmenu3 as text
%        str2double(get(hObject,'String')) returns contents of popupmenu3 as a double


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
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


% --- Executes on button press in done.
function done_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
try
    tmp = str2double(get(handles.edit4,'String'));
    if isempty(tmp), error('The width must be a number.');end
    if isnan(tmp), error('The width must be a number.');end
    if tmp > browserObj.streamHandle.originalStreamObj.timeStamp(browserObj.timeIndex(end))...
            - browserObj.streamHandle.originalStreamObj.timeStamp(browserObj.timeIndex(1))
        tmp = browserObj.streamHandle.originalStreamObj.timeStamp(browserObj.timeIndex(end))...
            - browserObj.streamHandle.originalStreamObj.timeStamp(browserObj.timeIndex(1));
    end
    userData.windowWidth = tmp;    
    
    tmp = str2double(get(handles.edit12,'String'));
    if isempty(tmp), error('The channels must be a numbers.');end
    if isnan(tmp), error('The channels must be a numbers.');end
    userData.channels = tmp;
   
    speed = get(handles.popupmenu2,'String');
    index = get(handles.popupmenu2,'Value');
    speed = speed{index};
    speed(speed=='x') = [];
    speed = str2double(speed);
    if speed < 1, speed = 1/abs(speed);end
    userData.speed = speed;
          
    userData.onscreenDisplay = get(handles.checkbox4,'Value');
    userData.colorCodeKinematicFlag = get(handles.checkbox6,'Value');
    userData.showVectorsFlag = get(handles.checkbox7,'Value');
    userData.magnitudeOrCurvature = get(handles.checkbox8,'Value');
        
    if userData.colorCodeKinematicFlag
        stringName = get(handles.popupmenu3,'String');
        if ~isempty(stringName)
            if iscellstr(stringName)
                name = stringName{get(handles.popupmenu3,'Value')};
            else
                name = stringName;
            end
            loc(1) = find(name == '(');
            loc(2) = find(name == ')');
            index = str2double(name(loc(1)+1:loc(2)-1));
            userData.colorObjIndex = index;
        end
    else
        userData.showVectorsFlag = false;
        userData.colorObjIndex = 0;
    end
    set(handles.figure1,'userData',userData);
    uiresume
catch ME
    errordlg2(ME.message);
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
close(handles.figure1);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
web http://sccn.ucsd.edu/wiki/Mobilab_software/MoBILAB_toolbox_tutorial



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


% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4



% --- Executes on button press in checkbox5.
function checkbox5_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox5



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


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox6.
function checkbox6_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox6


% --- Executes on button press in checkbox7.
function checkbox7_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox7


% --- Executes on button press in checkbox8.
function checkbox8_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox8


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
h = Events2display(browserObj);
uiwait(h);
if ~ishandle(h), return;end
labels = get(findobj(h,'tag','listbox2'),'String');
close(h);
if isempty(labels);
    browserObj.eventObj = event;
    return;
end
browserObj.eventObj = event;
for it=1:length(labels)
    latency = browserObj.streamHandle.event.getLatencyForEventLabel(labels{it});
    browserObj.eventObj = browserObj.eventObj.addEvent(latency,labels{it});
end
