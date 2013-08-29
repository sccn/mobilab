function varargout = mocapBrowserSettings(varargin)
% MOCAPBROWSERSETTINGS MATLAB code for mocapBrowserSettings.fig
%      MOCAPBROWSERSETTINGS, by itself, creates a new MOCAPBROWSERSETTINGS or raises the existing
%      singleton*.
%
%      H = MOCAPBROWSERSETTINGS returns the handle to a new MOCAPBROWSERSETTINGS or the handle to
%      the existing singleton*.
%
%      MOCAPBROWSERSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOCAPBROWSERSETTINGS.M with the given input arguments.
%
%      MOCAPBROWSERSETTINGS('Property','Value',...) creates a new MOCAPBROWSERSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before mocapBrowserSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to mocapBrowserSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help mocapBrowserSettings

% Last Modified by GUIDE v2.5 19-Sep-2011 18:33:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @mocapBrowserSettings_OpeningFcn, ...
                   'gui_OutputFcn',  @mocapBrowserSettings_OutputFcn, ...
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


% --- Executes just before mocapBrowserSettings is made visible.
function mocapBrowserSettings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};
if length(browserObj.channelIndex) == browserObj.streamHandle.numberOfChannels/3
    set(handles.edit2,'String',[num2str(browserObj.channelIndex(1)) ':' num2str(browserObj.channelIndex(end))]);
else
    set(handles.edit2,'String',num2str(browserObj.channelIndex));
end

speedList = get(handles.popupmenu2,'String');
sList = zeros(length(speedList),1);
for it=1:length(speedList), A = speedList{it}; A(A == 'x') = [];sList(it) = str2double(A);end
if browserObj.speed < 1 
    speed = -round(1/browserObj.speed);
else
    speed = browserObj.speed;
end
set(handles.popupmenu2,'Value',find(sList == speed));
if isa(browserObj.master,'browserHandleList'), set(handles.popupmenu2,'Enable','off');end
set(handles.checkbox3,'Value',browserObj.showChannelNumber);
set(handles.checkbox4,'Value',browserObj.onscreenDisplay);
Data = {browserObj.roomSize.x(1) browserObj.roomSize.x(2);browserObj.roomSize.y(1) browserObj.roomSize.y(2);browserObj.roomSize.z(1) browserObj.roomSize.z(2);};
set(handles.uitable1,'Data',Data)

path = [browserObj.streamHandle.container.container.path filesep 'skin'];
CData = imread([path filesep 'selectColor.png']);
set(handles.pushbutton4,'CData',CData);
set(handles.pushbutton5,'CData',CData);
set(handles.pushbutton6,'CData',CData);
set(handles.pushbutton7,'CData',CData);
set(handles.edit21,'String',num2str(browserObj.lineWidth));

set(handles.text18,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text19,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text20,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text21,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text22,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text6,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.text9,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.uipanel1,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.checkbox3,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.checkbox4,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.figure1,'userData',browserObj,'Color',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);
set(handles.uipanel6,'BackgroundColor',browserObj.streamHandle.container.container.preferences.gui.backgroundColor);





% --- Outputs from this function are returned to the command line.
function varargout = mocapBrowserSettings_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in done.
function done_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');

tmp = sort(eval(['[' get(handles.edit2,'String') '];']));
if isempty(tmp)
    error('Markes must be an integer or a vector of integers.');
elseif isnan(tmp)
    error('Marker must be an integer or a vector of integers.');
elseif tmp(end) > browserObj.streamHandle.numberOfChannels/3;
    error(['Index exceeds markers dimensions. The maximum number of channels is ' num2str(browserObj.streamHandle.numberOfChannels/3)]);
elseif tmp(1) < 1;
    error(['Index exceeds markers dimensions. The maximum number of channels is ' num2str(browserObj.streamHandle.numberOfChannels/3)]);
else
    userData.channels = tmp;
end

speed = get(handles.popupmenu2,'String');
index = get(handles.popupmenu2,'Value');
speed = speed{index};
speed(speed=='x') = [];
speed = str2double(speed);
if speed < 1, speed = 1/abs(speed);end
userData.speed = speed;

userData.showChannelNumber = get(handles.checkbox3,'Value');
userData.onscreenDisplay = get(handles.checkbox4,'Value');

userData.newColor = get(handles.pushbutton4,'userData');
userData.lineColor = get(handles.pushbutton5,'userData');
userData.floorColor = get(handles.pushbutton6,'userData');
userData.background = get(handles.pushbutton7,'userData');
userData.lineWidth = str2double(get(handles.edit21,'String'));
userData.limits = cell2mat(get(handles.uitable1,'Data'));

if ~isempty(userData.newColor)
    userData.newColor.channel = eval(['[' get(handles.edit19,'String') ']']);
    if any(isnan(userData.newColor.channel)), userData.newColor.channel = [];end
    if isempty(userData.newColor.channel), userData.newColor.channel = [];end
    %if length(userData.newColor.channel) > 1, userData.newColor.channel = [];end
    
    I = ismember(userData.channels,userData.newColor.channel);
    if ~any(I), userData.newColor.channel = [];end
end

set(handles.figure1,'userData',userData);
uiresume


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
close(handles.figure1);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



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



function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popupmenu2 as text
%        str2double(get(hObject,'String')) returns contents of popupmenu2 as a double


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox5.
function checkbox5_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox5



function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
newColor.color = uisetcolor;
if ~length(newColor.color) == 3, newColor = [];end
set(hObject,'userData',newColor);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
lineColor = uisetcolor;
if ~length(lineColor) == 3, lineColor = [];end
set(hObject,'userData',lineColor);


function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
floorColor = uisetcolor;
if ~length(floorColor) == 3, floorColor = [];end
set(hObject,'userData',floorColor);


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
background = uisetcolor;
if ~length(background) == 3, background = [];end
set(hObject,'userData',background);
