function varargout = browserHandleListSettings(varargin)
% BROWSERHANDLELISTSETTINGS MATLAB code for browserHandleListSettings.fig
%      BROWSERHANDLELISTSETTINGS, by itself, creates a new BROWSERHANDLELISTSETTINGS or raises the existing
%      singleton*.
%
%      H = BROWSERHANDLELISTSETTINGS returns the handle to a new BROWSERHANDLELISTSETTINGS or the handle to
%      the existing singleton*.
%
%      BROWSERHANDLELISTSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BROWSERHANDLELISTSETTINGS.M with the given input arguments.
%
%      BROWSERHANDLELISTSETTINGS('Property','Value',...) creates a new BROWSERHANDLELISTSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before browserHandleListSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to browserHandleListSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help browserHandleListSettings

% Last Modified by GUIDE v2.5 21-Nov-2011 16:23:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @browserHandleListSettings_OpeningFcn, ...
                   'gui_OutputFcn',  @browserHandleListSettings_OutputFcn, ...
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


% --- Executes just before browserHandleListSettings is made visible.
function browserHandleListSettings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};

set(handles.edit2,'String',num2str(browserObj.startTime));
set(handles.edit3,'String',num2str(browserObj.endTime));
set(handles.edit5,'String',num2str(browserObj.font.size));
value = find(ismember(get(handles.popupmenu2,'String'),{browserObj.font.weight}));
set(handles.popupmenu2,'Value',value);

speedList = get(handles.popupmenu1,'String');
sList = zeros(length(speedList),1);
for it=1:length(speedList), A = speedList{it}; A(A == 'x') = [];sList(it) = str2double(A);end
if browserObj.speed < 1
    speed = -1/browserObj.speed;
else
    speed = browserObj.speed;
end
set(handles.popupmenu1,'Value',find(sList == speed));




% --- Outputs from this function are returned to the command line.
function varargout = browserHandleListSettings_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
uiresume


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
try
    speed = get(handles.popupmenu1,'String');
    index = get(handles.popupmenu1,'Value');
    speed = speed{index};
    speed(speed=='x') = [];
    speed = str2double(speed);
    if speed < 1, speed = 1/abs(speed);    end
    userData.speed = speed;
    
    userData.startTime = str2double(get(handles.edit2,'String'));
    userData.endTime = str2double(get(handles.edit3,'String'));
    if ~(isnumeric(userData.startTime) && isnumeric(userData.endTime)), error('Enter a number.');end
    
    userData.font.size = str2double(get(handles.edit5,'String'));
    if ~isnumeric(userData.font.size), error('Enter a number.');end

    value = get(handles.popupmenu2,'Value');
    weight = get(handles.popupmenu2,'String');
    userData.font.weight = weight{value};
    
    set(handles.figure1,'userData',userData);
    uiresume
catch ME
    uiresume
    errordlg2(ME.message);
    return
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
