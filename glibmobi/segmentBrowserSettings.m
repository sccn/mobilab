function varargout = segmentBrowserSettings(varargin)
% SEGMENTBROWSERSETTINGS MATLAB code for segmentBrowserSettings.fig
%      SEGMENTBROWSERSETTINGS, by itself, creates a new SEGMENTBROWSERSETTINGS or raises the existing
%      singleton*.
%
%      H = SEGMENTBROWSERSETTINGS returns the handle to a new SEGMENTBROWSERSETTINGS or the handle to
%      the existing singleton*.
%
%      SEGMENTBROWSERSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEGMENTBROWSERSETTINGS.M with the given input arguments.
%
%      SEGMENTBROWSERSETTINGS('Property','Value',...) creates a new SEGMENTBROWSERSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before segmentBrowserSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to segmentBrowserSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help segmentBrowserSettings

% Last Modified by GUIDE v2.5 22-Oct-2011 18:19:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @segmentBrowserSettings_OpeningFcn, ...
    'gui_OutputFcn',  @segmentBrowserSettings_OutputFcn, ...
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


% --- Executes just before segmentBrowserSettings is made visible.
function segmentBrowserSettings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};

set(handles.edit1,'String',num2str(browserObj.gain));
if browserObj.numberOfChannelsToPlot == browserObj.streamHandle.numberOfChannels
    set(handles.edit2,'String',['1:' num2str(browserObj.numberOfChannelsToPlot(end))]);
else
    set(handles.edit2,'String',num2str(browserObj.channelIndex(:)'));
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
if strcmp(class(browserObj.master),'browserHandleList')
    set(handles.popupmenu2,'Enable','off');
end
set(handles.edit4,'String',num2str(browserObj.windowWidth));
set(handles.checkbox1,'Value',browserObj.normalizeFlag);
set(handles.checkbox3,'Value',browserObj.showChannelNumber);
set(handles.checkbox4,'Value',browserObj.onscreenDisplay);
cmap = get(handles.popupmenu1,'String');
I = ismember(cmap,browserObj.colormap);
set(handles.popupmenu1,'Value',find(I));
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep 'selectColor.png']);
set(handles.pushbutton4,'CData',CData);
set(handles.figure1,'userData',browserObj);





% --- Outputs from this function are returned to the command line.
function varargout = segmentBrowserSettings_OutputFcn(hObject, eventdata, handles)
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
    tmp = str2double(get(handles.edit1,'String'));
    if isempty(tmp), error('Gain must be a number.');end
    if isnan(tmp), error('Gain must be a number.');end
    userData.gain = tmp;
    tmp = sort(eval(['[' get(handles.edit2,'String') '];']));
    if isempty(tmp)
        error('Channels must be an integer or a vector of integers.');
    elseif isnan(tmp)
        error('Channels must be an integer or a vector of integers.');
    elseif tmp(end) > browserObj.streamHandle.numberOfChannels;
        error(['Index exceeds channels dimensions. The maximum number of channels is ' num2str(browserObj.streamHandle.numberOfChannels)]);
    elseif tmp(1) < 1;
        error(['Index exceeds channels dimensions. The maximum number of channels is ' num2str(browserObj.streamHandle.numberOfChannels)]);
    else
        userData.channels = tmp;
    end
    
    tmp = str2double(get(handles.edit4,'String'));
    if isempty(tmp), error('The width must be a number.');end
    if isnan(tmp), error('The width must be a number.');end
    userData.windowWidth = tmp;
        
    browserObj.nowCursor = browserObj.streamHandle.originalStreamObj.timeStamp(browserObj.timeIndex(1)) + userData.windowWidth/2;
    
    speed = get(handles.popupmenu2,'String');
    index = get(handles.popupmenu2,'Value');
    speed = speed{index};
    speed(speed=='x') = [];
    speed = str2double(speed);
    if speed < 1, speed = 1/abs(speed);end
    userData.speed = speed;
          
    userData.normalizeFlag     = get(handles.checkbox1,'Value');
    userData.showChannelNumber = get(handles.checkbox3,'Value');
    userData.onscreenDisplay   = get(handles.checkbox4,'Value');
    
    userData.colormap = get(handles.popupmenu1,'String');
    userData.colormap = userData.colormap{get(handles.popupmenu1,'Value')};
    userData.newColor = get(handles.pushbutton4,'userData');
    
    if ~isempty(userData.newColor)
        userData.newColor.channel = eval(['[' get(handles.edit11,'String') ']']);
        if any(isnan(userData.newColor.channel)), userData.newColor.channel = [];end
        if isempty(userData.newColor.channel), userData.newColor.channel = [];end
        
        I = ismember(userData.channels,userData.newColor.channel);
        if ~any(I), userData.newColor.channel = [];end
        if ~isempty(userData.newColor.channel), userData.colormap = 'custom';end
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


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
newColor = get(hObject,'userData');
newColor.color = uisetcolor;
if ~length(newColor.color) == 3, return;end
set(hObject,'userData',newColor);


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
