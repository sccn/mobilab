function varargout = EventsEditor2(varargin)
% EVENTSEDITOR2 MATLAB code for EventsEditor2.fig
%      EVENTSEDITOR2, by itself, creates a new EVENTSEDITOR2 or raises the existing
%      singleton*.
%
%      H = EVENTSEDITOR2 returns the handle to a new EVENTSEDITOR2 or the handle to
%      the existing singleton*.
%
%      EVENTSEDITOR2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EVENTSEDITOR2.M with the given input arguments.
%
%      EVENTSEDITOR2('Property','Value',...) creates a new EVENTSEDITOR2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EventsEditor2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EventsEditor2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EventsEditor2

% Last Modified by GUIDE v2.5 11-Jun-2012 16:19:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EventsEditor2_OpeningFcn, ...
                   'gui_OutputFcn',  @EventsEditor2_OutputFcn, ...
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


% --- Executes just before EventsEditor2 is made visible.
function EventsEditor2_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

if isempty(varargin),
    try
        varargin{1} = evalin('base','mobilab');
    catch %#ok
        error('MoBILAB:noRunning','You have to have MoBILAB running. Try ''runmobilab'' first.');
    end
end
handles.mobilab = varargin{1};
guidata(hObject, handles);        

allDataStreams = handles.mobilab.allStreams;
userData.inhibitedWindowLength = 0.6; % 600 ms
set(handles.figure1,'userData',userData);
count = 1;
for it=1:length(allDataStreams.item)
    if isa(allDataStreams.item{it},'vectorMeasureInSegments') && isa(allDataStreams.item{it},'projectedMocap') &&...
            ~strcmp(allDataStreams.item{it}.parentCommand.commandName,'smoothDerivative')
        segStreamNames{count,1} = allDataStreams.item{it}.name; %#ok
        segStreamUUIDs{count,1} = allDataStreams.item{it}.uuid; %#ok
        count = count+1;
    end
    streamNames{it,1} = allDataStreams.item{it}.name; %#ok
    streamUUIDs{it,1} = allDataStreams.item{it}.uuid; %#ok
end
if count == 1, errordlg('Run PCA on mocap data first.');return;end
if length(streamNames) == 1, streamNames = streamNames{1};end
if ~exist('segStreamNames','var'), segStreamNames = {''};end
set(handles.popupmenu3,'String',segStreamNames,'userData',segStreamUUIDs,'Value',1);
popupmenu3_Callback(handles.popupmenu3, [], handles);
set(handles.popupmenu1,'String',streamNames,'userData',streamUUIDs,'Value',1);
popupmenu1_Callback(handles.popupmenu1, [], handles)


% --- Outputs from this function are returned to the command line.
function varargout = EventsEditor2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;




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

function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


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

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


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


% --- Executes on selection change in popupmenu5.
function popupmenu5_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu5


% --- Executes during object creation, after setting all properties.
function popupmenu5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
allDataStreams = handles.mobilab.allStreams;
channel = str2double(get(handles.popupmenu4,'string'));
channel = channel(get(handles.popupmenu4,'Value'));
uuids = get(handles.popupmenu14,'userData');
index = get(handles.popupmenu14,'value');
dataObjectIndex = allDataStreams.findItem(uuids{index});

uuids = get(handles.popupmenu3,'userData');
index = get(handles.popupmenu3,'value');
placeEventsInThisItem = allDataStreams.findItem(uuids{index});

name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
criteria = name{index};

userData = get(handles.figure1,'userData');
inhibitedWindowLength = userData.inhibitedWindowLength;

userData = get(handles.pushbutton14,'userData');
if isempty(userData)
    if get(handles.radiobutton1,'Value')
        fprintf('Running:\n   allDataStreams.item{%i}.createEventsFromMagnitude(%i,%1.1f,%i,''%s'');\n',...
            dataObjectIndex,channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
        I = allDataStreams.item{dataObjectIndex}.createEventsFromMagnitude(channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
    elseif get(handles.radiobutton2,'Value')
        fprintf('Running:\n   allDataStreams.item{%i}.createEventsFromAngle(%i,%1.1f,%i);\n',...
            dataObjectIndex,channel,inhibitedWindowLength,placeEventsInThisItem);
        I = allDataStreams.item{dataObjectIndex}.createEventsFromAngle(channel,inhibitedWindowLength,placeEventsInThisItem);
    else
        fprintf('Running:\n   allDataStreams.item{dataObjectIndex}.createEventsFromChannel(%i,%1.1f,%i,''%s'');\n',...
            channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
        I = allDataStreams.item{dataObjectIndex}.createEventsFromChannel(channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
    end
else
    I = userData.I;
    criteria = userData.criteria;
    allDataStreams.item{placeEventsInThisItem}.event = allDataStreams.item{placeEventsInThisItem}.event.addEvent(I,criteria);
end
if ~isempty(placeEventsInThisItem), allDataStreams.save;end
set(handles.pushbutton14,'userData',[]);
disp('Done!!!')



% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
allDataStreams = handles.mobilab.allStreams;
uuids = get(handles.popupmenu3,'userData');
index = get(handles.popupmenu3,'Value');
if isempty(uuids{index}), return;end
index = allDataStreams.findItem(uuids{index});
channels = zeros(allDataStreams.item{index}.numberOfChannels,1);
for it=1:allDataStreams.item{index}.numberOfChannels
    channels(it) = str2double(allDataStreams.item{index}.label{it}(2:end));
end
channels = unique(channels);
channels = 1:length(channels);
strCh = cell(length(channels),1);
for it=1:length(channels), strCh{it} = num2str(channels(it));end
set(handles.popupmenu4,'String',strCh);
set(handles.popupmenu4,'Value',1);


set(handles.radiobutton1,'Value',1);
set(handles.radiobutton2,'Value',0);
set(handles.radiobutton3,'Value',0);

indices = allDataStreams.gObj.getDescendants(index+1)-1;
if isempty(indices), set(handles.popupmenu14,'String','');return;end
name = cell(length(indices),1);
uuids = cell(length(indices),1);
for it=1:length(indices)
    name{it} = allDataStreams.item{indices(it)}.name;
    uuids{it} = allDataStreams.item{indices(it)}.uuid;
end
if length(name)==1, name = name{1};end
set(handles.popupmenu14,'String',name,'Value',1,'userData',uuids);



% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)




% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
allDataStreams = handles.mobilab.allStreams;
names = get(hObject,'string');
uuids = get(hObject,'userData');
index = get(hObject,'value');

if ~strcmp(names{index},'EEG')
    index = allDataStreams.findItem(uuids{index});
    events = allDataStreams.item{index}.event.uniqueLabel;
else
    EEG = evalin('base','EEG');
    events = EEG.event;
    events = {events(:).type};
    events = unique(events);
end
if isempty(events), events = ' ';end
set(handles.popupmenu2,'value',1)
set(handles.popupmenu2,'String',events)



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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
allDataStreams = handles.mobilab.allStreams;
uuids = get(handles.popupmenu1,'userData');
index = get(handles.popupmenu1,'value');

type = get(handles.popupmenu2,'string');
index2 = get(handles.popupmenu2,'value');
try
    type = type{index2};
catch %#ok
    return;
end
index = allDataStreams.findItem(uuids{index});
allDataStreams.item{index}.event = allDataStreams.item{index}.event.deleteAllEventsWithThisLabel(type);
popupmenu1_Callback(handles.popupmenu1, eventdata, handles);



% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
allDataStreams = handles.mobilab.allStreams;
channel = str2double(get(handles.popupmenu4,'string'));
channel = channel(get(handles.popupmenu4,'Value'));
uuids = get(handles.popupmenu3,'userData');
index = get(handles.popupmenu3,'value');
placeEventsInThisItem = allDataStreams.findItem(uuids{index});

uuids = get(handles.popupmenu14,'userData');
index = get(handles.popupmenu14,'value');
dataObjectIndex = allDataStreams.findItem(uuids{index});


name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
criteria = name{index};

userData = get(handles.figure1,'userData');
inhibitedWindowLength = userData.inhibitedWindowLength;

if get(handles.radiobutton1,'Value')
    fprintf('Running:\n   allDataStreams.item{%i}.createEventsFromMagnitude(%i,%1.1f,%i,''%s'');\n',dataObjectIndex,...
        channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
    I = allDataStreams.item{dataObjectIndex}.createEventsFromMagnitude(channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
elseif get(handles.radiobutton2,'Value')
    fprintf('Running:\n   allDataStreams.item{%i}.createEventsFromAngle(%i,%1.1f,%i);\n',dataObjectIndex,...
        channel,inhibitedWindowLength,placeEventsInThisItem);
    I = allDataStreams.item{dataObjectIndex}.createEventsFromChannel(channel,inhibitedWindowLength,placeEventsInThisItem);
else
    fprintf('Running:\n   allDataStreams.item{%i}.createEventsFromChannel(%i,%1.1f,%i,''%s'');\n',dataObjectIndex,...
        channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
    I = allDataStreams.item{dataObjectIndex}.createEventsFromChannel(channel,inhibitedWindowLength,placeEventsInThisItem,criteria);
end

II = ismember(allDataStreams.item{placeEventsInThisItem}.event.latencyInFrame,I);
userData.I = I;
userData.criteria = char(unique(allDataStreams.item{placeEventsInThisItem}.event.label(II)));
set(handles.pushbutton14,'userData',userData);

dt = 30;
defaults.startTime = allDataStreams.item{dataObjectIndex}.timeStamp(min(I))-dt;
defaults.endTime  = allDataStreams.item{dataObjectIndex}.timeStamp(max(I))+dt;
defaults.channelIndex = channel;
if defaults.endTime-defaults.startTime < 5
    defaults.windowWidth = defaults.endTime-defaults.startTime;
else
    defaults.windowWidth = dt;
end
if isa(allDataStreams.item{dataObjectIndex},'projectedMocap')
    browserObj = allDataStreams.item{placeEventsInThisItem}.projectionBrowser(defaults);
elseif isa(allDataStreams.item{dataObjectIndex},'vectorMeasureInSegments')
    browserObj = allDataStreams.item{placeEventsInThisItem}.vectorBrowser(defaults);
else
    browserObj = allDataStreams.item{placeEventsInThisItem}.dataStreamBrowser(defaults);
end
uiwait(browserObj.figureHandle);

allDataStreams.item{placeEventsInThisItem}.event = allDataStreams.item{placeEventsInThisItem}.event.deleteAllEventsWithThisLabel(userData.criteria);



function popupmenu4_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popupmenu4 as text
%        str2double(get(hObject,'String')) returns contents of popupmenu4 as a double


% --- Executes during object creation, after setting all properties.
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% Settings
function pushbutton17_Callback(hObject, eventdata, handles)
userData = get(handles.figure1,'userData');
userData.figure1 = handles.figure1;
EventEditorSettings(userData);


% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
delete(hObject);



% --- Executes on selection change in popupmenu14.
function popupmenu14_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu14 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu14


% --- Executes during object creation, after setting all properties.
function popupmenu14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when selected object is changed in uipanel7.
function uipanel7_SelectionChangeFcn(hObject, eventdata, handles)
if get(handles.radiobutton1,'Value') || get(handles.radiobutton2,'Value')
    set(handles.popupmenu6,'Enable','off');
else
    set(handles.popupmenu6,'Enable','on');
end


% --- Executes during object creation, after setting all properties.
function uipanel7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uipanel7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
