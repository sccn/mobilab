function varargout = EventsEditor(varargin)
% EVENTSEDITOR MATLAB code for EventsEditor.fig
%      EVENTSEDITOR, by itself, creates a new EVENTSEDITOR or raises the existing
%      singleton*.
%
%      H = EVENTSEDITOR returns the handle to a new EVENTSEDITOR or the handle to
%      the existing singleton*.
%
%      EVENTSEDITOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EVENTSEDITOR.M with the given input arguments.
%
%      EVENTSEDITOR('Property','Value',...) creates a new EVENTSEDITOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EventsEditor_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EventsEditor_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EventsEditor

% Last Modified by GUIDE v2.5 12-Sep-2011 15:37:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EventsEditor_OpeningFcn, ...
                   'gui_OutputFcn',  @EventsEditor_OutputFcn, ...
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


% --- Executes just before EventsEditor is made visible.
function EventsEditor_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

if isempty(varargin)
    try
        allDataStreams = evalin('base','allDataStreams');
        flag = evalin('base','exist(''EEG'',''var'')');
    catch ME
        ME.rethrow;
    end
    
    userData.percentMaxVelocityAssumedStopped = 0.05;
    userData.inhibitedWindowLength = 0.6; % 600 ms
    set(handles.figure1,'userData',userData);

    streamNames = cell(length(allDataStreams.item),1);
    for it=1:length(allDataStreams.item)
        streamNames{it} = allDataStreams.item{it}.name;
    end
    if length(allDataStreams.item) == 1, streamNames = streamNames{1};end
    set(handles.popupmenu5,'String',streamNames);
    set(handles.popupmenu8,'String',streamNames);
    
    if flag
        streamNames{end+1} = 'EEG';
        streamNames = circshift(streamNames,1);
        set(handles.edit4,'string','EEG');
    else
        set(handles.edit4,'string','');
    end
    set(handles.popupmenu1,'String',streamNames);
    popupmenu1_Callback(handles.popupmenu1, [], handles)
    popupmenu5_Callback(handles.popupmenu5, [], handles)
else
    allDataStreams = evalin('base','allDataStreams');
    sinkStreams = get(handles.edit4,'string');
    if ischar(sinkStreams), sinkStreams = {sinkStreams};end
    sinkStreams = unique(sinkStreams);
    %sinkStreams(ismember(sinkStreams,{''})) = [];
    
    if ~any(ismember(sinkStreams,{allDataStreams.item{varargin{1}}.name}))
        sinkStreams{end+1} = allDataStreams.item{varargin{1}}.name;
        set(handles.edit4,'string',sinkStreams);
    end
end


% --- Outputs from this function are returned to the command line.
function varargout = EventsEditor_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox3.
function listbox3_Callback(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox3


% --- Executes during object creation, after setting all properties.
function listbox3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
eventsList = get(handles.listbox2,'String');
if isempty(eventsList), return;end
if ~iscell(eventsList) && ~isempty(eventsList), eventsList = {eventsList};end
index = get(handles.listbox2,'Value');
startEvents = get(handles.listbox3,'String');
if isempty(startEvents), startEvents = cell(0);end
if ischar(startEvents) && ~isempty(startEvents), startEvents = {startEvents};end
if ~iscellstr(startEvents), startEvents = cell(0);end
startEvents(end+1) = eventsList(index);
set(handles.listbox3,'String',startEvents)
set(handles.listbox3,'Value',1);
 


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
eventsList = get(handles.listbox2,'String');
if isempty(eventsList), return;end
if ~iscell(eventsList) && ~isempty(eventsList), eventsList = {eventsList};end
index = get(handles.listbox2,'Value');
startEvents = get(handles.listbox5,'String');
if isempty(startEvents), startEvents = cell(0);end
if ischar(startEvents) && ~isempty(startEvents), startEvents = {startEvents};end
if ~iscellstr(startEvents), startEvents = cell(0);end
startEvents(end+1) = eventsList(index);
set(handles.listbox5,'String',startEvents)
set(handles.listbox5,'Value',1);
 



% --- Executes on selection change in listbox4.
function listbox4_Callback(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox4


% --- Executes during object creation, after setting all properties.
function listbox4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox5.
function listbox5_Callback(hObject, eventdata, handles)
% hObject    handle to listbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox5 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox5


% --- Executes during object creation, after setting all properties.
function listbox5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
startEvents = get(handles.listbox3,'String');
index = get(handles.listbox3,'Value');
if isempty(startEvents), return;end
if ischar(startEvents) && ~isempty(startEvents), startEvents = {startEvents};end
if ~iscellstr(startEvents), startEvents = cell(0);end
startEvents(index) = [];
set(handles.listbox3,'String',startEvents)
set(handles.listbox3,'Value',1);

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
startEvents = get(handles.listbox5,'String');
index = get(handles.listbox5,'Value');
if isempty(startEvents), return;end
if ischar(startEvents) && ~isempty(startEvents), startEvents = {startEvents};end
if ~iscellstr(startEvents), startEvents = cell(0);end
startEvents(index) = [];
set(handles.listbox5,'String',startEvents)
set(handles.listbox5,'Value',1);


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


% --- Executes on selection change in popupmenu4.
function popupmenu4_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu4


% --- Executes during object creation, after setting all properties.
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
channels = str2double(get(handles.edit5,'string'));
if ~isnumeric(channels) || isempty(channels), warndlg2('You must select a channel.');return;end
name = get(handles.popupmenu8,'string');
index = get(handles.popupmenu8,'value');
allDataStreams = evalin('base','allDataStreams');
dataObjectIndex = allDataStreams.findItem(name{index});

if ~any(ismember(channels,1:allDataStreams.item{dataObjectIndex}.numberOfChannels)), errordlg('Channel out of range.');return;end

name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
eventObjectIndex = allDataStreams.findItem(name{index});

startMark = get(handles.listbox3,'string');
endMark = get(handles.listbox5,'string');

if any([isempty(startMark) isempty(endMark)])
    errordlg('Select first the ''start'' and ''end'' events');
    return
    %startMark = str2double(get(handles.edit2,'string'));
    %endMark = str2double(get(handles.edit3,'string'));
    %if any([~isnumeric(startMark) ~isnumeric(endMark)])
    %    error('Start and end latencies must be numbers.');
    %end
end
name = get(handles.popupmenu4,'string');
index = get(handles.popupmenu4,'value');
criteria = name{index};
name = get(handles.edit4,'string');

I = ismember(name,'EEG');
exportToEEG = any(I);
I = find(~I);

sinkObjIndex = zeros(length(I),1);
for it=1:length(I)
    sinkObjIndex(it) = allDataStreams.findItem(name{I(it)});
end

EEG = evalin('base','EEG');

userData = get(handles.figure1,'userData');
inhibitedWindowLength = round(userData.inhibitedWindowLength*allDataStreams.item{dataObjectIndex}.samplingRate);
percentMaxVelocityAssumedStopped = userData.percentMaxVelocityAssumedStopped;

userData = get(handles.pushbutton14,'userData');
if isempty(userData)
    createBasicEvents(allDataStreams,EEG,eventObjectIndex,startMark,endMark,criteria,dataObjectIndex,...
        channels,sinkObjIndex,exportToEEG,inhibitedWindowLength,percentMaxVelocityAssumedStopped);
else
    I = userData.I;
    onset = userData.onset;
    offset = userData.offset;
    
    if (strcmp(criteria,'all above') || strcmp(criteria,'maxima')) && ~isempty(I)
        for it=1:length(sinkObjIndex)
            allDataStreams.item{sinkObjIndex(it)}.event = allDataStreams.item{sinkObjIndex(it)}.event.addEvent(I,'mx',pi);
        end
        if exportToEEG, EEG = eeg_addnewevents(EEG, {I}, {'mx'});end
    end
    if (strcmp(criteria,'all above') || strcmp(criteria,'onset')) && ~isempty(onset)
        for it=1:length(sinkObjIndex)
            allDataStreams.item{sinkObjIndex(it)}.event = allDataStreams.item{sinkObjIndex(it)}.event.addEvent(onset,'onset',pi);
        end
        if exportToEEG, EEG = eeg_addnewevents(EEG, {onset}, {'onset'});end
    end
    
    if (strcmp(criteria,'all above') || strcmp(criteria,'offset')) && ~isempty(offset)
        for it=1:length(sinkObjIndex)
            allDataStreams.item{sinkObjIndex(it)}.event = allDataStreams.item{sinkObjIndex(it)}.event.addEvent(offset,'offset',pi);
        end
        if exportToEEG, EEG = eeg_addnewevents(EEG, {offset}, {'offset'});end
    end
    
    
    if ~isempty(sinkObjIndex), allDataStreams.save;end
    if exportToEEG
        assignin('base','EEG',EEG);
        eeglab('redraw');
    end
    set(handles.pushbutton14,'userData',[]);
end

    

% --- Executes on selection change in edit4.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns edit4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from edit4


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)



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
allDataStreams = evalin('base','allDataStreams');
names = get(hObject,'string');
index = get(hObject,'value');

if ~strcmp(names{index},'EEG')
    index = allDataStreams.findItem(names{index});
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
allDataStreams = evalin('base','allDataStreams');
names = get(handles.popupmenu1,'string');
index = get(handles.popupmenu1,'value');

type = get(handles.popupmenu2,'string');
index2 = get(handles.popupmenu2,'value');
try
    type = type{index2};
catch %#ok
    return;
end
if ~strcmp(names{index},'EEG')
    index = allDataStreams.findItem(names{index});
    allDataStreams.item{index}.event = allDataStreams.item{index}.event.deleteAllEventsWithThisLabel(type);
else
    EEG = evalin('base','EEG');
    I = zeros(length(EEG.event),1);
    for it=1:length(EEG.event)
        if strcmp(EEG.event(it).type,type)
            I(it) = it;
        end
    end
    I = nonzeros(I);
    if ~isempty(I)
        EEG.event(I) = [];
        assignin('base','EEG',EEG);
        evalin('base','eeglab(''redraw'');');
    end
end
popupmenu1_Callback(handles.popupmenu1, eventdata, handles);


% --- Executes on selection change in popupmenu5.
function popupmenu5_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
names = get(hObject,'string');
index = get(hObject,'value');
index = allDataStreams.findItem(names{index});
events = allDataStreams.item{index}.event.uniqueLabel;
set(handles.listbox2,'String',events)
set(handles.listbox2,'Value',1)


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


% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
channels = str2double(get(handles.edit5,'string'));
if ~isnumeric(channels) || isempty(channels), warndlg2('You must select a channel.');return;end
name = get(handles.popupmenu8,'string');
index = get(handles.popupmenu8,'value');
allDataStreams = evalin('base','allDataStreams');
dataObjectIndex = allDataStreams.findItem(name{index});

if ~any(ismember(channels,1:allDataStreams.item{dataObjectIndex}.numberOfChannels)), errordlg('Channel out of range.');return;end

name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
eventObjectIndex = allDataStreams.findItem(name{index});

startMark = get(handles.listbox3,'string');
endMark = get(handles.listbox5,'string');

if any([isempty(startMark) isempty(endMark)])
    errordlg('Select first the ''start'' and ''end'' events');
    return
    %startMark = str2double(get(handles.edit2,'string'));
    %endMark = str2double(get(handles.edit3,'string'));
    %if any([~isnumeric(startMark) ~isnumeric(endMark)])
    %    error('Start and end latency must be numbers.');
    %end
end
name = get(handles.popupmenu4,'string');
index = get(handles.popupmenu4,'value');
criteria = name{index};

exportToEEG = false;
sinkObjIndex = dataObjectIndex;

userData = get(handles.figure1,'userData');
inhibitedWindowLength = round(userData.inhibitedWindowLength*allDataStreams.item{dataObjectIndex}.samplingRate);
percentMaxVelocityAssumedStopped = userData.percentMaxVelocityAssumedStopped;

[I,onset,offset] = createBasicEvents(allDataStreams,[],eventObjectIndex,startMark,endMark,criteria,dataObjectIndex,...
    channels,sinkObjIndex,exportToEEG,inhibitedWindowLength,percentMaxVelocityAssumedStopped);
if isempty(I),  questdlg2('No events were found.','','Ok','Ok');return;end
userData.I = I;
userData.onset = onset;
userData.offset = offset;
set(handles.pushbutton14,'userData',userData);

dt = inhibitedWindowLength/2/allDataStreams.item{dataObjectIndex}.samplingRate;
defaults.startTime = allDataStreams.item{dataObjectIndex}.timeStamp(min([I(:)' onset(:)' offset(:)']))-dt;

defaults.endTime   = allDataStreams.item{dataObjectIndex}.timeStamp(max([I(:)' onset(:)' offset(:)']))+dt;
defaults.channels = channels;
if defaults.endTime-defaults.startTime < 5, defaults.windowWidth = defaults.endTime-defaults.startTime;end

obj = allDataStreams.item{dataObjectIndex}.dataStreamBrowser(defaults);
uiwait(obj.figureHandle);

if strcmp(criteria,'all above') || strcmp(criteria,'maxima')
    allDataStreams.item{dataObjectIndex}.event = allDataStreams.item{dataObjectIndex}.event.deleteAllEventsWithThisLabel('mx');
end
if strcmp(criteria,'all above') || strcmp(criteria,'onset')
    allDataStreams.item{dataObjectIndex}.event = allDataStreams.item{dataObjectIndex}.event.deleteAllEventsWithThisLabel('onset');
end
if strcmp(criteria,'all above') || strcmp(criteria,'offset')
    allDataStreams.item{dataObjectIndex}.event = allDataStreams.item{dataObjectIndex}.event.deleteAllEventsWithThisLabel('offset');
end
    


% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
allDataStreams.viewLogicStructure('addExport2_Callback');


% --- Executes on selection change in popupmenu8.
function popupmenu8_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu8 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu8


% --- Executes during object creation, after setting all properties.
function popupmenu8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
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


% Settings
function pushbutton17_Callback(hObject, eventdata, handles)
userData = get(handles.figure1,'userData');
userData.figure1 = handles.figure1;
EventEditorSettings(userData);
