function varargout = CreateEvent(varargin)
% CREATEEVENT MATLAB code for CreateEvent.fig
%      CREATEEVENT, by itself, creates a new CREATEEVENT or raises the existing
%      singleton*.
%
%      H = CREATEEVENT returns the handle to a new CREATEEVENT or the handle to
%      the existing singleton*.
%
%      CREATEEVENT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CREATEEVENT.M with the given input arguments.
%
%      CREATEEVENT('Property','Value',...) creates a new CREATEEVENT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CreateEvent_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CreateEvent_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CreateEvent

% Last Modified by GUIDE v2.5 21-Nov-2011 13:45:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CreateEvent_OpeningFcn, ...
                   'gui_OutputFcn',  @CreateEvent_OutputFcn, ...
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


% --- Executes just before CreateEvent is made visible.
function CreateEvent_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

if isempty(varargin), return;end
browserListObj = varargin{1};
set(handles.figure1,'userData',browserListObj);
set(handles.figure1,'Color',get(browserListObj.master,'Color'))
set(handles.text1,'BackgroundColor',get(browserListObj.master,'Color'))
set(handles.text2,'BackgroundColor',get(browserListObj.master,'Color'))
set(handles.text3,'BackgroundColor',get(browserListObj.master,'Color'))
set(handles.edit1,'String',num2str(browserListObj.nowCursor));
existEEG = evalin('base','exist(''EEG'',''var'')');
strNames = cell(length(browserListObj.list)+existEEG,1);
uuids = cell(length(browserListObj.list)+existEEG,1);
for it=1:length(browserListObj.list)
    strNames{it} = browserListObj.list{it}.streamHandle.name;
    uuids{it} = browserListObj.list{it}.streamHandle.uuid;
end
if existEEG, strNames{end} = 'EEG';end
set(handles.popupmenu1,'String',strNames,'Value',1,'userData',uuids);



% --- Outputs from this function are returned to the command line.
function varargout = CreateEvent_OutputFcn(hObject, eventdata, handles) 
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
try
    browserListObj = get(handles.figure1,'userData');
    if strcmp(browserListObj.timerObj.Running,'on')
        resume_timer = true;
        % stop(browserListObj.timerObj);
        browserListObj.play;
        pause(0.01)
        disp 'timer stop'
    else
        resume_timer = false;
    end
    mobilab = evalin('base','mobilab');
    allDataStreams = mobilab.allStreams;
    latency = str2double(get(handles.edit1,'String'));
    if isempty(latency), error('Invalid latency');end
    if ~isnumeric(latency), error('Invalid latency');end
    if ~(latency >= 0 && latency <= allDataStreams.item{1}.timeStamp(end)), error('This latency is out of range');end
    
    label = get(handles.edit2,'String');
    if isempty(label), error('You must enter a label (EEG event type).');end
    
    names = get(handles.popupmenu1,'String');
    index = get(handles.popupmenu1,'Value');
    insertHere = names{index};
    uuids = get(handles.popupmenu1,'userData');
    
    if strcmp(insertHere,'EEG')
        EEG = evalin('base','EEG');
        for it=1:length(allDataStreams.item)
            if length(allDataStreams.item{it}.timeStamp) == EEG.pnts, break;end
        end
        [~,latency] = min(abs(allDataStreams.item{it}.timeStamp-latency));
        EEG = eeg_addnewevents(EEG, {latency}, {label});
        assignin('base','EEG',EEG);
        evalin('base','eeglab(''redraw'')');
    else
        itemIndex = allDataStreams.findItem(uuids{index});
        latency = allDataStreams.item{itemIndex}.getTimeIndex(latency);
        allDataStreams.item{itemIndex}.event = allDataStreams.item{itemIndex}.event.addEvent(latency,label);  
    end
    if resume_timer
        %start(browserListObj.timerObj);
        browserListObj.play;
        disp 'timer resume'
    end
catch ME
    errordlg(ME.message);
end




% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
close(handles.figure1);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
mobilab = evalin('base','mobilab');
allDataStreams = mobilab.allStreams;

count = 1;
for it=1:length(allDataStreams.item)
    if isa(allDataStreams.item{it},'vectorMeasureInSegments')&&false || isa(allDataStreams.item{it},'projectedMocap')
        segStreamNames{count,1} = allDataStreams.item{it}.name; %#ok
        segStreamUUIDs{count,1} = allDataStreams.item{it}.uuid; %#ok
        count = count+1;
    end
end
if count == 1, errordlg('Run PCA on mocap data first.');return;end
%EventsEditor2;
mobilab.eventsEditor;