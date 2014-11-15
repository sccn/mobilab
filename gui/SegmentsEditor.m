function varargout = SegmentsEditor(varargin)
% SEGMENTSEDITOR MATLAB code for SegmentsEditor.fig
%      SEGMENTSEDITOR, by itself, creates a new SEGMENTSEDITOR or raises the existing
%      singleton*.
%
%      H = SEGMENTSEDITOR returns the handle to a new SEGMENTSEDITOR or the handle to
%      the existing singleton*.
%
%      SEGMENTSEDITOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEGMENTSEDITOR.M with the given input arguments.
%
%      SEGMENTSEDITOR('Property','Value',...) creates a new SEGMENTSEDITOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SegmentsEditor_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SegmentsEditor_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SegmentsEditor

% Last Modified by GUIDE v2.5 10-Oct-2011 17:50:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SegmentsEditor_OpeningFcn, ...
                   'gui_OutputFcn',  @SegmentsEditor_OutputFcn, ...
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


% --- Executes just before SegmentsEditor is made visible.
function SegmentsEditor_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
try
    allDataStreams = evalin('base','allDataStreams');
catch ME
    ME.rethrow;
end
if isempty(varargin)    
    segmentName = {''};
    objName = {};
    for it=1:length(allDataStreams.item)
        if isa(allDataStreams.item{it},'segmentList')
            segmentName = cell(length(allDataStreams.item{it}.item),1);
            for jt=1:length(allDataStreams.item{it}.item)
                segmentName{jt} = allDataStreams.item{it}.item{jt}.segmentName;
            end
        else
            objName{end+1} = allDataStreams.item{it}.name;%#ok
        end
    end
    set(handles.popupmenu5,'String',objName);
    set(handles.popupmenu5,'Value',1);
    
    set([handles.popupmenu10 handles.popupmenu11],'String',segmentName);
    set([handles.popupmenu10 handles.popupmenu11],'Value',1);
    set(handles.checkbox1,'Enable','off');
else
    sinkStreams = get(handles.edit8,'string');
    if isempty(sinkStreams), sinkStreams = {''};end
    if ischar(sinkStreams), sinkStreams = {sinkStreams};end
    sinkStreams = unique(sinkStreams);
    if ~any(ismember(sinkStreams,{allDataStreams.item{varargin{1}}.name}))
        if isempty(sinkStreams{end}), sinkStreams(end) = [];end
        sinkStreams{end+1} = allDataStreams.item{varargin{1}}.name;
        set(handles.edit8,'string',sinkStreams);
        if ~isempty(strfind(allDataStreams.item{varargin{1}}.name,'biosemi'))
            set(handles.checkbox1,'Enable','on');
        end
    end
end


% --- Outputs from this function are returned to the command line.
function varargout = SegmentsEditor_OutputFcn(hObject, eventdata, handles) 
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




% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
segmentName = get(handles.edit6,'string');
name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
allDataStreams = evalin('base','allDataStreams');
eventObjectIndex = allDataStreams.findItem(name{index});

startMark = get(handles.listbox3,'string');
endMark = get(handles.listbox5,'string');

if any([isempty(startMark) isempty(endMark)])
    errordlg('Select first the ''start'' and ''end'' events');
    return
end
if isempty(segmentName), segmentName = [startMark{1} '_' endMark{1}];end
hwait = waitdlg([],'Creating segment...');
segmentObj = basicSegment(allDataStreams.item{eventObjectIndex},startMark,endMark,segmentName);
close(hwait);
try
    segmenListObjIndex = allDataStreams.findItem('segmentList');
    allDataStreams.item{segmenListObjIndex} = allDataStreams.item{segmenListObjIndex}.addSegment(segmentObj);
catch ME
    if ~isempty(strfind(ME.message,'Wrong index.'))
        segmentList(segmentObj,allDataStreams);
    else
        ME.rethrow;
    end
end
allDataStreams.save
assignin('base','allDataStreams',allDataStreams);
popupmenu10_Callback(handles.popupmenu10, eventdata, handles);
    

% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
segmentName = get(handles.edit6,'string');
name = get(handles.popupmenu5,'string');
index = get(handles.popupmenu5,'value');
allDataStreams = evalin('base','allDataStreams');
eventObjectIndex = allDataStreams.findItem(name{index});

startMark = get(handles.listbox3,'string');
endMark = get(handles.listbox5,'string');

if any([isempty(startMark) isempty(endMark)])
    errordlg('Select first the ''start'' and ''end'' events');
    return
end
if isempty(segmentName), segmentName = [startMark{1} '_' endMark{1}];end
hwait = waitdlg([],'Creating segment...');
segmentObj = basicSegment(allDataStreams.item{eventObjectIndex},startMark,endMark,segmentName);
segDataObj = segmentObj.apply(allDataStreams.item{eventObjectIndex});
close(hwait);
browserObj = segDataObj.segmentBrowser;
uiwait(browserObj.figureHandle);
allDataStreams.deleteItem(length(allDataStreams.item));




% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu10.
function popupmenu10_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
segmentName = {''};
for it=1:length(allDataStreams.item)
    if isa(allDataStreams.item{it},'segmentList')
        segmentName = cell(length(allDataStreams.item{it}.item),1);
        for jt=1:length(allDataStreams.item{it}.item)
            segmentName{jt} = allDataStreams.item{it}.item{jt}.segmentName;
        end     
    end
end
if isempty(segmentName), segmentName = {''};end
set([handles.popupmenu10 handles.popupmenu11],'String',segmentName);
%set([handles.popupmenu10 handles.popupmenu11],'Value',1);

val = get(handles.popupmenu10,'Value');
if val>length(segmentName), set(handles.popupmenu10,'Value',1);end

val = get(handles.popupmenu11,'Value');
if val>length(segmentName), set(handles.popupmenu11,'Value',1);end


% --- Executes during object creation, after setting all properties.
function popupmenu10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton20.
function pushbutton20_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
name = get(handles.popupmenu10,'String');
index = get(handles.popupmenu10,'value');
name = name{index};

segmentObjIndex = [];
for it=1:length(allDataStreams.item)
    if isa(allDataStreams.item{it},'segmentList')
        segmentObjIndex = it;
        break
    end
end
if isempty(segmentObjIndex), errordlg('Cannot find the ''segmentList'' object in the dataSource.');return;end
try
    index = allDataStreams.item{segmentObjIndex}.findItem(name);
    allDataStreams.item{segmentObjIndex}.item(index) = [];
end
allDataStreams.save
popupmenu10_Callback(handles.popupmenu10,eventdata,handles);


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


% --- Executes on selection change in popupmenu5.
function popupmenu5_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
names = get(hObject,'string');
index = get(hObject,'value');
index = allDataStreams.findItem(names{index});
events = allDataStreams.item{index}.event.uniqueLabel;
set(handles.listbox2,'String',events)


% --- Executes on selection change in popupmenu11.
function popupmenu11_Callback(hObject, eventdata, handles)
popupmenu10_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function popupmenu11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


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


% --- Executes on button press in pushbutton22.
function pushbutton22_Callback(hObject, eventdata, handles)
allDataStreams = evalin('base','allDataStreams');
allDataStreams.viewLogicStructure('addExport3_Callback');


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes on button press in pushbutton23.
function pushbutton23_Callback(hObject, eventdata, handles)
segmentName = get(handles.popupmenu11,'string');
index = get(handles.popupmenu11,'value');
segmentName = segmentName{index};

name = get(handles.edit8,'string');
name = unique(name);
if isempty(name), return;end
if ~iscellstr(name)
    if ischar(name), 
        name = {name};
    else
        return
    end
end
    

allDataStreams = evalin('base','allDataStreams');
segmenListObjIndex = [];
for it=1:length(allDataStreams.item)
    if isa(allDataStreams.item{it},'segmentList')
        segmenListObjIndex = it;
        break
    end
end
if isempty(segmenListObjIndex), errordlg('Cannot find the list of segments inside the dataSource object.');return;end
segmentIndex = allDataStreams.item{segmenListObjIndex}.findItem(segmentName);
hwait = waitbar(0,'Creating new objects...');
set(hwait,'color',[0.66 0.76 1]);
drawnow;
for it=1:length(name)
    streamObjIndex = allDataStreams.findItem(name{it});
    allDataStreams.item{segmenListObjIndex}.item{segmentIndex}.apply(allDataStreams.item{streamObjIndex});
    if get(handles.checkbox1,'value')
        if ~isempty(strfind(name{it},'biosemi'))
            allDataStreams.item{end}.export2EEGLAB;
        end
    end
    waitbar(it/length(name),hwait);
end
close(hwait);


% --- Executes on button press in pushbutton24.
function pushbutton24_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
