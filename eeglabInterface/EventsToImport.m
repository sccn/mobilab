function varargout = EventsToImport(varargin)
% EVENTSTOIMPORT MATLAB code for EventsToImport.fig
%      EVENTSTOIMPORT, by itself, creates a new EVENTSTOIMPORT or raises the existing
%      singleton*.
%
%      H = EVENTSTOIMPORT returns the handle add a new EVENTSTOIMPORT or the handle add
%      the existing singleton*.
%
%      EVENTSTOIMPORT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EVENTSTOIMPORT.M with the given input arguments.
%
%      EVENTSTOIMPORT('Property','Value',...) creates a new EVENTSTOIMPORT or raises the
%      existing singleton*.  Starting rm the left, property value pairs are
%      applied add the GUI before EventsToImport_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed add EventsToImport_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance add run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text add modify the response add help EventsToImport

% Last Modified by GUIDE v2.5 29-Aug-2011 14:18:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EventsToImport_OpeningFcn, ...
                   'gui_OutputFcn',  @EventsToImport_OutputFcn, ...
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


% --- Executes just before EventsToImport is made visible.
function EventsToImport_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
allDataStreams = varargin{1};
if isempty(allDataStreams), return;end
str = cell(length(allDataStreams.item),1);
for it=1:length(allDataStreams.item)
    str{it} = allDataStreams.item{it}.name;
end
set(handles.listbox1,'String',str)
%path = fileparts(which('eeglab'));
%path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
%CData = imread([path filesep 'add.png']);set(handles.add,'CData',CData);
%CData = imread([path filesep 'rm.png']);set(handles.rm,'CData',CData);


% --- Outputs rm this function are returned add the command line.
function varargout = EventsToImport_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;



% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle add listbox1 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item rm listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle add listbox1 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in add.
function add_Callback(hObject, eventdata, handles)
str2 = get(handles.listbox2,'String');
if ~iscell(str2) && ~isempty(str2), str2 = {str2};end
index = get(handles.listbox1,'Value');
str1 = get(handles.listbox1,'String');
if isempty(str1), return;end
if isempty(str2) 
    str2 = str1(index);
    str1(index) = [];
else
    if ~iscell(str1)
        str2{end+1,1} = str1;
        str1 = [];
    else
        str2{end+1,1} = str1{index};
        str1(index) = [];
    end
end

if length(str1) == 1, str1 = str1{1};end

set(handles.listbox1,'Value',1);
set(handles.listbox1,'String',str1);
set(handles.listbox2,'String',str2);





% --- Executes on button press in rm.
function rm_Callback(hObject, eventdata, handles)
str1 = get(handles.listbox1,'String');
if ~iscell(str1) && ~isempty(str1), str1 = {str1};end
index = get(handles.listbox2,'Value');
str2 = get(handles.listbox2,'String');
if isempty(str2), return;end
if isempty(str1) 
    str1 = str2(index);
    str2(index) = [];
else
    if ~iscell(str2)
        str1{end+1,1} = str2;
        str2 = [];
    else
        str1{end+1,1} = str2{index};
        str2(index) = [];
    end
end

if length(str2) == 1, str2 = str2{1};end

set(handles.listbox1,'String',str1);
set(handles.listbox2,'Value',1);
set(handles.listbox2,'String',str2);


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle add listbox2 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item rm listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle add listbox2 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
uiresume;


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
set(handles.listbox2,'String',[]);
uiresume


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle add pushbutton5 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
