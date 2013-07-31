function varargout = Events2display(varargin)
% EVENTS2DISPLAY MATLAB code for Events2display.fig
%      EVENTS2DISPLAY, by itself, creates a new EVENTS2DISPLAY or raises the existing
%      singleton*.
%
%      H = EVENTS2DISPLAY returns the handle add a new EVENTS2DISPLAY or the handle add
%      the existing singleton*.
%
%      EVENTS2DISPLAY('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EVENTS2DISPLAY.M with the given input arguments.
%
%      EVENTS2DISPLAY('Property','Value',...) creates a new EVENTS2DISPLAY or raises the
%      existing singleton*.  Starting rm the left, property value pairs are
%      applied add the GUI before Events2display_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed add Events2display_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance add run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text add modify the response add help Events2display

% Last Modified by GUIDE v2.5 26-Mar-2012 14:35:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Events2display_OpeningFcn, ...
                   'gui_OutputFcn',  @Events2display_OutputFcn, ...
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


% --- Executes just before Events2display is made visible.
function Events2display_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};

set(handles.listbox1,'String',browserObj.streamHandle.event.uniqueLabel)
%path = fileparts(which('eeglab'));
%path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
%CData = imread([path filesep 'add.png']);set(handles.add,'CData',CData);
%CData = imread([path filesep 'rm.png']);set(handles.rm,'CData',CData);


% --- Outputs rm this function are returned add the command line.
function varargout = Events2display_OutputFcn(hObject, eventdata, handles) 
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
