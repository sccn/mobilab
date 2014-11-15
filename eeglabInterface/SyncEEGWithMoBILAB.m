function varargout = SyncEEGWithMoBILAB(varargin)
% SYNCEEGWITHMOBILAB MATLAB code for SyncEEGWithMoBILAB.fig
%      SYNCEEGWITHMOBILAB, by itself, creates a new SYNCEEGWITHMOBILAB or raises the existing
%      singleton*.
%
%      H = SYNCEEGWITHMOBILAB returns the handle to a new SYNCEEGWITHMOBILAB or the handle to
%      the existing singleton*.
%
%      SYNCEEGWITHMOBILAB('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SYNCEEGWITHMOBILAB.M with the given input arguments.
%
%      SYNCEEGWITHMOBILAB('Property','Value',...) creates a new SYNCEEGWITHMOBILAB or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SyncEEGWithMoBILAB_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SyncEEGWithMoBILAB_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SyncEEGWithMoBILAB

% Last Modified by GUIDE v2.5 08-Jun-2012 20:25:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SyncEEGWithMoBILAB_OpeningFcn, ...
                   'gui_OutputFcn',  @SyncEEGWithMoBILAB_OutputFcn, ...
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


% --- Executes just before SyncEEGWithMoBILAB is made visible.
function SyncEEGWithMoBILAB_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

mobilab = varargin{1};
handles.mobilab = mobilab;
guidata(hObject,handles);

set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel1,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel2,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.pushbutton1,'BackgroundColor',mobilab.preferences.gui.buttonColor);
set(handles.pushbutton3,'BackgroundColor',mobilab.preferences.gui.buttonColor);
set(handles.pushbutton4,'BackgroundColor',mobilab.preferences.gui.buttonColor);

N = length(mobilab.allStreams.item);
userData.name = [];
userData.uuid = [];
for it=1:N
    if isa(mobilab.allStreams.item{it},'dataStream') && ~ isa(mobilab.allStreams.item{it},'mocap')
        userData.name{end+1} = mobilab.allStreams.item{it}.name;
        userData.uuid{end+1} = mobilab.allStreams.item{it}.uuid;
    end
end
set(handles.popupmenu1,'userData',userData);
set(handles.popupmenu1,'String',userData.name);


if isempty(mobilab.allStreams), errordlg('Load some data first.');end



% --- Outputs from this function are returned to the command line.
function varargout = SyncEEGWithMoBILAB_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
setFile = get(handles.edit1,'string');
if ~exist(setFile,'file'), errordlg('The .set file doesn''t exist.');return;end

userData = get(handles.popupmenu1,'userData');
uuid = userData.uuid{get(handles.popupmenu1,'Value')};

if ~isa(uuid,'java.util.UUID'), errordlg('Cannot find the parent stream in MoBILAB''s tree.');return;end

mobilab = handles.mobilab;

itemIndex = mobilab.allStreams.findItem(uuid);
if isempty(itemIndex), errordlg('Cannot find the parent stream in MoBILAB''s tree.');return;end

clear userData
userData.file = setFile;
userData.parentIndex = itemIndex;
set(handles.figure1,'userData',userData);
uiresume


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
try close(get(handles.figure1,'userData'));end %#ok
set(handles.figure1,'userData',[]);
uiresume;


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


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
mobilab = handles.mobilab;
treeHandle = mobilab.gui('selectStream_Callback');
set(handles.figure1,'userData',treeHandle);



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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
[FileName,PathName] = uigetfile2({'*.set','EEGLAB file'},'Select the .set file');
if any([isnumeric(FileName) isnumeric(PathName)]), return;end
filename = fullfile(PathName,FileName);
set(handles.edit1,'string',filename);



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
