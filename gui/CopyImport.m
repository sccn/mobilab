function varargout = CopyImport(varargin)
% COPYIMPORT MATLAB code for CopyImport.fig
%      COPYIMPORT, by itself, creates a new COPYIMPORT or raises the existing
%      singleton*.
%
%      H = COPYIMPORT returns the handle to a new COPYIMPORT or the handle to
%      the existing singleton*.
%
%      COPYIMPORT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in COPYIMPORT.M with the given input arguments.
%
%      COPYIMPORT('Property','Value',...) creates a new COPYIMPORT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CopyImport_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CopyImport_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CopyImport

% Last Modified by GUIDE v2.5 11-Apr-2013 20:30:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CopyImport_OpeningFcn, ...
                   'gui_OutputFcn',  @CopyImport_OutputFcn, ...
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


% --- Executes just before CopyImport is made visible.
function CopyImport_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
mobilab = varargin{1};

set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel1,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel3,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.pushbutton1,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.pushbutton3,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.pushbutton4,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.pushbutton5,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.figure1,'userData',[]);


% --- Outputs from this function are returned to the command line.
function varargout = CopyImport_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% OK
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)

source = get(handles.edit1,'String');
if isempty(source), return;end

destination = get(handles.edit3,'String');
if isempty(destination), return;end

userData = struct('source',source,'destination',destination);
set(handles.figure1,'userData',userData);
uiresume;


% Cancel
% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
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


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
folder = uigetdir2('Select the destination folder');
set(handles.edit3,'String',folder);


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
folder = uigetdir2('Select the folder to copy');
set(handles.edit1,'String',folder);


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(hObject);



% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
