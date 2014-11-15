function varargout = ImportFolder2EEGLAB(varargin)
% IMPORTFOLDER2EEGLAB MATLAB code for ImportFolder2EEGLAB.fig
%      IMPORTFOLDER2EEGLAB, by itself, creates a new IMPORTFOLDER2EEGLAB or raises the existing
%      singleton*.
%
%      H = IMPORTFOLDER2EEGLAB returns the handle to a new IMPORTFOLDER2EEGLAB or the handle to
%      the existing singleton*.
%
%      IMPORTFOLDER2EEGLAB('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMPORTFOLDER2EEGLAB.M with the given input arguments.
%
%      IMPORTFOLDER2EEGLAB('Property','Value',...) creates a new IMPORTFOLDER2EEGLAB or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImportFolder2EEGLAB_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImportFolder2EEGLAB_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImportFolder2EEGLAB

% Last Modified by GUIDE v2.5 31-Jul-2012 12:14:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImportFolder2EEGLAB_OpeningFcn, ...
                   'gui_OutputFcn',  @ImportFolder2EEGLAB_OutputFcn, ...
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


% --- Executes just before ImportFolder2EEGLAB is made visible.
function ImportFolder2EEGLAB_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImportFolder2EEGLAB (see VARARGIN)
userData.source = [];
userData.mobiDataDirectory= [];
userData.streamListFlag = false;
set(handles.figure1,'UserData',userData);
% set(handles.edit1,'string',source);
% set(handles.edit4,'string',rootDir);
% set(handles.edit3,'string',dataSourceName);
handles.output = hObject;
guidata(hObject, handles);

% UIWAIT makes ImportFolder2EEGLAB wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ImportFolder2EEGLAB_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% OK
function pushbutton1_Callback(hObject, eventdata, handles)
source = get(handles.edit1,'string');
rootDir        = get(handles.edit4,'string');
dataSourceName = get(handles.edit3,'string');
ind = strfind(dataSourceName,'_MoBI');
if ~isempty(ind), 
    dataSourceName(ind:ind+4) = [];
    set(handles.edit3,'string',dataSourceName);
end

mobiDataDirectory = fullfile(rootDir,[dataSourceName '_MoBI']);
if ~exist([rootDir filesep dataSourceName '_MoBI'],'dir')
    try
        mkdir(mobiDataDirectory);
    catch ME
        mobiDataDirectory = [];
        uiresume;
        errordlg(ME.message);
    end
elseif ~isempty(dir(mobiDataDirectory))
    choice = questdlg2(...
        sprintf('This folder contains a dataSource.\nAll the previous data will be removed.\n Would you like to continue?'),...
        'Warning!!!','Yes','No','No');
    switch choice
        case 'Yes'
            warning off %#ok
            try rmdir(mobiDataDirectory,'s');end %#ok
            try mkdir(mobiDataDirectory);end     %#ok
            warning on %#ok
        case 'No'
            mobiDataDirectory = [];
            return
        otherwise
            mobiDataDirectory = [];
            return
    end
end
userData.source = source;
userData.mobiDataDirectory = mobiDataDirectory;
userData.streamListFlag = get(handles.checkbox4,'Value');
set(handles.figure1,'UserData',userData);
uiresume
% close(handles.figure1);


% Cancel
function pushbutton2_Callback(hObject, eventdata, handles)
set(handles.figure1,'userData',[]);
uiresume


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1



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


% Enter the DataRiver (.drf) file:
function pushbutton3_Callback(hObject, eventdata, handles)
source = uigetdir2('Select folder');
if isnumeric(source), return;end
[pathName,name,ext] = fileparts(source);
set(handles.edit1,'string',source);
set(handles.edit4,'string',pathName);
set(handles.edit3,'string',name);


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


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


% Provide the directory where save the MoBI dataSource:
function pushbutton6_Callback(hObject, eventdata, handles)
rootDir = uigetdir2('Select the root directory for the memory-mapped dataSource files');
if isnumeric(rootDir), return;end
if isempty(rootDir), return;end
set(handles.edit4,'string',rootDir);



% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


% Help
function pushbutton7_Callback(hObject, eventdata, handles)
web http://sccn.ucsd.edu/wiki/Mobilab_software/MoBILAB_toolbox_tutorial


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
set(handles.figure1,'userData',[]);
uiresume


% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4
