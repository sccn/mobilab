function varargout = ImportFromDatariverBDF(varargin)
% IMPORTFROMDATARIVERBDF MATLAB code for ImportFromDatariverBDF.fig
%      IMPORTFROMDATARIVERBDF, by itself, creates a new IMPORTFROMDATARIVERBDF or raises the existing
%      singleton*.
%
%      H = IMPORTFROMDATARIVERBDF returns the handle to a new IMPORTFROMDATARIVERBDF or the handle to
%      the existing singleton*.
%
%      IMPORTFROMDATARIVERBDF('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMPORTFROMDATARIVERBDF.M with the given input arguments.
%
%      IMPORTFROMDATARIVERBDF('Property','Value',...) creates a new IMPORTFROMDATARIVERBDF or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImportFromDatariverBDF_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImportFromDatariverBDF_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImportFromDatariverBDF

% Last Modified by GUIDE v2.5 22-Jun-2011 18:48:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImportFromDatariverBDF_OpeningFcn, ...
                   'gui_OutputFcn',  @ImportFromDatariverBDF_OutputFcn, ...
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


% --- Executes just before ImportFromDatariverBDF is made visible.
function ImportFromDatariverBDF_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImportFromDatariverBDF (see VARARGIN)
handles.output = hObject;
guidata(hObject, handles);

userData.source = [];
userData.rootDir = [];
userData.dataSourceName = [];
userData.configList = [];
set(handles.edit1,'string',userData.source);
set(handles.edit4,'string',userData.rootDir);
set(handles.edit3,'string',userData.dataSourceName);
set(handles.figure1,'userData',userData);
if isempty(varargin), return;end
mobilab = varargin{1};
set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel1,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel2,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel3,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text2,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text3,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text6,'BackgroundColor',mobilab.preferences.gui.backgroundColor);

% UIWAIT makes ImportFromDatariverBDF wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ImportFromDatariverBDF_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% OK
function pushbutton1_Callback(hObject, eventdata, handles)
userData.source = get(handles.edit1,'string');
userData.rootDir        = get(handles.edit4,'string');
userData.dataSourceName = get(handles.edit3,'string');
userData.configList     = get(handles.uitable1,'Data');
ind = strfind(userData.dataSourceName,'_MoBI');
if ~isempty(ind), userData.dataSourceName(ind:ind+4) = [];set(handles.edit3,'string',userData.dataSourceName);end
userData.mobiDataDirectory = [userData.rootDir filesep userData.dataSourceName '_MoBI'];
if exist(userData.mobiDataDirectory,'dir') && ~isempty(dir(userData.mobiDataDirectory)) 
    choice = questdlg2(sprintf('The folder is not empty.\nAll the previous files will be zipped.\n Would you like to continue?'),...
        'Warning!!!','Yes','No','No');
    if strcmp(choice,'No')
        set(handles.figure1,'UserData',[]);
        uiresume
        return
    end
end
set(handles.figure1,'userData',userData);
uiresume;


% Cancel
function pushbutton2_Callback(hObject, eventdata, handles)
userData = []; 
set(handles.figure1,'userData',userData);
uiresume;


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


% Enter the DataRiver (.bdf) file:
function pushbutton3_Callback(hObject, eventdata, handles)

[FileName,PathName] = uigetfile2({'*.bdf','Datariver File (.bdf)'},'Select the Datariver source file');
if any([isnumeric(FileName) isnumeric(PathName)]),return;end
source = fullfile(PathName,FileName);
[~,filename,ext] = fileparts(source);
set(handles.edit1,'string',source);
set(handles.edit3,'string',filename);




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
userData = get(handles.figure1,'userData');
if exist(userData.rootDir,'dir')
    userData.rootDir = uigetdir(userData.rootDir,'Select the rirectory where save save the MoBI dataSource');
else
    userData.rootDir = uigetdir(pwd,'Select the directory where save the MoBI dataSource');
end
if isempty(userData.source),
    userData.source = get(handles.edit1,'string');
    [~,filename,ext] = fileparts(userData.source);
    set(handles.edit3,'string',[filename ext]);
else
    set(handles.edit4,'string',userData.rootDir);
end
[~,filename,ext] = fileparts(userData.source);
set(handles.edit4,'string',userData.rootDir);
set(handles.edit3,'string',filename);
set(handles.figure1,'userData',userData);


% Help
function pushbutton7_Callback(hObject, eventdata, handles)
web http://sccn.ucsd.edu/wiki/Mobilab_software/MoBILAB_toolbox_tutorial


% --- Executes when selected cell(s) is changed in uitable1.
function uitable1_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to uitable1 (see GCBO)
% eventdata  structure with the following fields (see UITABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
uiresume
