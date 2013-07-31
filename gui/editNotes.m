function varargout = editNotes(varargin)
% EDITNOTES MATLAB code for editNotes.fig
%      EDITNOTES, by itself, creates a new EDITNOTES or raises the existing
%      singleton*.
%
%      H = EDITNOTES returns the handle to a new EDITNOTES or the handle to
%      the existing singleton*.
%
%      EDITNOTES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EDITNOTES.M with the given input arguments.
%
%      EDITNOTES('Property','Value',...) creates a new EDITNOTES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before editNotes_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to editNotes_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help editNotes

% Last Modified by GUIDE v2.5 27-Jan-2013 02:51:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @editNotes_OpeningFcn, ...
                   'gui_OutputFcn',  @editNotes_OutputFcn, ...
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


% --- Executes just before editNotes is made visible.
function editNotes_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to editNotes (see VARARGIN)

% Choose default command line output for editNotes
handles.output = hObject;
if length(varargin) < 1, error('Not enough input arguments');end
if ~isa(varargin{1},'mobiAnnotator'), error(['Undefined function  ''editNotes'' for input agguments type ''' class(varargin{1}) '''.']);end
handles.notesObj = varargin{1};
guidata(hObject, handles);

if isa(handles.notesObj.container,'coreStreamObject')
    name = handles.notesObj.container.name;
    name = [': ' name];
elseif isa(handles.notesObj.container,'dataSource')
    [~,name] = fileparts(handles.notesObj.container.mobiDataDirectory);
    name = [': ' name];
else
    name = '';
end
name = ['MoBILAB annotator' name];

try %#ok
    mobilab = evalin('base','mobilab');
    
    set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor,'Name',name);
    set(handles.pushbutton1,'BackgroundColor',mobilab.preferences.gui.buttonColor);
    set(handles.pushbutton2,'BackgroundColor',mobilab.preferences.gui.buttonColor);
end
tx = handles.notesObj.text(:);
if ~strcmp(tx{end},char(13)), tx{end+1} = char(13);end
set(handles.edit1,'String',tx);


% --- Outputs from this function are returned to the command line.
function varargout = editNotes_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
notesObj = handles.notesObj;
notesObj.text = get(handles.edit1,'String');
if ~strcmp(notesObj.text{end},char(13)), notesObj.text{end+1} = char(13);end
close(handles.figure1);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
close(handles.figure1);
