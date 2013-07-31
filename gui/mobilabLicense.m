function varargout = mobilabLicense(varargin)
% MOBILABLICENSE MATLAB code for mobilabLicense.fig
%      MOBILABLICENSE, by itself, creates a new MOBILABLICENSE or raises the existing
%      singleton*.
%
%      H = MOBILABLICENSE returns the handle to a new MOBILABLICENSE or the handle to
%      the existing singleton*.
%
%      MOBILABLICENSE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOBILABLICENSE.M with the given input arguments.
%
%      MOBILABLICENSE('Property','Value',...) creates a new MOBILABLICENSE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before mobilabLicense_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to mobilabLicense_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help mobilabLicense

% Last Modified by GUIDE v2.5 09-Jun-2012 23:47:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @mobilabLicense_OpeningFcn, ...
                   'gui_OutputFcn',  @mobilabLicense_OutputFcn, ...
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


% --- Executes just before mobilabLicense is made visible.
function mobilabLicense_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
mobilab = varargin{1};
str = fileread([mobilab.path filesep 'license.txt']);
set(handles.text2,'String',str);



% --- Outputs from this function are returned to the command line.
function varargout = mobilabLicense_OutputFcn(hObject, eventdata, handles) 
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
