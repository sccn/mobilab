function varargout = streamBrowser2(varargin)
% STREAMBROWSER2 MATLAB code for streamBrowser2.fig
%      STREAMBROWSER2, by itself, creates a new STREAMBROWSER2 or raises the existing
%      singleton*.
%
%      H = STREAMBROWSER2 returns the handle to a new STREAMBROWSER2 or the handle to
%      the existing singleton*.
%
%      STREAMBROWSER2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STREAMBROWSER2.M with the given input arguments.
%
%      STREAMBROWSER2('Property','Value',...) creates a new STREAMBROWSER2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before streamBrowser2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to streamBrowser2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help streamBrowser2

% Last Modified by GUIDE v2.5 28-Oct-2011 16:28:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @streamBrowser2_OpeningFcn, ...
                   'gui_OutputFcn',  @streamBrowser2_OutputFcn, ...
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


% --- Executes just before streamBrowser2 is made visible.
function streamBrowser2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to streamBrowser2 (see VARARGIN)
% Choose default command line output for streamBrowser2
handles.output = hObject;
% Update handles structure
guidata(hObject, handles);
browserObj = varargin{1};
browserObj.figureHandle = handles.figure1;
set(handles.axes1,'tag','axes1');

setappdata(handles.axes1,'LegendColorbarManualSpace',1);
setappdata(handles.axes1,'LegendColorbarReclaimSpace',1);

browserObj.axesHandle = handles.axes1;
browserObj.axesHandle2 = handles.axes2;
browserObj.timeTexttHandle = handles.text6;
browserObj.sliderHandle = handles.slider3;
browserObj.dcmHandle = datacursormode(handles.figure1);
browserObj.dcmHandle.SnapToDataVertex = 'on';
set(browserObj.dcmHandle,'UpdateFcn',@datatip2event);

browserObj.createGraphicObjects(browserObj.nowCursor);
set(handles.figure1,'userData',browserObj);


% --- Outputs from this function are returned to the command line.
function varargout = streamBrowser2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

browserObj = get(handles.figure1,'userData');
delete(browserObj);



% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
if strcmp(class(browserObj.master),'browserHandleList')
    set(hObject,'Value',browserObj.nowCursor);
else
    newNowCursor = get(hObject,'Value');
    browserObj.plotThisTimeStamp(newNowCursor);
end


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
if strcmp(class(browserObj.master),'browserHandleList')
    set( findobj(browserObj.master.master,'tag','edit8'),'String',browserObj.streamHandle.name);
    set( findobj(browserObj.master.master,'tag','listbox1'),'Value',1);
    set( findobj(browserObj.master.master,'tag','listbox1'),'String',browserObj.streamHandle.event.uniqueLabel);
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
delete(hObject);
