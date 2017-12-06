function varargout = streamBrowserNG(varargin)
% STREAMBROWSERNG MATLAB code for streamBrowserNG.fig
%      STREAMBROWSERNG, by itself, creates a new STREAMBROWSERNG or raises the existing
%      singleton*.
%
%      H = STREAMBROWSERNG returns the handle to a new STREAMBROWSERNG or the handle to
%      the existing singleton*.
%
%      STREAMBROWSERNG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STREAMBROWSERNG.M with the given input arguments.
%
%      STREAMBROWSERNG('Property','Value',...) creates a new STREAMBROWSERNG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before streamBrowserNG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to streamBrowserNG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help streamBrowserNG

% Last Modified by GUIDE v2.5 23-May-2012 22:23:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @streamBrowserNG_OpeningFcn, ...
                   'gui_OutputFcn',  @streamBrowserNG_OutputFcn, ...
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


% --- Executes just before streamBrowserNG is made visible.
function streamBrowserNG_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
browserObj = varargin{1};

try
    mobilab = browserObj.streamHandle.container.container;
    path = [mobilab.path filesep 'skin'];
    if ~exist(path,'dir')
        path = [fileparts(fileparts(which('streamBrowserNG.m'))) filesep 'skin'];
    end
catch 
    % takes MoBILAB default colors (I hard coded this because at the end eegplotNG will be absorbed)
    mobilab.preferences.gui.buttonColor = [1 1 1];
    mobilab.preferences.gui.backgroundColor = [0.93 0.96 1];
end
CData = imread([path filesep '32px-Dialog-apply.svg.png']);set(handles.connectLine,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-edit-clear.svg.png']);set(handles.deleteLine,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-media-seek-backward.svg.png']);set(handles.play_rev,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-media-seek-forward.svg.png']);set(handles.play_fwd,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-preferences-system.svg.png']); set(handles.settings,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-media-seek-backward.svg.png']);set(handles.previous,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-media-seek-forward.svg.png']);set(handles.next,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);
CData = imread([path filesep '32px-Gnome-media-playback-start.svg.png']);set(handles.play,'CData',CData,'BackgroundColor',mobilab.preferences.gui.buttonColor);

set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor);
set(handles.text4,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text5,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text6,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text10,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.slider3,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.text6,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel6,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ShadowColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel8,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ShadowColor',mobilab.preferences.gui.backgroundColor,...
    'ForegroundColor',mobilab.preferences.gui.backgroundColor,'HighlightColor',mobilab.preferences.gui.backgroundColor);

set(handles.play,'userData',{imread([path filesep '32px-Gnome-media-playback-start.svg.png']) imread([path filesep '32px-Gnome-media-playback-pause.svg.png'])});

browserObj.figureHandle = handles.figure1;
set(handles.axes1,'tag','axes1');

setappdata(handles.axes1,'LegendColorbarManualSpace',1);
setappdata(handles.axes1,'LegendColorbarReclaimSpace',1);
xlabel(handles.axes1,'');
ylabel(handles.axes1,'');
zlabel(handles.axes1,'');
browserObj.axesHandle = handles.axes1;
browserObj.timeTexttHandle = handles.text6;
browserObj.sliderHandle = handles.slider3;
try
    hListener = handle.listener(handles.slider3,'ActionEvent',@slider3_Callback);
catch
    hListener = addlistener(handles.slider3,'ContinuousValueChange',@slider3_Callback);
end
setappdata(browserObj.sliderHandle,'sliderListeners',hListener);

set( handles.listbox1,'Value',1);
set( handles.listbox1,'String',browserObj.streamHandle.event.uniqueLabel);

browserObj.createGraphicObjects(browserObj.nowCursor);
set(handles.figure1,'userData',browserObj);


% --- Outputs from this function are returned to the command line.
function varargout = streamBrowserNG_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
delete(browserObj);



% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
if isMultipleCall, return;  end
browserObj = get(get(hObject,'parent'),'userData');
newNowCursor = get(browserObj.sliderHandle,'Value');
browserObj.plotThisTimeStamp(newNowCursor);


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
if isa(browserObj.master,'browserHandleList')
    set( handles.listbox1,'Value',1);
    set( handles.listbox1,'String',browserObj.streamHandle.event.uniqueLabel);
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
delete(hObject);


% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
try close(get(browserObj.axesHandle,'userData'));end %#ok
set(browserObj.axesHandle,'userData',[]);

if isa(browserObj,'eegplotNGHandle') %isa(browserObj,'streamBrowserHandle')
    if strcmp(browserObj.zoomHandle.Enable,'off'), browserObj.editRoi;end
end


% --- Executes on button press in previous.
function previous_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
if isa(browserObj,'segmentedStreamBrowserHandle')
    eventObj = browserObj.eventObj;
else
    eventObj = browserObj.streamHandle.event;
end
ind = get(handles.listbox1,'Value');
if ~isempty(eventObj.label)
    [~,loc] = ismember( eventObj.label, eventObj.uniqueLabel{ind});
    tmp  = eventObj.latencyInFrame(logical(loc));
    tmp2 = browserObj.streamHandle.timeStamp(eventObj.latencyInFrame(logical(loc))) -  browserObj.nowCursor;
    tmp(tmp2>0) = [];
    tmp2(tmp2>=0) = [];
    [~,loc1] = max(tmp2);
    jumpLatency = tmp(loc1);
    if ~isempty(jumpLatency)
        set(browserObj.sliderHandle,'Value',browserObj.streamHandle.timeStamp(jumpLatency));
        slider3_Callback(browserObj.sliderHandle,[],handles); 
    end
end


% --- Executes on button press in next.
function next_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
if isa(browserObj,'segmentedStreamBrowserHandle')
    eventObj = browserObj.eventObj;
else
    eventObj = browserObj.streamHandle.event;
end

ind = get(handles.listbox1,'Value');
if ~isempty(eventObj.label)
    [~,loc] = ismember( eventObj.label, eventObj.uniqueLabel{ind});
    tmp  = eventObj.latencyInFrame(logical(loc));
    tmp2 = browserObj.streamHandle.timeStamp(eventObj.latencyInFrame(logical(loc))) -  browserObj.nowCursor;
    tmp(tmp2<=0) = [];
    tmp2(tmp2<=0) = [];
    [~,loc1] = min(tmp2);
    jumpLatency = tmp(loc1);
    if ~isempty(jumpLatency)
        set(browserObj.sliderHandle,'Value',browserObj.streamHandle.timeStamp(jumpLatency));
        slider3_Callback(browserObj.sliderHandle,[],handles); 
    end
end

% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in play_rev.
function play_rev_Callback(hObject, eventdata, handles)
try
    browserObj = get(handles.figure1,'userData');
    browserObj.plotStep(-browserObj.step);
catch ME
    ME.rethrow;
end

% --- Executes on button press in play.
function play_Callback(hObject, eventdata, handles)
try
    browserObj = get(handles.figure1,'userData');
    CData = get(handles.play,'UserData');
    if ~browserObj.state
        set(handles.play,'CData',CData{2});
    else
        set(handles.play,'CData',CData{1});
    end
    browserObj.play;
catch ME
    ME.rethrow;
end

% --- Executes on button press in play_fwd.
function play_fwd_Callback(hObject, eventdata, handles)
try
    browserObj = get(handles.figure1,'userData');
    browserObj.plotStep(browserObj.step);
catch ME
    ME.rethrow;
end

% --- Executes on button press in settings.
function settings_Callback(hObject, eventdata, handles)
try
    browserListObj = get(handles.figure1,'userData');
    browserListObj.changeSettings;
catch ME
    ME.rethrow;
end


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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


% --- Executes when uipanel8 is resized.
function uipanel8_ResizeFcn(hObject, eventdata, handles)


% --- Executes on button press in connectLine.
function connectLine_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
browserObj.connectMarkers;


% --- Executes on button press in deleteLine.
function deleteLine_Callback(hObject, eventdata, handles)
browserObj = get(handles.figure1,'userData');
browserObj.deleteConnection;


% --- Executes on scroll wheel click while the figure is in focus.
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	VerticalScrollCount: signed integer indicating direction and number of clicks
%	VerticalScrollAmount: number of lines scrolled for each click
% handles    structure with handles and user data (see GUIDATA)
% browserObj = get(handles.figure1,'userData');
% if isa(browserObj,'streamBrowserHandle') || isa(browserObj,'eegplotNGHandle')
%     if strcmp(browserObj.zoomHandle.enable,'on'), browserObj.zoom;end
% end
