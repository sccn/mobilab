function varargout = StreamsList(varargin)
% STREAMSLIST MATLAB code for StreamsList.fig
%      STREAMSLIST, by itself, creates a new STREAMSLIST or raises the existing
%      singleton*.
%
%      H = STREAMSLIST returns the handle to a new STREAMSLIST or the handle to
%      the existing singleton*.
%
%      STREAMSLIST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STREAMSLIST.M with the given input arguments.
%
%      STREAMSLIST('Property','Value',...) creates a new STREAMSLIST or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before StreamsList_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to StreamsList_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help StreamsList

% Last Modified by GUIDE v2.5 04-Nov-2011 13:03:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @StreamsList_OpeningFcn, ...
                   'gui_OutputFcn',  @StreamsList_OutputFcn, ...
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


% --- Executes just before StreamsList is made visible.
function StreamsList_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
guidata(hObject, handles);
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep '32px-Gnome-preferences-system.svg.png']); set(handles.settings,'CData',CData);

if isempty(varargin), return;end
browserListObj = varargin{1};
set(handles.figure1,'Color',get(browserListObj.master,'Color'))

N = length(browserListObj.list);
if N < 1, return;end
set(handles.figure1,'userData',browserListObj.master);
position = get(handles.figure1,'position');    set(handles.figure1,'Position',[position(1:3) position(4)+2*(N-1)]);
position = get(handles.edit1,'position');      set(handles.edit1,'Position',[position(1) position(2)+2*(N-1) position(3:4)]);
position = get(handles.settings,'position');   set(handles.settings,'Position',[position(1) position(2)+2*(N-1) position(3:4)]);
position = get(handles.popupmenu1,'position'); set(handles.popupmenu1,'Position',[position(1) position(2)+2*(N-1) position(3:4)]);

for it=1:N
    Value = 1;
    switch class(browserListObj.list{it}.streamHandle)
        case 'eeg'
            strPopup = {'DataStreamBrowser'};
        case 'icaStream'
            strPopup = {'DataStreamBrowser'};
        case {'dataStream','wii'}
            strPopup = {'DataStreamBrowser'};
        case 'mocap'
            strPopup = {'mocapBrowserHandle' 'DataStreamBrowser' 'generalizedCoordinatesBrowserHandle','cometBrowserHandle'};%'phaseSpaceBrowser'
        case 'videoStream'
            strPopup = {'videoStreamBrowserHandle'};
        case 'audioStream'
            strPopup = {'audioStreamBrowserHandle'};
        case 'wii'
            strPopup = {'mocapBrowserHandle' 'DataStreamBrowser'};
        case 'projectedMocap'
            strPopup = {'projectionBrowserHandle' 'vectorBrowserHandle'};
        case 'coreTimeFrequencyObject'
            strPopup = 'spectrogramBrowserHandle';
        case 'pcdStream'
             strPopup = {'pcdBrowserHandle','plotROI'};
        otherwise
            strPopup = 'DataStreamBrowser';
    end
    Value = find(strcmp(strPopup,class(browserListObj.list{it})));
    if isempty(Value), Value = 1;end
    if it==1
        He = findobj(handles.figure1,'tag','edit1');
        set(He,'String',browserListObj.list{it}.streamHandle.name)
        Hp = findobj(handles.figure1,'tag','popupmenu1');
        set(Hp,'String',strPopup,'Value',Value,'userData',{it browserListObj.list{it}});
        Hb = findobj(handles.figure1,'tag','settings');
        set(Hb,'CData',CData,'callback',['browserListObj = get(get(gcf,''userData''),''userData'');try browserListObj.list{' num2str(it) '}.changeSettings;end']);
    else
        position = get(He,'position');
        HeTmp = copyobj(He,handles.figure1);
        set(HeTmp,'String',browserListObj.list{it}.streamHandle.name,'Position',[position(1) position(2)-2*(it-1) position(3:4)],'tag','edit1Tmp');
        
        position = get(Hp,'position');
        HpTmp = copyobj(Hp,handles.figure1);
        set(HpTmp,'String',strPopup,'Value',Value,'userData',{it browserListObj.list{it}},...
            'Position',[position(1) position(2)-2*(it-1) position(3:4)],'tag','popupmenu1Tmp');
        
        position = get(Hb,'position');
        HbTmp = copyobj(Hb,handles.figure1);
        set(HbTmp,'CData',CData,'callback',['browserListObj = get(get(gcf,''userData''),''userData'');try browserListObj.list{' num2str(it) '}.changeSettings;end'],...
            'Position',[position(1) position(2)-2*(it-1) position(3:4)],'tag','settingsTmp');   
    end
end


% --- Outputs from this function are returned to the command line.
function varargout = StreamsList_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popupmenu1 as text
%        str2double(get(hObject,'String')) returns contents of popupmenu1 as a double


% --- Executes during object creation, after setting all properties.
function settings_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
index = get(hObject,'Value');
browserType = get(hObject,'String');
browserType = char(browserType(index));

userData = get(hObject,'userData');
if ~isvalid(userData{2}), return;end
listIndex = userData{1};

browserListObj = get(get(handles.figure1,'userData'),'userData');
streamHandle = browserListObj.list{listIndex}.streamHandle;
browserListObj.list{listIndex}.delete;
browserListObj.addHandle(streamHandle,browserType);


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


% --- Executes on button press in popupmenu1.
function settings_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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
