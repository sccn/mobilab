function varargout = MultiStreamBrowserSettings(varargin)
% MULTISTREAMBROWSERSETTINGS MATLAB code for MultiStreamBrowserSettings.fig
%      MULTISTREAMBROWSERSETTINGS, by itself, creates a new MULTISTREAMBROWSERSETTINGS or raises the existing
%      singleton*.
%
%      H = MULTISTREAMBROWSERSETTINGS returns the handle to a new MULTISTREAMBROWSERSETTINGS or the handle to
%      the existing singleton*.
%
%      MULTISTREAMBROWSERSETTINGS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MULTISTREAMBROWSERSETTINGS.M with the given input arguments.
%
%      MULTISTREAMBROWSERSETTINGS('Property','Value',...) creates a new MULTISTREAMBROWSERSETTINGS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MultiStreamBrowserSettings_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MultiStreamBrowserSettings_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MultiStreamBrowserSettings

% Last Modified by GUIDE v2.5 14-Jun-2011 16:24:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MultiStreamBrowserSettings_OpeningFcn, ...
    'gui_OutputFcn',  @MultiStreamBrowserSettings_OutputFcn, ...
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


% --- Executes just before MultiStreamBrowserSettings is made visible.
function MultiStreamBrowserSettings_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
userData = varargin{1};
if exist(userData.session,'file')
    S = load(userData.session,'-mat');
    if isfield(S,'defaults')
        set(handles.edit1,'String',num2str(S.defaults.startTime));
        set(handles.edit2,'String',num2str(S.defaults.endTime));
        set(handles.edit3,'String',num2str(S.defaults.frame/S.defaults.samplingRate));
        set(handles.edit4,'String',num2str(S.defaults.step));
        set(handles.checkbox2,'Value',S.defaults.showNumberFlag);
        set(handles.figure1,'userData',userData);
    else
        set(handles.edit1,'String',num2str(userData.defaults.startTime));
        set(handles.edit2,'String',num2str(userData.defaults.endTime));
        set(handles.edit3,'String',num2str(userData.defaults.frame/userData.defaults.samplingRate));
        set(handles.edit4,'String',num2str(double(userData.defaults.step)));
        set(handles.checkbox2,'Value',userData.defaults.showNumberFlag);
        set(handles.figure1,'userData',userData);
    end
else
    set(handles.edit1,'String',num2str(userData.defaults.startTime));
    set(handles.edit2,'String',num2str(userData.defaults.endTime));
    set(handles.edit3,'String',num2str(userData.defaults.frame/userData.defaults.samplingRate));
    set(handles.edit4,'String',num2str(userData.defaults.step));
    set(handles.checkbox2,'Value',userData.defaults.showNumberFlag);
    set(handles.figure1,'userData',userData);
end


% --- Outputs from this function are returned to the command line.
function varargout = MultiStreamBrowserSettings_OutputFcn(hObject, eventdata, handles)
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



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



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



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in done.
function done_Callback(hObject, eventdata, handles)
userData = get(handles.figure1,'userData');
try
    userData = get(userData.parent,'userData');
    
    tmp = str2double(get(handles.edit1,'String'));
    if isempty(tmp), error('Start time must be a number');end
    if tmp > userData.allDataStreams.item{1}.timeStamp(end), error('Start time out of the recording time.');end
    userData.defaults.startTime = tmp;
    
    tmp = str2double(get(handles.edit2,'String'));
    if isempty(tmp), error('End time must be a number');end
    if tmp < userData.defaults.startTime, error('End time must be greater than stat time.');end
    userData.defaults.endTime = tmp;
    
    tmp = str2double(get(handles.edit3,'String'));
    if isempty(tmp), error('Frame must be a number');end
    if tmp < 0, error('Frame must be greater than 0.');end
    userData.defaults.frame = int32(tmp*userData.defaults.samplingRate);
    
    tmp = str2double(get(handles.edit4,'String'));
    if isempty(tmp), error('The speed must be a number');end
    if tmp > userData.defaults.frame, error('This speed requires a wider window.');end
    userData.defaults.step = int32(tmp);
    
    userData.defaults.showNumberFlag = get(handles.checkbox2,'Value');
    
    if ~isempty(userData.handler)
        for it=1:length(userData.addList)
            figure(userData.handler(it))
            userDataTmp = get(userData.handler(it),'userData');
            [t1,t2] = userDataTmp.obj.getTimeIndex([userData.defaults.startTime userData.defaults.endTime]);
            userDataTmp.timeIndex = t1:t2;
            userDataTmp.frame = userData.defaults.frame;
            userDataTmp.step  = userData.defaults.step;
            userDataTmp.rate  = userData.defaults.rate;
            userDataTmp.now   = int32(userDataTmp.frame/2);
            userDataTmp.showNumberFlag = userData.defaults.showNumberFlag;
            
            switch userDataTmp.class
                case 'dataStream'
                    userDataTmp.t1    = 1;
                    userDataTmp.t2    = userDataTmp.t1 + userDataTmp.frame;
                    userDataTmp.N = length(userDataTmp.timeIndex);
                    userDataTmp.YTickLabel = cell(userDataTmp.Nch,1);
                    labels = cell(userDataTmp.Nch,1);
                    if userDataTmp.showNumberFlag
                        if userDataTmp.Nch > 1
                            for jt=fliplr(1:userDataTmp.Nch), labels{jt} = num2str(userDataTmp.channels(userDataTmp.Nch-jt+1));end
                        else
                            labels{1} = num2str(userDataTmp.channels);
                        end
                    else
                        if userDataTmp.Nch > 1
                            for jt=fliplr(1:userDataTmp.Nch), labels{jt} = userDataTmp.obj.label{userDataTmp.channels(userDataTmp.Nch-jt+1)};end
                        else
                            if ~isempty(userDataTmp.obj.label)
                                labels{1} = userDataTmp.obj.label{userDataTmp.channels};
                            else
                                labels{1} = '';
                            end
                        end
                    end
                    userDataTmp.YTickLabel = labels;
                case 'mocap'
                    if length(userDataTmp.markers) == 1
                        userDataTmp.label = {userDataTmp.markers};
                    else
                        tmp = num2cell(userDataTmp.markers,[1,length(userDataTmp.markers)]);
                        for jt=1:length(userDataTmp.markers), userDataTmp.label{jt} = tmp{jt};end
                    end
                otherwise
                    userDataTmp.t1    = 1;
                    userDataTmp.t2    = userDataTmp.t1 + userDataTmp.frame;
                    userDataTmp.N = length(userDataTmp.timeIndex);
                    userDataTmp.YTickLabel = cell(userDataTmp.Nch,1);
                    labels = cell(userDataTmp.Nch,1);
                    if userDataTmp.showNumberFlag
                        for jt=fliplr(1:userDataTmp.Nch), labels{jt} = num2str(userDataTmp.channels(userDataTmp.Nch-jt+1));end
                    else
                        for jt=fliplr(1:userDataTmp.Nch), labels{jt} = userDataTmp.obj.label{userDataTmp.channels(userDataTmp.Nch-jt+1)};end
                    end
                    userDataTmp.YTickLabel = labels;
            end
            if it==1
                N = length(userDataTmp.timeIndex);
                slider1 = findobj(userData.parent,'tag','slider1');
                set(slider1,'Max',N);
                set(slider1,'Min',1);
                set(slider1,'Value',userDataTmp.now);
                set(slider1,'SliderStep',(25/N)*ones(1,2));
                set(findobj(userData.parent,'tag','text6'),'String',num2str(userData.defaults.startTime));
                set(findobj(userData.parent,'tag','text7'),'String',num2str(userData.defaults.endTime));
                slider3 = findobj(userData.handler,'tag','slider3');
                set(slider3,'Max',N);
                set(slider3,'Min',1);
                set(slider3,'Value',userDataTmp.now);
                set(slider3,'SliderStep',(25/N)*ones(1,2));
                set(findobj(userData.handler,'tag','text4'),'String',num2str(userData.defaults.startTime));
                set(findobj(userData.handler,'tag','text5'),'String',num2str(userData.defaults.endTime));
            end
            switch userDataTmp.class
                case 'dataStream'
                    plotFrame(userDataTmp,'fwd');
                case 'mocap'
                    plotMocapFrame(userDataTmp,'fwd');
                otherwise
                    plotFrame(userDataTmp,'fwd');
            end
        end
    else
        [t1,t2] = userData.allDataStreams.item{1}.getTimeIndex([userData.defaults.startTime userData.defaults.endTime]);
        N = length(userData.allDataStreams.item{1}.timeStamp(t1:t2));
        slider1 = findobj(userData.parent,'tag','slider1');
        set(slider1,'Max',N);
        set(slider1,'Min',1);
        set(slider1,'Value',1);
        set(slider1,'SliderStep',(25/N)*ones(1,2));
        set(findobj(userData.parent,'tag','text6'),'String',num2str(userData.defaults.startTime));
        set(findobj(userData.parent,'tag','text7'),'String',num2str(userData.defaults.endTime));
    end
    defaults = userData.defaults;
    dataSourceLocation = userData.allDataStreams.dataSourceLocation;
    addList = userData.addList;
    tableHandler = findobj(userData.parent,'tag','uitable1');
    Data = get(tableHandler,'Data');
    save(userData.session,'defaults','dataSourceLocation','addList','Data','-mat');
    set(userData.parent,'userData',userData);
    close(handles.figure1)
catch ME
    errordlg2(ME.message);
end


% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
userData = get(handles.figure1,'userData');
userData = get(userData.parent,'userData');

userData.defaults.startTime    = userData.allDataStreams.item{1}.timeStamp(1);
userData.defaults.endTime      = userData.allDataStreams.item{1}.timeStamp(end);
userData.defaults.samplingRate = userData.allDataStreams.item{1}.samplingRate;
userData.defaults.frame        = int32(userData.defaults.samplingRate*5); % display 5 sec
userData.defaults.step         = int32(userData.defaults.frame/40);       % update the 40% of the frame each plot
userData.defaults.showNumberFlag = 0;

if ~isempty(userData.handler)
    for it=1:length(userData.addList)
        figure(userData.handler(it))
        userDataTmp = get(userData.handler(it),'userData');
        [t1,t2] = userDataTmp.obj.getTimeIndex([userData.defaults.startTime userData.defaults.endTime]);
        userDataTmp.timeIndex = t1:t2;
        userDataTmp.frame = userData.defaults.frame;
        userDataTmp.step  = userData.defaults.step;
        userDataTmp.now   = int32(userDataTmp.frame/2);
        userDataTmp.showNumberFlag = userData.defaults.showNumberFlag;
        
        switch userDataTmp.class
            case 'dataStream'
                userDataTmp.t1    = userDataTmp.timeIndex(1);
                userDataTmp.t2    = userDataTmp.t1 + userDataTmp.frame;
                userDataTmp.N = length(userDataTmp.timeIndex);
                userDataTmp.YTickLabel = cell(userDataTmp.Nch,1);
                for jt=1:userDataTmp.Nch, userDataTmp.YTickLabel{jt} = userDataTmp.obj.label{userDataTmp.channels(userDataTmp.Nch-jt+1)};end
            case 'mocap'
                if length(userDataTmp.markers) == 1
                    userDataTmp.label = {userDataTmp.markers};
                else
                    tmp = num2cell(userDataTmp.markers,[1,length(userDataTmp.markers)]);
                    for jt=1:length(userDataTmp.markers), userDataTmp.label{jt} = tmp{jt};end
                end
            otherwise
                userDataTmp.t1    = userDataTmp.timeIndex(1);
                userDataTmp.t2    = userDataTmp.t1 + userDataTmp.frame;
                userDataTmp.N = length(userDataTmp.timeIndex);
                userDataTmp.YTickLabel = cell(userDataTmp.Nch,1);
                for jt=1:userDataTmp.Nch, userDataTmp.YTickLabel{jt} = userDataTmp.obj.label{userDataTmp.channels(userDataTmp.Nch-jt+1)};end
        end
        if it==1
            N = length(userDataTmp.timeIndex);
            slider1 = findobj(userData.parent,'tag','slider1');
            set(slider1,'Max',userDataTmp.timeIndex(end));
            set(slider1,'Min',userDataTmp.timeIndex(1));
            set(slider1,'Value',userDataTmp.now);
            set(slider1,'SliderStep',(25/N)*ones(1,2));
            set(findobj(userData.parent,'tag','text6'),'String',num2str(userData.defaults.startTime));
            set(findobj(userData.parent,'tag','text7'),'String',num2str(userData.defaults.endTime));
            slider3 = findobj(userData.handler,'tag','slider3');
            set(slider3,'Max',userDataTmp.timeIndex(end));
            set(slider3,'Min',userDataTmp.timeIndex(1));
            set(slider3,'Value',userDataTmp.now);
            set(slider3,'SliderStep',(25/N)*ones(1,2));
            set(findobj(userData.handler,'tag','text4'),'String',num2str(userData.defaults.startTime));
            set(findobj(userData.handler,'tag','text5'),'String',num2str(userData.defaults.endTime));
        end
        switch userDataTmp.class
            case 'dataStream'
                plotFrame(userDataTmp,'fwd');
            case 'mocap'
                plotMocapFrame(userDataTmp,'fwd');
            otherwise
                plotFrame(userDataTmp,'fwd');
        end
    end
end
defaults = userData.defaults;
dataSourceLocation = userData.allDataStreams.dataSourceLocation;
addList = userData.addList;
tableHandler = findobj(userData.parent,'tag','uitable1');
Data = get(tableHandler,'Data');
save(userData.session,'defaults','dataSourceLocation','addList','Data','-mat');

set(handles.edit1,'String',num2str(userData.defaults.startTime));
set(handles.edit2,'String',num2str(userData.defaults.endTime));
set(handles.edit3,'String',num2str(userData.defaults.frame/userData.defaults.samplingRate));
set(handles.edit4,'String',num2str(userData.defaults.step));
set(handles.checkbox2,'Value',userData.defaults.showNumberFlag);
set(handles.figure1,'userData',userData);
set(userData.parent,'userData',userData);
close(handles.figure1)


% --- Executes on button press in help.
function help_Callback(hObject, eventdata, handles)
web http://sccn.ucsd.edu/wiki/Mobilab_software/MoBILAB_toolbox_tutorial#The_MultiStream_Browser



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
