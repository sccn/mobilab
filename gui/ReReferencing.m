function varargout = ReReferencing(varargin)
% REREFERENCING MATLAB code for ReReferencing.fig
%      REREFERENCING, by itself, creates a new REREFERENCING or raises the existing
%      singleton*.
%
%      H = REREFERENCING returns the handle add a new REREFERENCING or the handle add
%      the existing singleton*.
%
%      REREFERENCING('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REREFERENCING.M with the given input arguments.
%
%      REREFERENCING('Property','Value',...) creates a new REREFERENCING or raises the
%      existing singleton*.  Starting rm1 the left, property value pairs are
%      applied add the GUI before ReReferencing_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed add ReReferencing_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance add run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text add modify the response add help ReReferencing

% Last Modified by GUIDE v2.5 26-Jun-2012 23:39:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ReReferencing_OpeningFcn, ...
                   'gui_OutputFcn',  @ReReferencing_OutputFcn, ...
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


% --- Executes just before ReReferencing is made visible.
function ReReferencing_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
obj = varargin{1};
handles.obj = obj;
guidata(hObject, handles);
set(handles.panel_all,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.panel_ch,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
%set(handles.lb_all,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
%set(handles.lb_ch,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
%set(handles.lb_av,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.lb_all,'backgroundcolor',[1 1 1],'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.lb_ch,'backgroundcolor',[1 1 1],'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.lb_av,'backgroundcolor',[1 1 1],'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.text3,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.text4,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.text5,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.text7,'backgroundcolor',obj.container.container.preferences.gui.backgroundColor,'foregroundcolor',obj.container.container.preferences.gui.fontColor);
set(handles.figure1,'color',obj.container.container.preferences.gui.backgroundColor);
set(handles.lb_all,'String',obj.label);
try
    h = obj.plotMontage(false);
    delete(h);
    h = zeros(obj.numberOfChannels,1);
    hold(handles.axes1,'on');
    for it=1:obj.numberOfChannels
        h(it) = scatter3(obj.channelSpace(it,1),obj.channelSpace(it,2),obj.channelSpace(it,3),'filled','MarkerEdgeColor','k','MarkerFaceColor','y','parent',handles.axes1);
    end
    hold(handles.axes1,'off');
    set(handles.axes1,'XColor',obj.container.container.preferences.gui.fontColor,'YColor',obj.container.container.preferences.gui.fontColor,'ZColor',obj.container.container.preferences.gui.fontColor)
    rotate3d on;
catch ME
    if strcmp(ME.identifier,'MoBILAB:noChannelSpace')
        position = [mean(get(handles.axes1,'xlim')) mean(get(handles.axes1,'ylim'))]*1/2;
        text(position(1),position(2),ME.message,'Parent',handles.axes1);
        h = [];
    else
        ME.rethrow;
    end
    
end
set(handles.figure1,'UserData',h,'Visible','on');



% --- Outputs rm1 this function are returned add the command line.
function varargout = ReReferencing_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;



% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle add listbox1 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item rm1 listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle add listbox1 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle add listbox2 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item rm1 listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle add listbox2 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
channels2BeReferenced = get(handles.lb_ch,'String');
channels2BeAveraged = get(handles.lb_av,'String');
set(handles.figure1,'userData',{channels2BeReferenced,channels2BeAveraged});
uiresume



% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
uiresume

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle add pushbutton5 (see GCBO)
% eventdata  reserved - add be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in lb_ch.
function lb_ch_Callback(hObject, eventdata, handles)
% hObject    handle to lb_ch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lb_ch contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_ch


% --- Executes during object creation, after setting all properties.
function lb_ch_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_ch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lb_av.
function lb_mu_Callback(hObject, eventdata, handles)
% hObject    handle to lb_av (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lb_av contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_av


% --- Executes during object creation, after setting all properties.
function lb_mu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_av (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lb_av.
function lb_av_Callback(hObject, eventdata, handles)
% hObject    handle to lb_av (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lb_av contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_av


% --- Executes during object creation, after setting all properties.
function lb_av_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_av (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lb_ch1.
function lb_ch1_Callback(hObject, eventdata, handles)
% hObject    handle to lb_ch1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lb_ch1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_ch1


% --- Executes during object creation, after setting all properties.
function lb_ch1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_ch1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lb_ch2.
function lb_ch2_Callback(hObject, eventdata, handles)
% hObject    handle to lb_ch2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lb_ch2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_ch2


% --- Executes during object creation, after setting all properties.
function lb_ch2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_ch2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lb_all.
function lb_all_Callback(hObject, eventdata, handles)
ind = get(hObject,'Value');
h = get(handles.figure1,'userData');
if isempty(h), return;end
oldInd = get(handles.axes1,'userData');
try set(h(oldInd),'MarkerFaceColor','y');end %#ok
set(h(ind),'MarkerFaceColor','r');
set(handles.axes1,'userData',ind);

% Hints: contents = cellstr(get(hObject,'String')) returns lb_all contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lb_all


% --- Executes during object creation, after setting all properties.
function lb_all_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lb_all (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function referenced_Callback(hObject, eventdata, handles)
avList0 = get(handles.lb_ch,'String');
avList1 = get(handles.lb_all,'String');
ind = get(handles.lb_all,'Value');
if isempty(ind), return;end

if ~isempty(avList0)
    avList = cat(1,avList0,avList1(ind));
else
    avList = avList1(ind);
end
avList = unique(avList);
set(handles.lb_ch,'String',avList);


% --------------------------------------------------------------------
function averaged_Callback(hObject, eventdata, handles)
avList0 = get(handles.lb_av,'String');
avList1 = get(handles.lb_all,'String');
ind = get(handles.lb_all,'Value');
if isempty(ind), return;end

if ~isempty(avList0)
    avList = cat(1,avList0,avList1(ind));
else
    avList = avList1(ind);
end
avList = unique(avList);
set(handles.lb_av,'String',avList);

% --------------------------------------------------------------------
function uicmenu_Callback(hObject, eventdata, handles)
% hObject    handle to uicmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over lb_all.
function lb_all_ButtonDownFcn(hObject, eventdata, handles)



% --------------------------------------------------------------------
function rm2_Callback(hObject, eventdata, handles)
list = get(handles.lb_av,'String');
if isempty(list), return;end
ind = get(handles.lb_av,'Value');
if isempty(ind), return;end
list(ind) = [];
list = unique(list);
if isempty(list), list = '';end
set(handles.lb_av,'Value',1,'String',list);


% --------------------------------------------------------------------
function rm1_Callback(hObject, eventdata, handles)
list = get(handles.lb_ch,'String');
if isempty(list), return;end
ind = get(handles.lb_ch,'Value');
if isempty(ind), return;end
list(ind) = [];
list = unique(list);
if isempty(list), list = '';end
set(handles.lb_ch,'Value',1,'String',list);


% --------------------------------------------------------------------
function sall_Callback(hObject, eventdata, handles)
list = get(handles.lb_all,'String');
if isempty(list), return;end
set(handles.lb_all,'Value',1:length(list));

ind = 1:length(list);
h = get(handles.figure1,'userData');
if isempty(h), return;end
oldInd = get(handles.axes1,'userData');
try set(h(oldInd),'MarkerFaceColor','y');end %#ok
set(h(ind),'MarkerFaceColor','r');
set(handles.axes1,'userData',ind);


% --------------------------------------------------------------------
function sall1_Callback(hObject, eventdata, handles)
list = get(handles.lb_ch,'String');
if isempty(list), return;end
set(handles.lb_ch,'Value',1:length(list));


% --------------------------------------------------------------------
function sall2_Callback(hObject, eventdata, handles)
list = get(handles.lb_av,'String');
if isempty(list), return;end
set(handles.lb_av,'Value',1:length(list));
