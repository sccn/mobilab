function varargout = MocapPipeline(varargin)
% MOCAPPIPELINE MATLAB code for MocapPipeline.fig
%      MOCAPPIPELINE, by itself, creates a new MOCAPPIPELINE or raises the existing
%      singleton*.
%
%      H = MOCAPPIPELINE returns the handle to a new MOCAPPIPELINE or the handle to
%      the existing singleton*.
%
%      MOCAPPIPELINE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MOCAPPIPELINE.M with the given input arguments.
%
%      MOCAPPIPELINE('Property','Value',...) creates a new MOCAPPIPELINE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MocapPipeline_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MocapPipeline_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MocapPipeline

% Last Modified by GUIDE v2.5 27-Jul-2012 23:13:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MocapPipeline_OpeningFcn, ...
                   'gui_OutputFcn',  @MocapPipeline_OutputFcn, ...
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


% --- Executes just before MocapPipeline is made visible.
function MocapPipeline_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MocapPipeline (see VARARGIN)

handles.output = hObject;
if ~isa(varargin{1},'mobilabApplication'), error('Cannot run without mobilab.');end
mobilab = varargin{1};
handles.mobilab = mobilab;
guidata(hObject, handles);

path = fullfile(mobilab.path,'skin');
CData = imread([path filesep '32px-Gnome-preferences-system.svg.png']);

set(handles.figure1,'Color',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel1,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.uipanel2,'BackgroundColor',mobilab.preferences.gui.backgroundColor);
set(handles.listbox1,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.listbox2,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.checkbox1,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.checkbox2,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.checkbox3,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.checkbox5,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.preferences,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor,'CData',CData);
set(handles.selectFunction,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.help,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.cancel,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.save,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.run,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.text1,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.text2,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.text3,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);
set(handles.popupmenu1,'BackgroundColor',mobilab.preferences.gui.backgroundColor,'ForegroundColor',mobilab.preferences.gui.fontColor);

ind = mobilab.allStreams.getItemIndexFromItemClass('mocap');
if isempty(ind), error('No motipn capture in this folder.');end
indDescendants = mobilab.allStreams.gObj.getDescendants(1)-1;

loc = ismember(indDescendants,ind);
ind = indDescendants(loc);
N = length(ind);
name = cell(N,1);
uuid = cell(N,1);
for it=1:N
    name{it} = mobilab.allStreams.item{ind(it)}.name;
    uuid{it} = mobilab.allStreams.item{ind(it)}.uuid;
end
set(handles.listbox1,'String',name,'userData',uuid);

N = length(indDescendants);
name = cell(N,1);
uuid = cell(N,1);
rmThis = false(N,1);
for it=1:N
    if ~isempty(mobilab.allStreams.item{indDescendants(it)}.event.uniqueLabel)
        name{it} = mobilab.allStreams.item{indDescendants(it)}.name;
        uuid{it} = mobilab.allStreams.item{indDescendants(it)}.uuid;
    else
        rmThis(it) = true;
    end
end
name(rmThis) = [];
uuid(rmThis) = [];
set(handles.popupmenu1,'String',name,'userData',uuid,'Value',1);

preferences.interpolation = handles.mobilab.preferences.mocap.interpolation;
preferences.cutOff = handles.mobilab.preferences.mocap.lowpassCutoff;
preferences.order = handles.mobilab.preferences.mocap.derivationOrder;
set(handles.preferences,'userData',preferences);
set(handles.checkbox1,'Value',true);
set(handles.checkbox2,'Value',true);
set(handles.checkbox3,'Value',false);


% --- Outputs from this function are returned to the command line.
function varargout = MocapPipeline_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes on button press in preferences.
function preferences_Callback(hObject, eventdata, handles)
preferences = get(handles.preferences,'userData');
prefObj = [...
    PropertyGridField('interpolation', preferences.interpolation,'Type',PropertyType('char', 'row', {'pchip','linear','nearest','spline'}),...
    'Category', 'Mocap', 'DisplayName', 'Interpolation method','Description',...
    'Interpolation method to fill-in occluded markers at certain time point (spline methods are recommended: ''pchip'', and ''spline'').')...
    PropertyGridField('cutOff', preferences.cutOff, 'Category', 'Mocap', 'DisplayName', 'Lowpass cutoff','Description','')...
    PropertyGridField('order', preferences.order, 'Category', 'Mocap', 'DisplayName', 'Derivative order','Description',...
    'Specifies the maximum order of mocap time derivatives, set it to 3 will compute: 1) velocity, 2) acceleration, and 3) jerk.')];

prefObj = prefObj.GetHierarchy();

% create figure
f = figure('MenuBar','none','Name','Preferences','NumberTitle', 'off','Toolbar', 'none');
position = get(f,'position');
set(f,'position',[position(1:2) 385 224]);
g = PropertyGrid(f,'Properties', prefObj,'Position', [0 0 1 1]);
uiwait(f); % wait for figure to close

val = g.GetPropertyValues();
preferences.interpolation = val.interpolation;
preferences.cutOff = val.cutOff;
preferences.order = val.order;
set(handles.preferences,'userData',preferences);



% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2


% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3


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


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in selectFunction.
function selectFunction_Callback(hObject, eventdata, handles)
[FileName,PathName] = uigetfile2({'*.m','MATLAB file'},'Select the .m file');
if any([isnumeric(FileName) isnumeric(PathName)]),
    set(handles.selectFunction,'String','selectFunction');
    return;
end
mfile = fullfile(PathName,FileName);
set(handles.selectFunction,'String',FileName(1:end-2),'userData',mfile);



% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


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


% --- Executes on button press in checkbox5.
function checkbox5_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox5


% --- Executes on button press in save.
function outFile = save_Callback(hObject, eventdata, handles)
uuids = get(handles.listbox2,'userData');
if isempty(uuids), return;end
source = handles.mobilab.allStreams.findItem(uuids{1});
sourceName = handles.mobilab.allStreams.item{source}.name;

preferences = get(handles.preferences,'userData');

cmd{1} = sprintf('mocapIndex = mobilab.allStreams.getItemIndexFromItemName(''%s'');',sourceName);
cmd{2} = sprintf('tmpObj = mobilab.allStreams.item{ mocapIndex };');
if get(handles.checkbox1,'Value')
    interpolation = preferences.interpolation;
    cmd{end+1} = sprintf('tmpObj = tmpObj.removeOcclusionArtifact(''%s'');',interpolation);
end
if get(handles.checkbox2,'Value')
    cutOff = preferences.cutOff;
    cmd{end+1} = sprintf('tmpObj = tmpObj.lowpass(%i);',cutOff);
end
if get(handles.checkbox3,'Value')
    order = preferences.order;
    cmd{end+1} = sprintf('tmpObj = tmpObj.smoothDerivative(%i);',order);
end

funct = get(handles.selectFunction,'userData');
if ~isempty(funct)
    uuids = get(handles.popupmenu1,'userData');
    ind = get(handles.popupmenu1,'Value');
    eventItem = handles.mobilab.allStreams.findItem(uuids{ind});
    eventItemName = handles.mobilab.allStreams.item{eventItem}.name;
    
    cmd{end+1} = sprintf('eventIndex = mobilab.allStreams.getItemIndexFromItemName(''%s'');',eventItemName);
    cmd{end+1} = sprintf('eventObj = mobilab.allStreams.item{ eventIndex };');
    
    [newPath,functName] = fileparts(funct);
    cmd{end+1} = sprintf('cd(''%s'');',newPath);
    cmd{end+1} = sprintf('[ startEndTimePointes, conditions, channels] = %s( eventObj );',functName);
    cmd{end+1} = sprintf('cd(''%s'');', pwd);
    cmd{end+1} = sprintf('\n%% loop by conditions\nN = length(conditions);\nfor it=1:N\n    try');
    cmd{end+1} = '        bsObj = basicSegment( startEndTimePointes{it},  conditions{it} );';
    cmd{end+1} = sprintf('\n        %% segmenting\n        segObj = bsObj.apply( tmpObj ,channels{it} ); ');
    cmd{end+1} = sprintf('\n        %% PCA\n        prjObj = projectDataPCA( segObj ); ');
    cmd{end+1} = sprintf('\n        %% time derivative\n        smoothDerivative( prjObj, %i, %i );',preferences.order,preferences.cutOff*3);    
    cmd{end+1} = '    catch ME';
    cmd{end+1} = '        disp(ME.message);';
    cmd{end+1} = '    end';
    cmd{end+1} = 'end';
end
outFile = fullfile(handles.mobilab.allStreams.mobiDataDirectory,'mocapScript.m');
fid = fopen(outFile,'w');
for it=1:length(cmd), fprintf(fid,'%s\n',cmd{it});end
fclose(fid);
fprintf('\nSaved in:\n');
fprintf([' ' outFile '\n\n']);




% --- Executes on button press in cancel.
function cancel_Callback(hObject, eventdata, handles)
close(handles.figure1)


% --- Executes on button press in help.
function help_Callback(hObject, eventdata, handles)
% hObject    handle to help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in checkbox6.
function checkbox6_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox6


% --- Executes on button press in run.
function run_Callback(hObject, eventdata, handles)
scriptFile = save_Callback(hObject, eventdata, handles);
handles.mobilab.batch(scriptFile);


% --------------------------------------------------------------------
function selectAll_Callback(hObject, eventdata, handles)
N = length(get(handles.listbox1,'String'));
if ~N, return;end
set(handles.listbox1,'Value',1:N);


% --------------------------------------------------------------------
function add_Callback(hObject, eventdata, handles)
name = get(handles.listbox1,'String');
uuid = get(handles.listbox1,'userData');
ind = get(handles.listbox1,'Value');
set(handles.listbox2,'Value',1,'String',name(ind),'userData',uuid(ind));



% --------------------------------------------------------------------
function pool_Callback(hObject, eventdata, handles)
% hObject    handle to pool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function selectAll2_Callback(hObject, eventdata, handles)
N = length(get(handles.listbox2,'String'));
if ~N, return;end
set(handles.listbox2,'Value',1:N);


% --------------------------------------------------------------------
function toProcess_Callback(hObject, eventdata, handles)
% hObject    handle to toProcess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function rm_Callback(hObject, eventdata, handles)
name = get(handles.listbox2,'String');
uuid = get(handles.listbox2,'userData');
ind = get(handles.listbox2,'Value');
name(ind) = [];
uuid(ind);
if isempty(uuid)
    name = {''};
    uuid = {};
end
set(handles.listbox2,'Value',1,'String',name,'userData',uuid);
