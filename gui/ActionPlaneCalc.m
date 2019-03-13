function varargout = ActionPlaneCalc(varargin)
% ACTIONPLANECALC MATLAB code for ActionPlaneCalc.fig
%      ACTIONPLANECALC, by itself, creates a new ACTIONPLANECALC or raises the existing
%      singleton*.
%
%      H = ACTIONPLANECALC returns the handle to a new ACTIONPLANECALC or the handle to
%      the existing singleton*.
%
%      ACTIONPLANECALC('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ACTIONPLANECALC.M with the given input arguments.
%
%      ACTIONPLANECALC('Property','Value',...) creates a new ACTIONPLANECALC or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ActionPlaneCalc_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ActionPlaneCalc_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ActionPlaneCalc

% Last Modified by GUIDE v2.5 31-Jan-2012 15:23:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @ActionPlaneCalc_OpeningFcn, ...
    'gui_OutputFcn',  @ActionPlaneCalc_OutputFcn, ...
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


% --- Executes just before ActionPlaneCalc is made visible.
function ActionPlaneCalc_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ActionPlaneCalc (see VARARGIN)

% Choose default command line output for ActionPlaneCalc
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];

CData = imread([path filesep 'inlinePrj.png']);  set(handles.inlinePrj,'CData',CData);
CData = imread([path filesep 'outlinePrj.png']); set(handles.outlinePrj,'CData',CData);
CData = imread([path filesep 'scaleBy.png']);    set(handles.scale,'CData',CData);
CData = imread([path filesep 'subtract.png']);    set(handles.minus,'CData',CData);
CData = imread([path filesep 'inlinePrj.png']);  set(handles.currentCommand,'CData',CData,'userData','inlinePrj');

try
    allDataStreams = evalin('base','allDataStreams');
catch ME
    ME.rethrow;
end
if isempty(varargin)
    
    N = length(allDataStreams.item);
    index = zeros(N,1);
    segmentName = cell(N,1);
    for it=1:length(allDataStreams.item)
        if isa(allDataStreams.item{it},'projectedMocap')
            index(it,1) = it;
            segmentName{it} = allDataStreams.item{it}.segmentObj.segmentName;
        end
    end
    I = index==0;
    index(I) = [];
    segmentName(I) = [];
    [uSegmentName,loc] = unique(segmentName,'first');
    index = index(loc);
    if isempty(index), warndlg2('You have to do PCA on segmented data before you use this tool.');return;end
    if length(uSegmentName) > 1
        bgObj = allDataStreams.viewLogicalStructure('add2ActionPlaneCalc_Callback',index);
        drawnow;
        set(handles.figure1,'userData',bgObj.hgFigure);
    elseif length(uSegmentName) == 1
        [~,bgObj] = allDataStreams.viewLogicalStructure;
        index = getIndex4aBranch(bgObj,index);
        if length(index)<2, warndlg2('You must have more than two PCA-projected items to use this tool.');return;end
        N = length(index);
        nameList = cell(N,1);
        for it=1:N
            nameList{it} = ['(' num2str(index(it)) ') ' allDataStreams.item{index(it)}.name];
        end
        set([handles.popupmenu3 handles.popupmenu4],'String',nameList);
        set([handles.popupmenu3 handles.popupmenu4],'Value',1);
    end
else
    if isnumeric(varargin{1});
        seed = varargin{1};
        [~,bgObj] = allDataStreams.viewLogicalStructure;
        index = getIndex4aBranch(bgObj,seed);
        if isempty(index), warndlg2('You must have more than two PCA-projected items to use this tool.');return;end
        N = length(index);
        nameList = cell(N,1);
        for it=1:N
            nameList{it} = ['(' num2str(index(it)) ') ' allDataStreams.item{index(it)}.name];
        end
        set([handles.popupmenu3 handles.popupmenu4],'String',nameList);
        set([handles.popupmenu3 handles.popupmenu4],'Value',1);
    end
end



% --- Outputs from this function are returned to the command line.
function varargout = ActionPlaneCalc_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
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


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in inlinePrj.
function inlinePrj_Callback(hObject, eventdata, handles)
h = get(handles.figure1,'userData');close(h(ishandle(h)));
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep 'inlinePrj.png']);  
set(handles.currentCommand,'CData',CData,'userData','inlinePrj');



% --- Executes on button press in outlinePrj.
function outlinePrj_Callback(hObject, eventdata, handles)
h = get(handles.figure1,'userData');close(h(ishandle(h)));
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep 'outlinePrj.png']);  
set(handles.currentCommand,'CData',CData,'userData','outlinePrj');



% --- Executes on button press in scale.
function scale_Callback(hObject, eventdata, handles)
h = get(handles.figure1,'userData');close(h(ishandle(h)));
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep 'scaleBy.png']);  
set(handles.currentCommand,'CData',CData,'userData','scale');


% --- Executes on button press in minus.
function minus_Callback(hObject, eventdata, handles)
h = get(handles.figure1,'userData');close(h(ishandle(h)));
path = fileparts(which('eeglab'));
path = [path filesep 'plugins' filesep 'mobilab' filesep 'skin'];
CData = imread([path filesep 'subtract.png']);  
set(handles.currentCommand,'CData',CData,'userData','minus');



% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3


% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu4.
function popupmenu4_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu4


% --- Executes during object creation, after setting all properties.
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in currentCommand.
function currentCommand_Callback(hObject, eventdata, handles)
% hObject    handle to currentCommand (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of currentCommand


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
index = get([handles.popupmenu3 handles.popupmenu4],'Value');
nameList = get(handles.popupmenu3,'String');
item1 = nameList{index{1}};
ind(1) = find(item1=='(');
ind(2) = find(item1==')');
item1 = str2double(item1(ind(1)+1:ind(2)-1));


item2 = nameList{index{2}};
ind(1) = find(item2=='(');
ind(2) = find(item2==')');
item2 = str2double(item2(ind(1)+1:ind(2)-1));

allDataStreams = evalin('base','allDataStreams');
a = allDataStreams.item{item1};
b = allDataStreams.item{item2};
if a==b, warndlg2('Vector a and b must be different!!!');return;end
try
    switch get(handles.currentCommand,'userData');
        case 'inlinePrj'
            fprintf('Running:\n   allDataStreams.item{%i}.projectInlineItem(%i);\n',item1,item2);
            obj = a.projectInlineItem(item2);
        case 'outlinePrj'
            fprintf('Running:\n   allDataStreams.item{%i}.projectOutlineItem(%i);\n',item1,item2);
            obj = a.projectOutlineItem(item2);
        case 'scale'
            fprintf('Running:\n   allDataStreams.item{%i}.scaleByItem(%i);\n',item1,item2);
            obj = a.scaleByItem(item2);
        case 'minus'
            fprintf('Running:\n   allDataStreams.item{%i} - allDataStreams.item{%i};\n',item1,item2)
            obj = b - a;
    end
    nameList{end+1} = ['(' num2str(length(allDataStreams.item)) ') ' obj.name];
    set([handles.popupmenu3 handles.popupmenu4],'String',nameList);
        
catch ME
    errordlg(ME.message);
end

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
