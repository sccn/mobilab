function fig = Export2EEGLAB(mobilab)
fig = figure('Menu','none','WindowStyle','normal','NumberTitle','off','Name','Export to EEGLAB','UserData',mobilab);
fig.Position(3:4) = [560   340];
n = length(mobilab.allStreams.item);
TData= cell(n,3);
for k=1:n
    TData{k,1} = false;
    TData{k,2} = mobilab.allStreams.item{k}.name;
    TData{k,3} = class(mobilab.allStreams.item{k});
end
tbl_source = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name','Type'},'Units','normalized','Position',[0.03 0.2647 0.45 0.65],...
    'TooltipString','Select the items containing the events of interest','ColumnEditable',true);
tbl_target = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name','Type'},'Units','normalized','Position',[0.53 0.2647, 0.45 0.65],...
    'TooltipString','Select the items receiving the events of interest','ColumnEditable',true);

uicontrol(fig,'Style','text','String','Data','Position',[84 314 60 20],'FontWeight','bold','Units','normalized');
uicontrol(fig,'Style','text','String','Events','Position',[394 314 60 20],'FontWeight','bold','Units','normalized');
uicontrol(fig,'Style','text','String','EEG','Position',[254 6 60 23],'FontWeight','bold','Units','normalized');

try
    skinPath = [fileparts(fileparts(which('CoreBrowser.m'))) filesep 'skin'];
    insertIcon  = imread([skinPath filesep '32px-Gnome-media-seek-forward.svg.png']);
catch
    skinPath = fileparts(which('CoreBrowser.m'));
    insertIcon  = imread(pickfiles(skinPath,'32px-Gnome-media-seek-forward.svg.png'));
end
uicontrol('Parent', fig, 'Style', 'pushbutton','Callback',@onExport,'CData',permute(insertIcon,[2 1 3]),...
    'Position',[259 43 47 43],'TooltipString','Export to EEGLAB','Units','normalized');
end

function onExport(src, evnt)
mobilab = src.Parent.UserData;
tbl_data = src.Parent.Children(6);
tbl_events = src.Parent.Children(5);
sel = cell2mat(tbl_data.Data(:,1));
data = mobilab.allStreams.getItemIndexFromItemName(tbl_data.Data(sel,2));
sel = cell2mat(tbl_events.Data(:,1));
evnts = mobilab.allStreams.getItemIndexFromItemName(tbl_events.Data(sel,2));
src.Parent.Pointer='watch';
drawnow;
mobilab.allStreams.export2eeglab(data,evnts);
src.Parent.Pointer='arrow';
drawnow;
end
