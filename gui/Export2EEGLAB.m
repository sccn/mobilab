function fig = Export2EEGLAB(mobilab)
fig = figure('Menu','none','WindowStyle','normal','NumberTitle','off','Name','Export to EEGLAB','UserData',mobilab);
fig.Position(3:4) = [560   340];
n = length(mobilab.allStreams.item);
TData= cell(n,2);
for k=1:n
    TData{k,1} = false;
    TData{k,2} = mobilab.allStreams.item{k}.name;
end
tbl_source = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name'},'Units','normalized','Position',[0.03 0.2647 0.45 0.65],...
    'TooltipString','Select the items containing the events of interest','ColumnEditable',true);
tbl_target = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name'},'Units','normalized','Position',[0.53 0.2647, 0.45 0.65],...
    'TooltipString','Select the items receiving the events of interest','ColumnEditable',true);

uicontrol(fig,'Style','text','String','Data','Position',[84 314 60 20],'FontWeight','bold');
uicontrol(fig,'Style','text','String','Events','Position',[394 314 60 20],'FontWeight','bold');
uicontrol(fig,'Style','text','String','EEG','Position',[254 6 60 23],'FontWeight','bold');

skinPath = [fileparts(fileparts(which('CoreBrowser.m'))) filesep 'skin'];
insertIcon  = imread([skinPath filesep '32px-Gnome-media-seek-forward.svg.png']);
uicontrol('Parent', fig, 'Style', 'pushbutton','Callback',@onExport,'CData',permute(insertIcon,[2 1 3]),...
    'Position',[259 43 47 43],'TooltipString','Export to EEGLAB');
end

function onExport(src, evnt)
mobilab = src.Parent.UserData;
tbl_data = src.Parent.Children(6);
tbl_events = src.Parent.Children(5);
sel = cell2mat(tbl_data.Data(:,1));
data = mobilab.allStreams.getItemIndexFromItemName(tbl_data.Data(sel,2));
sel = cell2mat(tbl_events.Data(:,1));
evnts = mobilab.allStreams.getItemIndexFromItemName(tbl_events.Data(sel,2));
mobilab.allStreams.export2eeglab(data,evnts);
end
