function fig = InsertEvents(mobilab)
fig = figure('Menu','none','WindowStyle','normal','NumberTitle','off','Name','Insert events','UserData',mobilab);
fig.Position(3:4) = [570   256];
n = length(mobilab.allStreams.item);
TData= cell(n,2);
for k=1:n
    TData{k,1} = false;
    TData{k,2} = mobilab.allStreams.item{k}.name;
end
tbl_source = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name'},'Units','normalized','Position',[0.03 0.05 0.4 0.85],...
    'TooltipString','Select the items containing the events of interest','ColumnEditable',true);
tbl_target = uitable(fig, 'Data',TData,'ColumnName',{'Select','Name'},'Units','normalized','Position',[0.56 0.05, 0.4, 0.85],...
    'TooltipString','Select the items receiving the events of interest','ColumnEditable',true);

uicontrol(fig,'Style','text','String','Source','Position',[84 232 60 20],'FontWeight','bold');
uicontrol(fig,'Style','text','String','Target','Position',[394 232 60 20],'FontWeight','bold');

skinPath = [fileparts(fileparts(which('CoreBrowser.m'))) filesep 'skin'];
insertIcon  = imread([skinPath filesep '32px-Gnome-media-seek-forward.svg.png']);
btn_insert = uicontrol('Parent', fig, 'Style', 'pushbutton','Callback',@onInsert,'CData',insertIcon,...
    'Position',[259 106 47 43],'TooltipString','Insert event markers');
end

function onInsert(src, evnt)
mobilab = src.Parent.UserData;
tbl_source = src.Parent.Children(5);
tbl_target = src.Parent.Children(4);
sel = cell2mat(tbl_source.Data(:,1));
source = mobilab.allStreams.getItemIndexFromItemName(tbl_source.Data(sel,2));
sel = cell2mat(tbl_target.Data(:,1));
target = mobilab.allStreams.getItemIndexFromItemName(tbl_target.Data(sel,2));
for t=1:length(target)
    for s=1:length(source)
        srcObj = mobilab.allStreams.item{source(s)};
        srcLatency = srcObj.timeStamp(srcObj.event.latencyInFrame);
        
        trgObj = mobilab.allStreams.item{target(t)};
        trgLatency = trgObj.getTimeIndex(srcLatency);
        trgObj.event = trgObj.event.addEvent(trgLatency, srcObj.event.label);
    end
end
end
