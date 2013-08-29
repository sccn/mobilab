function updateCursor(hCursor)
if strcmp(hCursor.visible,'off'), return;end
browserObj = get(get(get(hCursor.Target,'Parent'),'parent'),'userData');
if isempty(browserObj)
    browserObj = get(get(hCursor.Target,'Parent'),'userData');
end
if browserObj.nowCursor == hCursor.Position(1), return;end
try %#ok
    
    if isa(browserObj,'projectionBrowserHandle')
        delete(browserObj.cursorHandle(ishandle(browserObj.cursorHandle)));
        browserObj.cursorHandle.gh(1) = plot3(browserObj.axesHandle,pos(1),pos(2),pos(3),...
            'o','linewidth',2,'MarkerSize',10,'MarkerEdgeColor','k','MarkerFaceColor','r');
    end
    browserObj.nowCursor = hCursor.Position(1);
    
    set(browserObj.timeTexttHandle,'String',['Current latency = ' num2str(browserObj.nowCursor,4) ' sec']);
    set(browserObj.sliderHandle,'Value',browserObj.nowCursor);
    
    if isa(browserObj.master,'browserHandleList')
        
        set( findobj(browserObj.master.master,'Tag','edit10'),'String',num2str(browserObj.nowCursor,4));
        set( findobj(browserObj.master.master,'Tag','text8'),'String',['Current latency = ' num2str(browserObj.nowCursor,4) ' sec']);
        for it=1:length(browserObj.master.list)
            if browserObj.master.list{it} ~= browserObj
                if isa(browserObj.master.list{it},'topographyBrowserHandle') || isa(browserObj.master.list{it},'mocapBrowserHandle') || isa(browserObj.master.list{it},'cometBrowserHandle') ||...
                        isa(browserObj.master.list{it},'vectorBrowserHandle') || isa(browserObj.master.list{it},'videoStreamBrowserHandle') || isa(browserObj.master.list{it},'pcdBrowserHandle')
                    browserObj.master.list{it}.plotThisTimeStamp( hCursor.Position(1));
                    
                elseif isa(browserObj.master.list{it},'projectionBrowserHandle')
                    [~,loc] = min(abs(hCursor.Position(1)-browserObj.master.list{it}.streamHandle.timeStamp));
                    for jt=1:browserObj.master.list{it}.numberOfChannelsToPlot
                        set(browserObj.master.list{it}.cursorHandle.gh(jt),'XData',hCursor.Position(1),'YData',...
                            browserObj.master.list{it}.streamHandle.data(loc,browserObj.master.list{it}.superIndex(jt,1)),'ZData',...
                            browserObj.master.list{it}.streamHandle.data(loc,browserObj.master.list{it}.superIndex(jt,2)),...
                            'linewidth',2,'MarkerFaceColor','r');
                    end
                    browserObj.master.list{it}.nowCursor = browserObj.nowCursor;
                    set(browserObj.master.list{it}.sliderHandle,'Value',browserObj.nowCursor);
                    set(browserObj.master.list{it}.timeTexttHandle,'String',['Current latency = ' num2str(browserObj.nowCursor,4) ' sec']);
                else
                    if ~strcmp(get(browserObj.master.list{it}.cursorHandle.tb,'State'),'off')
                        % figure(browserObj.master.list{it}.figureHandle)
                        browserObj.master.list{it}.cursorHandle.gh.Position(1) = browserObj.nowCursor;
                        %browserObj.master.list{it}.cursorHandle.gh.updatePosition(hCursor);
                    end
                    % browserObj.master.list{it}.plotThisTimeStamp( hCursor.Position(1));
                    browserObj.master.list{it}.nowCursor = browserObj.nowCursor;
                    set(browserObj.master.list{it}.sliderHandle,'Value',browserObj.nowCursor);
                    set(browserObj.master.list{it}.timeTexttHandle,'String',['Current latency = ' num2str(browserObj.nowCursor,4) ' sec']);
                end
            end
        end
        browserObj.master.nowCursor = browserObj.nowCursor;
    end
end