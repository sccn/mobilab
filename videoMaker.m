classdef videoMaker < handle
    properties
        nowCursor
        hTimer
        hGraphics
        period
        hStreamObj
        videoFilename
        tmin
        tmax
        paintCallback
    end
    properties(Hidden = true)
        vObj
        progressBar
        statusBar
    end
    methods 
        function obj = videoMaker(hStreamObj,tmin,tmax,period, saveAs,hGraphics,paintCallback)
            obj.hStreamObj = hStreamObj;
            obj.tmin = tmin;
            obj.tmax = tmax;
            obj.period = period;
            obj.videoFilename = saveAs;
            obj.hGraphics = hGraphics;
            obj.paintCallback = paintCallback;
            obj.nowCursor = obj.tmin;
            hFigure = get(get(hGraphics,'Parent'),'Parent');
            obj.vObj = VideoWriter(obj.videoFilename, 'Motion JPEG AVI');
            obj.vObj.FrameRate = 1/period;
            open(obj.vObj);
            obj.statusBar = com.mathworks.mwswing.MJStatusBar;
            obj.progressBar = javax.swing.JProgressBar;
            set(obj.progressBar, 'Minimum',obj.tmin, 'Maximum',obj.tmax, 'Value',obj.tmin);
            obj.statusBar.add(obj.progressBar,'West');
            jFrame = get(handle(hFigure),'JavaFrame');
            jRootPane = jFrame.fHG1Client.getWindow;
            jRootPane.setStatusBar(obj.statusBar)
            obj.statusBar.setText('Writing...')
            obj.hTimer = timer('TimerFcn',{@playCallback, obj}, 'Period', period,'BusyMode','queue','ExecutionMode','fixedRate');
            start(obj.hTimer);
        end
        function delete(obj)
            close(obj.vObj)    
            stop(obj.hTimer);
            delete(obj.hTimer);
        end
    end
end

function playCallback(tobj,event,obj) %#ok
if obj.nowCursor+obj.period > obj.tmax
    stop(obj.hTimer);
    obj.statusBar.setText('Done')
    hFigure = get(get(obj.hGraphics,'Parent'),'Parent');
    close(hFigure);
    delete(obj);
    return
end
obj.paintCallback(obj.hGraphics,obj.nowCursor);

hAxes = get(obj.hGraphics,'Parent');
frame = getframe(hAxes);
if any(frame.cdata(:,end,1)), frame.cdata(:,end,:) = 0;end
writeVideo(obj.vObj,frame);
obj.nowCursor = obj.nowCursor+obj.period;
set(obj.progressBar,'Value',obj.nowCursor);
end