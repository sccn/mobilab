classdef dataSourceFromDRFFolder < dataSource
    methods
        %%
        function obj = dataSourceFromDRFFolder(folder,mobiDataDirectory)
            if nargin < 2, error('No enough input arguments.');end
            
            if ismac, [~,uuid] = system('uuidgen'); else uuid =  java.util.UUID.randomUUID;end
            obj@dataSource(mobiDataDirectory,char(uuid));
            obj.checkThisFolder(mobiDataDirectory);
            
            files = dir(folder);
            files([1 2]) = [];
            files =  {files.name};
            indFiles = zeros(length(files),1);
            for it=1:length(files), if strcmp(files{it}(end-3:end),'.drf'), indFiles(it) = it;end;end
            indFiles(indFiles==0) = [];
            if isempty(indFiles), disp('There are no .drf files in this folder.');obj = [];return;end
            files = files(indFiles);
            Nf = length(files);
            
            tmp = what(mobiDataDirectory);
            if isempty(tmp), error('prog:input','Second argument must be a valid folder.');end
            mobiDataDirectory = tmp.path;
            
            obj.mobiDataDirectory = mobiDataDirectory;
            obj.dataSourceLocation = fullfile(mobiDataDirectory,'descriptor.MoBI');
            obj.source = folder;
            
            tmpObj = cell(Nf,1);
            tmpDir = cell(Nf,1);
            obj.container.initStatusbar(1,Nf,'Reading files. Time to go for a coffee...');
            for it=1:Nf
                tmpDir{it} = fullfile(obj.mobiDataDirectory, ['tmpDir' num2str(it)]);
                mkdir(tmpDir{it});
                tmpObj{it} = dataSourceDRF(fullfile(folder,files{it}),tmpDir{it});
                obj.container.statusbar(it);
            end
            
            t0 = zeros(Nf,1);
            for it=1:Nf
                t0(it) = tmpObj{it}.item{1}.hardwareMetaData.originalTimeStamp(1);
            end
            [~,loc] = sort(t0);
            
            parentCommand.commandName = 'dataSourceFromDRFFolder';
            parentCommand.varargin = {folder,mobiDataDirectory};
            Nstreams = length(tmpObj{1}.item);
            for it=1:Nstreams
                
                if ismac, [~,uuid] = system('uuidgen'); else uuid =  java.util.UUID.randomUUID;end
                mmfName = fullfile(obj.mobiDataDirectory, [tmpObj{1}.item{it}.name '_' char(uuid) '.bin']);
                switch tmpObj{1}.item{it}.hardwareMetaData.name;
                    case 'biosemi'
                        obj.item{it} = dataStream('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',tmpObj{1}.item{it}.numberOfChannels,'label',tmpObj{1}.item{it}.label,...
                            'mmfName',mmfName,'writable',false,'parentCommand',parentCommand,'uuid',uuid,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    case 'phasespace'
                        obj.item{it} = mocap('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',tmpObj{1}.item{it}.numberOfChannels,'label',tmpObj{1}.item{it}.label,...
                            'mmfName',mmfName,'writable',false,'parentCommand',parentCommand,'uuid',uuid,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    case 'wii'
                        obj.item{it} = wii('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',tmpObj{1}.item{it}.numberOfChannels,'label',tmpObj{1}.item{it}.label,...
                            'mmfName',mmfName,'writable',false,'parentCommand',parentCommand,'uuid',uuid,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    case 'videostream'
                        obj.item{it} = videoStream('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',1,'label','video frame','mmfName',mmfName,'uuid',uuid,'writable',false,'parentCommand',parentCommand,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    case 'scenestream'
                        obj.item{it} = videoStream('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',1,'label','video frame','mmfName',mmfName,'writable',false,'uuid',uuid,...
                            'parentCommand',parentCommand,'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    case 'hotgazestream'
                        obj.item{it} = gazeStream('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',tmpObj{1}.item{it}.numberOfChannels,'label',tmpObj{1}.item{it}.label,...
                            'mmfName',mmfName,'writable',false,'parentCommand',parentCommand,'uuid',uuid,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                    otherwise
                        obj.item{it} = dataStream('name',tmpObj{1}.item{it}.name,'samplingRate',tmpObj{1}.item{it}.samplingRate,...
                            'numberOfChannels',tmpObj{1}.item{it}.numberOfChannels,'label',tmpObj{1}.item{it}.label,...
                            'mmfName',mmfName,'writable',false,'parentCommand',parentCommand,'uuid',uuid,...
                            'hardwareMetaData',tmpObj{1}.item{it}.hardwareMetaData);
                end
            end
            
            obj.container.initStatusbar(1,Nstreams,'Merging streams...');
            try
                for it=1:Nstreams
                    
                    ev = [];
                    tm = [];
                    latencyBoundaryEvent = zeros(Nf-1,1);
                    for jt=1:Nf
                        if obj.item{it}.numberOfChannels ~= tmpObj{loc(jt)}.item{it}.numberOfChannels
                            error('MoBILAB:merging','The streams you''re merging must have the same number of channels!!!');
                        end
                        ev = [ev tmpObj{loc(jt)}.item{it}.event.event2vector(tmpObj{loc(jt)}.item{it}.timeStamp)];%#ok
                        if jt > 1
                            tm = [tm tmpObj{loc(jt)}.item{it}.timeStamp+tm(end)+1/obj.item{it}.samplingRate];%#ok
                        else
                            tm = [tm tmpObj{loc(jt)}.item{it}.timeStamp];%#ok
                        end
                        if jt < Nf
                            latencyBoundaryEvent(jt) = length(tmpObj{loc(jt)}.item{it}.timeStamp);
                        end
                    end
                    latencyBoundaryEvent = cumsum(latencyBoundaryEvent);
                    obj.item{it}.addSamples([],tm,ev);
                    obj.item{it}.event = obj.item{it}.event.addEvent(latencyBoundaryEvent,'boundary');
                    
                    fid = fopen(obj.item{it}.mmfName,'w');
                    for ch=1:obj.item{it}.numberOfChannels
                        for jt=1:Nf
                            fwrite(fid,tmpObj{loc(jt)}.item{it}.data(:,ch),obj.item{it}.precision);
                        end
                    end
                    fclose(fid);
                    obj.container.statusbar(it);
                end
            catch ME
                if strcmp(ME.identifier,'MoBILAB:merging')
                    obj.container.statusbar(Nstreams);
                    fclose all;
                    for jt=1:Nf
                        tmpObj{jt}.disconnect;
                        for it=1:length(tmpObj{jt}.item)
                            delete(tmpObj{jt}.item{it}.mmfName);
                            delete(tmpObj{jt}.item{it}.header);
                        end
                        rmdir(tmpObj{jt}.mobiDataDirectory,'s');
                        delete(tmpObj{jt});
                    end
                else
                    ME.rethrow;
                end
            end
            
            obj.container = tmpObj{end}.container;
            obj.container.allStreams = obj;
            for jt=1:Nf
                tmpObj{jt}.disconnect;
                for del=1:length(tmpObj{jt}.item)
                    delete(tmpObj{jt}.item{del}.mmfName);
                    delete(tmpObj{jt}.item{del}.header);
                end
                tmpMobiDataDirectory = tmpObj{jt}.mobiDataDirectory;
                delete(tmpObj{jt});
                rmdir(tmpMobiDataDirectory,'s');
            end
            obj.connect;
            obj.findSpaceBoundary;
            obj.linkData;
            obj.updateLogicalStructure;
            obj.save(obj.mobiDataDirectory);
        end
    end
end