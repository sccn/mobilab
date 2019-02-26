classdef dataSourceFromFolder < dataSource
    methods
        %%
        function obj = dataSourceFromFolder(folder,mobiDataDirectory)
            if nargin < 2, error('No enough input arguments.');end
            
            uuid = generateUUID;
            obj@dataSource(mobiDataDirectory,uuid);
            obj.checkThisFolder(mobiDataDirectory);
            
            files = dir(folder);
            files([1 2]) = [];
            t0 = datenum({files.date},'dd-mmm-yyyy');
            files =  {files.name};
            indFiles = zeros(length(files),1);
            for it=1:length(files)
                if strcmp(files{it}(end-3:end),'.drf')
                    indFiles(it) = it;
                elseif strcmp(files{it}(end-3:end),'.xdf')
                    indFiles(it) = it;
                elseif strcmp(files{it}(end-3:end),'.xdfz')
                    indFiles(it) = it;
                end
            end
            indFiles(indFiles==0) = [];
            if isempty(indFiles), error('MoBILAB:unwknownFormat','Unknown file format.');end
            files = files(indFiles);
            Nf = length(files);
            t0 = t0(indFiles);
            [~,loc] = sort(t0);
            files = files(loc);
            [~,~,ext] = fileparts(files{1});
            switch ext
                case '.drf', readClass = @dataSourceDRF;
                case '.xdf', readClass = @dataSourceXDF;
                case 'xdfz', readClass = @dataSourceXDF;
                otherwise, error('MoBILAB:unwknownFormat','Unknown file format.');
            end
            
            seeLogFile = false;
            dsourceObjs = cell(Nf,1);
            tmpDir = cell(Nf,1);
            obj.container.initStatusbar(1,Nf,'Reading files. Time to go for a coffee...');
            for it=1:Nf
                tmpDir{it} = fullfile(obj.mobiDataDirectory, ['tmpDir' num2str(it)]);
                mkdir(tmpDir{it});
                dsourceObjs{it} = readClass(fullfile(folder,files{it}),tmpDir{it});
                logfile = pickfiles(dsourceObjs{it}.mobiDataDirectory,'logfile.txt');
                if ~isempty(logfile)
                    txt = readtxtfile(logfile);
                    fid = fopen([obj.mobiDataDirectory filesep 'logfile.txt'],'a');
                    fprintf(fid,'\n------------\nLogs file: %s',fullfile(folder,files{it}));
                    fprintf(fid,'%s\n',txt);
                    fclose(fid);
                    seeLogFile = true;
                end
                obj.container.statusbar(it);
            end
            
            % calling the merge method of the correspondent class
            dataSourceType = class(dsourceObjs{1});
            eval([dataSourceType '.merge(obj,dsourceObjs);']);
            
            disp('Cleaning temporal files and folders...');
            for it=1:Nf
                folder = dsourceObjs{it}.mobiDataDirectory;
                delete(dsourceObjs{it});
                files = dir(folder);
                files(1:2) = [];
                files = {files.name};
                for jt=1:length(files), java.io.File([folder filesep files{jt}]).delete();end
                rmdir(folder,'s')
            end
            obj.findSpaceBoundary;
            obj.updateLogicalStructure;
            if seeLogFile, disp(['Logs were saved in: ' [obj.mobiDataDirectory filesep 'logfile.txt']]);end
        end
    end
end