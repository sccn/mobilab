%% load raw calibration data (EEGLAB function)
EEG = exp_eval(io_loadset('bcilab:/userdata/tutorial/flanker_task/12-08-001_ERN.vhdr','channels',1:32));
EEG = exp_eval(flt_resample(EEG,250));
EEG = exp_eval(set_selinterval(EEG,1:30000,'samples'));

% design a highpass filter that we can use offline and online (FIR, 0.5Hz-1Hz transition band, minimum phase)
params = firpmord([0.5 1], [0 1], [0.001 0.01], EEG.srate, 'cell');
kernel = firpm(max(3,params{1}),params{2:end});
kernel = kernel+randn(1,length(kernel))*0.00001;
[dummy,kernel] = rceps(kernel);
kernel = kernel/abs(max(freqz(kernel)));

% filter the calibration data
[EEG.data,state] = filter(kernel,1,double(EEG.data),[],2);

%% write to disk
fid = fopen('flanker_set_small.dat','w+');
fwrite(fid,EEG.data,'double','ieee-le.l64');
fclose(fid);

%% read it back in...
FLT = EEG;
fid = fopen('flanker_set_small.dat.out','rb','native');
FLT.data=reshape(fread(fid,'*double'),EEG.nbchan,EEG.pnts);
fclose(fid);
% and display...
PADEEG=EEG; PADEEG.data=[zeros(EEG.nbchan,EEG.srate*0.25) EEG.data]; vis_hist({FLT,PADEEG})

%% loop test
FLT = EEG;
% design a spectrum shaping filter
[shaping_b,shaping_a] = yulewalk(6,[2*[0 2 3 13 14]/EEG.srate 1],[1 0.75 0.3 0.3 1 1]);
[cov_mat,mix_mat,state_ord0,state_ord1,state_ord2,state_buffer] = asr_calibrate(EEG.data, EEG.srate, 0.5, 0.25, shaping_b, shaping_a);
for pos=1:50:EEG.pnts
    rawchunk = double(EEG.data(:,pos:(pos+50-1)));
    [fltchunk,state_ord0,state_ord1,state_ord2,state_buffer] = asr_process(rawchunk, EEG.srate, 0.5, 0.25, 3, 32, 0.66, 256, cov_mat, mix_mat, shaping_b, shaping_a, state_ord0, state_ord1, state_ord2, state_buffer);
    FLT.data(:,pos:(pos+50-1)) = fltchunk;
end
PADEEG=EEG; PADEEG.data=[zeros(EEG.nbchan,EEG.srate*0.25) EEG.data]; vis_hist({FLT,PADEEG})

%% step-by-step-test
[cov_mat,mix_mat,state_ord1,state_ord2,state_buffer] = asr_calibrate(EEG.data, EEG.srate, 0.5, 0.25);
pos = 1; rawchunk = double(EEG.data(:,pos:(pos+50-1))); [fltchunk,state_ord1,state_ord2,state_buffer] = asr_process(rawchunk, EEG.srate, 0.5, 0.25, 3, 32, 0.66, 256, cov_mat, mix_mat, state_ord1, state_ord2, state_buffer);
pos = 51; rawchunk = double(EEG.data(:,pos:(pos+50-1))); [fltchunk,state_ord1,state_ord2,state_buffer] = asr_process(rawchunk, EEG.srate, 0.5, 0.25, 3, 32, 0.66, 256, cov_mat, mix_mat, state_ord1, state_ord2, state_buffer);
    
    
%% do simulated online processing
rawdata = []; fltdata = [];
scale = 75; duration=5;
figure('Position',[0 0 1000,500],'Tag','FigVis');
t0 = tic();
last_pos = 0;
while ~isempty(findobj('Tag','FigVis'))
    % get a new chunk from the simulated device
    pos = round(1+(toc(t0)+duration)*EEG.srate);
    rawchunk = double(EEG.data(:,(last_pos+1):pos)); 
    last_pos = pos;
    % apply a high-pass filter
    [rawchunk,state] = filter(kernel,1,rawchunk,state,2);    
    % apply the artifact removal
    [fltchunk,state_ord1x,state_ord2x,state_bufferx] = asr_process(rawchunk, EEG.srate, 0.5, 0.25, 2.5, 32, 0.5, 256, cov_mat, mix_mat, state_ord1, state_ord2, state_buffer);
    
    % display the raw and filtered data
    rawdata = [rawdata rawchunk]; rawdata(:,1:(size(rawdata,2)-EEG.srate*duration)) = [];
    fltdata = [fltdata fltchunk]; fltdata(:,1:(size(fltdata,2)-EEG.srate*duration)) = [];
    try
        for k=1:length(p{1})
            set(p{1}(k),'Ydata',rawdata(k,:)+k*scale); end
        for k=1:length(p{2})
            set(p{2}(k),'Ydata',fltdata(k,:)+k*scale); end
    catch
        p{1}=plot(EEG.srate*0.25 + (1:size(rawdata,2)),bsxfun(@plus,(1:EEG.nbchan)*scale,rawdata'),'r'); hold on; p{2} = plot((1:size(rawdata,2)),bsxfun(@plus,(1:EEG.nbchan)*scale,fltdata'),'b'); hold off;
    end
    axis([EEG.srate*0.25,size(rawdata,2),0,EEG.nbchan*scale+scale]);
    drawnow;
end
