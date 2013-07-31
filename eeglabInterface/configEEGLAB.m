function configEEGLAB

mobilab_opt_file = which('mobilab_eeg_options.m');
if ~isempty(mobilab_opt_file)
    copyfile(mobilab_opt_file,[getHomeDir filesep 'eeg_options.m']);
else
    opt_file = [getHomeDir filesep 'eeg_options.m'];
    fid = fopen(opt_file,'w');
    fprintf(fid,'option_storedisk = 0 ;    %% If set, keep at most one dataset in memory. This allows processing hundreds of datasets within studies.\n');
    fprintf(fid,'option_savetwofiles = 1 ; %% If set, save not one but two files for each dataset (header and data). This allows faster data loading in studies.\n');
    fprintf(fid,'option_saveica = 0 ; %% If set, write ICA activations to disk. This speeds up loading ICA components when dealing with studies.\n');
    fprintf(fid,'%% Memory options\n'); 
    fprintf(fid,'option_single = 1 ; %% If set, use single precision under Matlab 7.x. This saves RAM but can lead to rare numerical imprecisions.\n');
    fprintf(fid,'option_memmapdata = 1 ; %% If set, use memory mapped array under Matlab 7.x. This may slow down some computation.\n');
    fprintf(fid,'option_eegobject = 0 ; %% If set, use the EEGLAB EEG object instead of the standard EEG structure (beta).\n');
    fprintf(fid,'%% ICA options\n');
    fprintf(fid,'option_computeica = 1 ; %% If set, precompute ICA activations. This requires more RAM but allows faster plotting of component activations.\n');
    fprintf(fid,'option_scaleicarms = 1 ; %% If set, scale ICA component activities to RMS (Root Mean Square) in microvolt (recommended).\n');
    fprintf(fid,'%% Folder options\n');
    fprintf(fid,'option_rememberfolder = 1 ; %% If set, when browsing to open a new dataset assume the folder/directory of previous dataset.\n');
    fprintf(fid,'%% Toolbox options\n');
    fprintf(fid,'option_donotusetoolboxes = 0 ; %% If set, do not use Matlab additional toolboxes functions even if they are present.\n');
    fprintf(fid,'%% EEGLAB chat option\n');
    fprintf(fid,'option_chat = 0 ; %% If set, enable EEGLAB chat - currently UCSD only - restart EEGLAB after changing that option.\n');
    fclose(fid);
end