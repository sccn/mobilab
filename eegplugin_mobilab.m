% eegplugin_mobi2eeglab() - MOBILAB plugin for EEGLAB
%                           MOBILAB is a collection of functionalities
%                           for processing MoBI data, see (Makeig et al.,
%                           2009).
% Usage:
%   >> eegplugin_mobilab(fig, trystrs, catchstrs);
%
% Inputs:
%   fig        - [integer] eeglab figure.
%   trystrs    - [struct] "try" strings for menu callbacks.
%   catchstrs  - [struct] "catch" strings for menu callbacks.
%
% Author: Alejandro Ojeda, Nima Bigdely Shamlo, and Christian Kothe, SCCN, INC, UCSD
%
% See also: eeglab()

function v = eegplugin_mobilab(fig,try_strings, catch_strings)

v = '20210311'; % date of release
p = fileparts(which('runmobilab'));
if isempty(p)
    p = fileparts(which('eeglab'));
    p = [p filesep 'plugins' filesep 'mobilab'];
end
addpath(p);
runmobilab(false);

% JRI 7/3/14 -- original code was finding two "Import Data" menus (the one we want
%   plus one from LIMO plugin. So, need to be more specific: 
%   find the Import Data menu within the top-level File menu

%filemenu = findobj(fig,'Label','File','type','uimenu','parent',fig);
%h = findobj(filemenu,'Label','Import data');

% uimenu( h, 'label', 'From file (.xdf, .xdfz, .drf)', 'callback','allDataStreams = pop_load_file_mobilab;');
% uimenu( h, 'label', 'From folder (concat LSL or DataRiver files)', 'callback','allDataStreams = pop_load_folder_mobilab;');
% uimenu( h, 'label', 'From legacy DataRiver .bdf file', 'callback','allDataStreams = pop_load_bdf;');
% uimenu( h, 'label', 'From MoBILAB folder', 'callback','allDataStreams = pop_load_MoBI;');
% uimenu( h, 'label', 'Into MoBILAB', 'callback','runmobilab;');

h = uimenu( fig, 'label', 'MoBILAB');
uimenu(h,'Label','Import xdf file','Callback','mobilab.loadData(''file'');waitfor(findall(0,''Tag'',''MoBILAB:ImportFile''));mobilab.export2eeglab;');
uimenu(h,'Label','Load MoBI folder','Callback','mobilab.loadData(''mobi'');');
uimenu( h, 'label', 'GUI','CallBack','disp(''runmobilab'');runmobilab(true);');
uimenu( h, 'label', 'MultiStream Browser', 'callback','mobilab.msBrowser();');
uimenu( h, 'label', 'Insert Events', 'callback','mobilab.insertEvents();');
uimenu( h, 'label', 'Export to EEGLAB', 'callback','mobilab.export2eeglab();');
