% nft_forward_problem_solution() - BEM forward problem solution
%
% Usage:
%   >> nft_forward_problem_solution(subject_name, session_name, of)
%
% Inputs:
%   subject_name - subject name as entered in NFT main window
%   session_name - session name as entered in NFT main window
%   of - output folder where all the outputs are saved
%
% Optional keywords:
%
%   cond : conductivity vector (default = [0.33 0.0132 1.79 0.33])
%   mesh_name : mesh name that will be loaded (default: subject_name)
%   sensor_name:  sensor name (default: [subject_name '_' session_name '.sensors'])
%   ss_name : sourcespace name (default: [subject_name '_sourcespace.dip'])
%   solver : 'bem' or 'fem' (default: 'bem')
%
% Author: Zeynep Akalin Acar, SCCN, 2012

% Copyright (C) 2007 Zeynep Akalin Acar, SCCN, zeynep@sccn.ucsd.edu
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function nft_wrapper(subject_name, session_name, of, varargin)

% default conductivity values
cond = [0.33 0.0132 1.79 0.33];

mesh_name = subject_name;
sensor_name = [subject_name '_' session_name '.sensors'];
ss_name = [subject_name '_sourcespace.dip'];
solver = 'bem';

for i = 1:2:length(varargin) % for each Keyword
      Keyword = varargin{i};
      Value = varargin{i+1};

      if ~isstr(Keyword)
         fprintf('keywords must be strings')
         return
      end

      if strcmp(Keyword,'cond')
         if isstr(Value)
            fprintf('cond must be vector');
            return
         else
            cond = Value;
         end
      elseif strcmp(Keyword,'mesh_name')
         if ~isstr(Value)
            fprintf('mesh_name must be a string');
            return
         else
             mesh_name = Value;
         end
      elseif strcmp(Keyword,'sensor_name')
         if ~isstr(Value)
            fprintf('sensor_name must be a string');
            return
         else
             sensor_name = Value;
         end
      elseif strcmp(Keyword,'ss_name')
         if ~isstr(Value)
            fprintf('ss_name must be a string');
            return
         else
             ss_name = Value;
         end
      end
      if strcmp(Keyword,'solver')
         if ~isstr(Value)
            fprintf('solver must be a string');
            return
         else
             ss_name = Value;
         end
      end
end

current_folder = pwd;
lof = length(of);
if of(lof) ~= filesep
    of(lof+1) = filesep;
end
cd(of)


% load the BEM mesh
if solver == 'bem'
    mesh = bem_load_mesh(mesh_name);
    vol = mesh2volstr(mesh_name);
elseif solver == 'fem'
    mesh = bem_load_mesh(subject_name);
    vol = mesh2volstr(subject_name);
end

if (mesh.num_boundaries == 3 && length(cond) == 4)
    cond = [cond(1) cond(2) cond(4)];
end

vol.cond = cond;


if solver == 'bem'
    % generate model and model matrices
    model = bem_create_model(subject_name, mesh, cond, 3);
    bem_generate_eeg_matrices(model);
    
    % save model
    msave.name = model.name;
    msave.mesh_name = model.mesh.name;
    msave.cond = model.cond;
    msave.mod = model.mod;
    save([model.name '.model'], '-STRUCT', 'msave')
    
    % load sensors
    sens = load(sensor_name, '-mat');
    Smatrix = bem_smatrix_from_coordinates(mesh, sens.pnt);
    
    % create session
    session = bem_create_session(session_name, model, Smatrix);
    session = bem_generate_eeg_transfer_matrix(session);
    
    %save session
    ssave.name = session.name;
    ssave.model_name = session.model.name;
    ssave.Smatrix = Smatrix;
    ssave.sens = sens;
    save([session.name '.session'], '-STRUCT', 'ssave');
    
    % load source space
    ss = load(ss_name);
    % calculate LFM
    [LFM, session] = bem_solve_lfm_eeg(session, ss);
    vol.type = 'metubem';

elseif solver == 'fem'
    % set conductivity values
    sens = load(sensor_name, '-mat'); % sensor locations
    ss = load(ss_name); % sourcespace
    
    vol2 = metufem_set_mesh(mesh_name);
    
    sens.type = 'eeg';
    sens = metufem_calcrf(vol2, sens, of, cond);
    
    session.name = session_name;
    session.cond = cond;
    session.sens = sens;
    session.type = 'fem';
    session.vol = metufem_set_mesh([of mesh_name]);
    
    % save session
    msave.session = session;
    msave.mesh_name = mesh_name;
    msave.mesh_path = of;
    save([session.name '.session'], '-STRUCT', 'msave')
    
    metufem('setup', mesh_name, 'sensors.dat', '');
    metufem('setrf', session.sens.rf);
    
    LFM = metufem('pot', ss');
    vol.type = 'metufem';
end

save([subject_name '_vol.mat'],'vol');
save([session_name '_LFM.mat'],'LFM');
clear LFM
cd(current_folder)
    
    
