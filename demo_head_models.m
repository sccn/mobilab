eeglab
try
clear all
close all
clc
catch ME
    disp(ME.message) 
clear all
close all
end

eeglabPath = fileparts(which('eeglab'));
mobilabPath = [eeglabPath filesep 'plugins' filesep 'mobilab'];
addpath(genpath(mobilabPath));

templateFile = [mobilabPath filesep 'data' filesep 'headModelColin27_11997.mat'];
%templateFile = [mobilabPath filesep 'data' filesep 'headModelColin27_3751.mat'];
standardMontage = [mobilabPath filesep 'data' filesep 'standard_1020.elc'];
template = load([mobilabPath filesep 'data' filesep 'headModelColin27_11997.mat']);
individualMontage = [mobilabPath filesep 'data' filesep 'personA_chanlocs.sfp'];
surfFile = 'demo_Colin27_11997.mat';
surfData = template.surfData;
save(surfFile,'surfData');             


%% display template head model: Colin27 + aal atlas + 10/20 montage
[elec,label] = readMontage(standardMontage);
hmObj = headModel('surfaces',surfFile,'atlas',template.atlas,'fiducials',template.fiducials,'channelSpace',elec,'label',label);
plotHeadModel(hmObj);


%% warping individual channel space to template
individualSurfaces = 'demo_Colin27_11997.mat';

hmObj = headModel(individualMontage);
plotMontage(hmObj);

aff = hmObj.warpChannelSpace2Template(templateFile,individualSurfaces,'affine');
plotHeadModel(hmObj);


%% warping template to individual channel space
individualSurfaces = 'warped_demo_Colin27_11997.mat';  % name of the output file
% individualSurfaces = 'warped_demo_Colin27_3751.mat';

hmObj = headModel(individualMontage);
plotMontage(hmObj);

hmObj.warpTemplate2channelSpace(templateFile,individualSurfaces);
plotHeadModel(hmObj);


%% solving the forward problem with OpenMEEG
conductivity = [0.33 0.022 0.33]; % brain and scalp = 0.33 S/m, skull = 0.022 S/m; these conductivies were taken from
                                  % Valdes-Hernandez et al., 2009, Oostendrop TF, 2000; Wendel and Malmivuo, 2006      
normal2surface = true;
hmObj.computeLeadFieldBEM(conductivity,normal2surface);


%% solving the forward problem with NFT
% conductivity = [0.33 0.022 0.33]; % brain and scalp = 0.33 S/m, skull = 0.022 S/m; these conductivies were taken from
%                                   % Valdes-Hernandez et al., 2009, Oostendrop TF, 2000; Wendel and Malmivuo, 2006      
% normal2surface = true;
% hmObj.computeLeadFieldBEM_NFT(conductivity,normal2surface);



%% solving the inverse problem with sLORETA
% K: lead field matrix
% L: Laplaciian operator
% rmIndices: indices to be removed (the Thalamus)
% surfData(3).vertices: source space

[sourceSpace,K,L,rmIndices] = getSourceSpace4PEB(hmObj);

%--
[U,S,V] = svd(K/L,'econ');
Ut = U';
s2 = diag(S).^2;
iLV = L\V;
%--
load(hmObj.surfaces)
t = (0:512-1)/512;
x = cos(2*pi*10*t);
n = size(surfData(3).vertices,1);
ind = setdiff(1:n,rmIndices);


%%
% simulating a Gaussian sources
I = strfind(hmObj.atlas.label,'Precuneus');
I = ~cellfun(@isempty,I);
roi = find(I);
hmObj.atlas.label(roi)
roi = hmObj.atlas.colorTable == roi(end);
roi = surfData(3).vertices(roi,:);
roi = mean(roi);
d = sqrt(sum((surfData(3).vertices(ind,:) - ones(length(ind),1)*roi).^2,2));
gSource = normpdf(d,0,10);
gSource = 0.001*gSource/max(gSource);
gSource = bsxfun(@times,gSource,x);
J = zeros(n,length(t));
Jtrue = J;
Jest = J;
Jtrue(ind,:) = gSource;
Vtrue = K*Jtrue(ind,:);
hmObj.plotOnModel(Jtrue,Vtrue,'True source');

nlambdas = 100;
plotCSD = true;
Jest(ind,:) = inverseSolutionLoreta(Vtrue,K,L,nlambdas,plotCSD,[]);
Vest = K*Jest(ind,:);
hmObj.plotOnModel(Jest,Vest,'Estimated source (Loreta)');


%[Jt,alpha,beta] = dynamicLoreta(Ut,Vtrue,s2,V,iLV,L);

[Jest(ind,:),alpha,beta] = variationalDynLoreta(Ut,Vtrue,s2,iLV,L);
Vest = K*Jest(ind,:);
hmObj.plotOnModel(Jest,Vest,'Estimated source (varDynLoreta)');

[~,loc] = max(Jtrue(:,1))
figure;plot([Jtrue(loc,:)' Jest(loc,:)'])
%%  with noise

snr = 7;
vn1 = (diag((std(Vtrue,[],2)))/snr)*randn(size(Vtrue));

snr = 2;
vn2 = (diag((std(Vtrue,[],2)))/snr)*randn(size(Vtrue));

vn = [vn1(:,1:256) vn2(:,1:256)];
Vnoise = Vtrue + vn;


Jest(ind,:) = inverseSolutionLoreta(Vnoise,K,L,nlambdas,plotCSD,[]);
Vest = K*Jest(ind,:);
hmObj.plotOnModel(Jest,Vest,'Estimated source (Loreta)');
JestL = Jest;


% [Jest(ind,:),alpha,beta] = variationalDynLoreta(Ut,Vnoise,s2,V,iLV,L);

[Jest(ind,1),alpha,beta] = variationalDynLoreta(Ut,Vnoise(:,1),s2,iLV,L);
for it=2:32:size(Vtrue,2)
    try
        [Jest(ind,it:it+31),alpha,beta] = variationalDynLoreta(Ut,Vnoise(:,it:it+31),s2,iLV,L,alpha,beta);
    catch %#ok
        [Jest(ind,it:end),alpha,beta]   = variationalDynLoreta(Ut,Vnoise(:,it:end),s2,iLV,L,alpha,beta);
    end
end

Vest = K*Jest(ind,:);
hmObj.plotOnModel(Jest,Vest,'Estimated source (varDynLoreta)');

[~,loc] = max(Jtrue(:,1));
figure;plot([Jtrue(loc,:)' JestL(loc,:)' Jest(loc,:)'])