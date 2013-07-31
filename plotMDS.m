function [h1,h2] = plotMDS(Y,D,labels,colormap)
if nargin < 3
    labels = cell(size(Y,1),1);
    for it=1:size(Y,1), labels{it} = num2str(it);end
end
if nargin < 4, colormap = 'hsv';end
ntrials = length(labels);
uLabels = unique(labels);
color = eval([colormap '(length(uLabels));']);
warning off %#ok
h1 = figure;hold on;
for jt=1:ntrials
    I = ismember(uLabels,labels(jt));
    plot3(Y(jt,1),Y(jt,2),Y(jt,3),'o','Color',color(I,:),'MarkerFaceColor',color(I,:),'MarkerSize',25,'MarkerEdgeColor','k')
    text('Position',Y(jt,1:3),'String',labels{jt},'Color','k','FontWeight','bold')
end
title('MDS');
xlabel('X'),ylabel('Y');zlabel('Z');
grid on

h2 = figure;
imagesc(abs(D));
title('Similarity matrix')
warning on %#ok
end