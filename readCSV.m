function [Data,dataTable,dataLabel] = readCSV(csvFile)
fid = fopen(csvFile,'r');
fgetl(fid);
count = 1;

s = fgetl(fid);
ind = find(s==',');
Nc = length(ind);
fclose(fid);
fid = fopen(csvFile,'r');
txt = fgetl(fid);
data = zeros(1,Nc);
while ~feof(fid)
    s = fgetl(fid);
    ind = find(s==',');
    data(count,1) = str2double(s(1:ind(1)-1));
    for it=1:Nc-1
        data(count,it+1) = str2double(s(ind(it)+1:ind(it+1)-1));
    end
    count = count+1;
end
fclose(fid);

ind = find(txt==',');
fieldNames = cell(Nc,1);

fieldNames{1} = txt(1:ind(1)-1);
for it=1:Nc-1
    fieldNames{it+1} = txt(ind(it)+1:ind(it+1)-1);
    fieldNames{it+1}(fieldNames{it+1}==' ') = [];
    fieldNames{it+1}(fieldNames{it+1}=='(') = [];
    fieldNames{it+1}(fieldNames{it+1}==')') = [];
end
for it=1:Nc
    fieldNames{it}(fieldNames{it}==' ') = [];
    fieldNames{it}(fieldNames{it}=='(') = [];
    fieldNames{it}(fieldNames{it}==')') = [];
end

for it=1:length(fieldNames)
    eval(['Data.' fieldNames{it} '=data(:,it);'])
end
dataTable = data;
dataLabel = fieldNames;