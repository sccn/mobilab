function cell2textfile(file,linesInCell)
fid = fopen(file,'w');
for it=1:length(linesInCell), fprintf(fid,'%s\n',linesInCell{it});end
fclose(fid);
end