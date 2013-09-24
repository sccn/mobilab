function allocateFile(filename,precision,dim)
tmp = eval([precision '(1)']); %#ok
tmp = whos('tmp');
tmp.bytes;
maxSize = prod(dim);
sizeFile = maxSize*tmp.bytes;
cmd = ['fallocate -l '  int2str(sizeFile) ' ' filename];
out = system(cmd);
if out 
    fid = fopen(filename,'w');
    if fid<=0, error('Cannot create a new file. Check you you hdd space.');end;
    writeThis = zeros(dim(1),1);
    for it=1:dim(2), fwrite(fid,writeThis,precision);end
    fclose(fid);
end

