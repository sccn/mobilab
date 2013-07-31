function a = unixSendMail(mailaddress,subject,messageOrFile)
if nargin < 3, error('No enough input arguments');end

if iscellstr(messageOrFile) messageOrFile = char(messageOrFile(:));end
if ~exist(messageOrFile,'file')
    filename = tempname;
    fid = fopen(filename,'w');
    for it=1:size(messageOrFile,1), fprintf(fid,'%s\n',messageOrFile(it,:));end
    fclose(fid);
    messageOrFile = filename;
end
if iscellstr(mailaddress)
    for it=1:length(mailaddress)
        cmd = ['mail -s "' subject '" ' mailaddress{it} ' < ' messageOrFile];
        a = system(cmd);
    end
else
    cmd = ['mail -s "' subject '" ' mailaddress ' < ' messageOrFile];
    a = system(cmd);
end
delete(messageOrFile)