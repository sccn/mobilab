function [data,timestamp,event,number_of_channels,precision] = read_stream(fid)
timestamp  = fread(fid,1,'int32');
event      = fread(fid,1,'int32');
number_of_channels = fread(fid,1,'int32');
item_size  = fread(fid,1,'int32');
precision  = sprintf('int%i',8*item_size);
try
    data = fread(fid,number_of_channels,precision);
catch ME
    ME.rethrow;
end

