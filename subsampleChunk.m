function [values,timestamps,stream] = subsampleChunk(values,timestamps,stream,~)
try %#ok
    if str2double(stream.info.nominal_srate) > 20e3
        values = values(:,1:2:end);
        timestamps = timestamps(1:2:end);
    end
end