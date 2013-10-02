function [values,timestamps,stream] = subsampleChunk(values,timestamps,stream,~)
try %#ok
    if str2double(stream.info.nominal_srate) > 2e4
        values = values(:,1:2:end);
        timestamps = timestamps(1:2:end);
    end
end
