function uuid = generateUUID
try uuid =  java.util.UUID.randomUUID;
catch ME
    warning(ME.message);
    disp('Trying something different.');
    if ~ispc, [~,uuid] = system('uuidgen');
    else 
        set = 0:15;
        uuid = [dec2hex(set( randperm(16,8) ))' '-' dec2hex(set( randperm(16,4) ))' '-' dec2hex(set( randperm(16,4) ))' '-' dec2hex(set( randperm(16,12) ))'];
    end
end
uuid = lower(uuid);
if ~ischar(uuid), uuid = char(uuid);end