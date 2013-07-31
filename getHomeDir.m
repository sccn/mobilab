function homeDir = getHomeDir
if ispc
    homeDir= getenv('USERPROFILE');
else
    homeDir = getenv('HOME');
end