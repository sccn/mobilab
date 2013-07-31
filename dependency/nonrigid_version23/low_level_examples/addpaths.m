% add all needed function paths
try
    functionname='addpaths.m';
    functiondir=which(functionname);
    functiondir=functiondir(1:end-length(functionname)-19);
    addpath([functiondir '/functions'])
	addpath([functiondir '/functions_affine'])
	addpath([functiondir '/functions_nonrigid'])
	addpath([functiondir '/images'])
catch end