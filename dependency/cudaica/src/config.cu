#include <config.h>
#include <error.h>
#include <stdio.h>

#define xstr_val(val) #val
#define str_val(val) xstr_val(val)
#define PRINTINT(val) printf("\t%s = %d\n", str_val(val), (dataset->config.val));
#define PRINTREAL(val) printf("\t%s = %.16f\n", str_val(val), (dataset->config.val));
#define PRINTBOOL(val) printf("\t%s = %s\n", str_val(val), (dataset->config.val) == 0 ? "off" : "on" );
#define PRINTSTRING(val) printf("\t%s = %s\n", str_val(val), (dataset->config.val));


char* getParam(const char * needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strcmp(needle, haystack[i]) == 0) {
			if (i < count -1) {
				return haystack[i+1];
			}
		}
	}
	return 0;
}


int isParam(const char * needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strcmp(needle, haystack[i]) == 0) {
			return 1;
		}
	}
	return 0;
}



error getReal(char* buffer[], const char* string, int count, real* result) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = (real)atof(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}


error getBool(char* buffer[], const char* string, int count, natural* result) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				if (strcmp(item, "on") == 0) {
					*result = 1;
				} else if (strcmp(item, "off") == 0) {
					*result = 0;
				} else if (strcmp(item, "on\n") == 0) {
					*result = 1;
				} else if (strcmp(item, "off\n") == 0) {
					*result = 0;
				} else if (strcmp(item, "none") == 0) {
					*result = 2;
				} else if (strcmp(item, "none\n") == 0) {
					*result = 2;
				} else {
					return ERRORINVALIDPARAM;
				}
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getVerbose(char* buffer[], const char* string, int count, natural* result) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				if (strcmp(item, "on") == 0) {
					*result = 2;
				} else if (strcmp(item, "off") == 0) {
					*result = 0;
				} else if (strcmp(item, "on\n") == 0) {
					*result = 2;
				} else if (strcmp(item, "off\n") == 0) {
					*result = 0;
				} else if (strcmp(item, "matlab") == 0) {
					*result = 1;
				} else if (strcmp(item, "matlab\n") == 0) {
					*result = 1;
				} else {
					return ERRORINVALIDPARAM;
				}
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getInt(char* buffer[], const char* string, int count, natural* result) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = atoi(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getString(char* buffer[], const char* string, int count, char** result) {
	int i = 0;
	for (i = 0; i < count; i ++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				int len = strlen(item);
				*result = (char*)malloc(len +1);
				memset(*result, 0, len+1);
				strncpy(*result, item, len-1);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

char* programname;

void help() {
	printf("Usage: %s [OPTIONS] -f FILE\n", programname);
	printf("\n");
	printf("\t-h or --help		Print this help\n");
	printf("\n");
	printf("\t-f FILE			Run CUDAICA using the script configuration file FILE\n");
	printf("\n");
	printf("\tCurrent options are:\n");
	printf("\t-d N 			Use device N as cuda GPU\n");
	printf("\t-s FILE			Run in silent redirecting output to FILE and ignoring SIGHUP\n");
	printf("\n");
	printf("The configuration file is a text file where each nonblank line must be a\nparameter and its value separated by a space.\n\n");
	printf("Posible values:\n");
	printf("\tN:\t\tInteger\n");
	printf("\tFILE:\t\tFile path/name\n");
	printf("\tON/OFF:\t\tValue \"on\" or \"off\" (without quotes)\n");
	printf("\tF:\t\tFloat\n");

	printf("\n");
	printf("Required parameters:\n");
	printf("\tDataFile\tFILE\t\tBinary file with the inputa data in\n\t\t\t\t\tsingle precission values (matrix)\n");
	printf("\tchans\t\tN\t\tNumber of data channels (data rows)\n");
	printf("\tframes\t\tN\t\tNumber of data points per epoch (data columns)\n");
	printf("\tepochs\t\tN\t\tNumber of epochs\n");
	printf("\tWeightsOutFile\tFILE\t\tBinary file to store ICA weight matrix (floats)\n");
	printf("\tSphereFile\tFILE\t\tBinary file to store sphering matrix (floats)\n");
	printf("\t\n");

	printf("Optional parameters (with default values):\n");
	printf("\tsphering\tON/OFF\t\tToggles sphering of data (on/off)   {default: on}\n");
	printf("\tbias\t\tON/OFF\t\tPerform bias adjustment (on/off) {default: on}\n");
	printf("\textended\tN\t\tPerform \"extended-ICA\" using tanh() with kurtosis estimation\n\t\t\t\t\tevery N training blocks.If N < 0 fix number\n\t\t\t\t\tof sub-Gaussian components to -N {default|0: off}\n");
	printf("\tpca\t\tN\t\tDecompose a principal component subspace of the data.\n\t\t\t\t\tRetain N PCs. {default|0: all} NOT SUPPORTED (yet)\n");
	printf("\tWeightsInFile\tFILE\t\tStarting ICA weight matrix (chans by ncomps)\n\t\t\t\t\t{default: identity or sphering matrix}\n");
	printf("\tlrate\t\tF\t\tInitial ICA learning rate {default: heuristic ~5e-4}\n");
	printf("\tblocksize\tN\t\tICA block size {default: heuristic fraction of\n\t\t\t\t\tlog data length}\n");
	printf("\tstop\t\tF\t\tStop training when weight-change < this value\n\t\t\t\t\t{default: heuristic ~0.000001}\n");
	printf("\tmaxsteps\tN\t\tMax. number of ICA training steps {default: 128}\n");
	printf("\tposact\t\tON/OFF\t\tMake each component activation net-positive {default: on}\n");
	printf("\tannealstep\tF\t\tAnnealing factor (range  (0,1]) - controls the\n\t\t\t\t\tspeed of convergence.\n\t\t\t\t\t{default: 0.98 for extended, 0.90 for non extended ica}\n");
	printf("\tannealdeg\tN\t\tAngledelta threshold for annealing {default: 60}\n");
	printf("\tmomentum\tF\t\tMomentum gain (range [0,1]) {default: 0}\n");
	printf("\tverbose\tON (2) | MATLAB (1) | OFF (0)\t\tPrint extra information {default: on}\n");
	printf("\tseed\tF\t\tRandom seed {default: time()}\n");
	printf("\n");

	printf("Optional parameters (without default values):\n");
	printf("\tActivationsFile\tFILE\t\tActivations (matrix) of each component (ncomps by points)\n");
	printf("\tBiasFile\tFILE\t\tBias weights vector (ncomps)\n");
	printf("\tSignFile\tFILE\t\tSigns vector designating (-1) sub- and (1)super-Gaussian\n\t\t\t\t\tcomponents (ncomps)\n");
}

void printConfig(eegdataset_t *dataset) {
	fprintf(stdout, "====================================\n");
	fprintf(stdout, "          Configuration\n\n");
	PRINTSTRING(datafile);
	PRINTINT(nchannels);
	PRINTINT(nsamples);
	PRINTSTRING(weightsoutfile);
	PRINTSTRING(sphereoutfile);

	PRINTBOOL(sphering);
	PRINTBOOL(biasing);
	PRINTINT(extblocks);
	PRINTINT(pca);

	PRINTSTRING(weightsinfile);
	PRINTREAL(lrate)
	PRINTINT(block);
	PRINTREAL(nochange)
	PRINTINT(maxsteps);
	PRINTBOOL(posact);
	PRINTREAL(annealstep)
	PRINTREAL(annealdeg)
	PRINTREAL(momentum)

	PRINTINT(nsub);
	PRINTINT(pdfsize);
	PRINTINT(urextblocks);
	PRINTREAL(signsbias);
	PRINTBOOL(extended);

	PRINTINT(verbose);
	PRINTINT(seed);

	PRINTSTRING(activationsfile);
	PRINTSTRING(biasfile);
	PRINTSTRING(signfile);
	fprintf(stdout, "====================================\n\n");
}

error parseConfig(char* filename, eegdataset_t *dataset) {
	fprintf(stdout, "====================================\n");
	fprintf(stdout, " Opening config file \n");
	fprintf(stdout, "====================================\n\n");
	FILE* cfile = fopen(filename, "r");
	if (cfile == NULL) return ERRORINVALIDCONFIG;
	char * buffer =  (char*)malloc(5000);
	int lines = 0;
	while (fgets(buffer, 5000, cfile) != NULL) lines++;
	rewind(cfile);

	char **configs = (char**)malloc(lines * sizeof(char*));
	char *current;
	int i = 0;
	for (i = 0; i < lines; i++) {
		configs[i] = NULL;
		current = fgets(buffer, 5000, cfile);
		if (current != NULL) {
			configs[i] = (char*)malloc((strlen(current)+1) * sizeof(char));
			strcpy(configs[i], current);
		} else {
			configs[i] = NULL;
		}
	}

	if (getString(configs, "DataFile", lines, &dataset->config.datafile) != SUCCESS) {
		fprintf(stderr, "ERROR: Invalid data file\n");
		help();
		exit(0);
	}

	if (getInt(configs, "chans", lines, (natural*) &dataset->config.nchannels) != SUCCESS) {
		fprintf(stderr,"ERROR: Invalid number of channels\n");
		help();
		exit(0);
	}

	/*
	 * frames and epochs are used to calc datalength
	 * Original Infomax uses them for probperm.
	 */
	natural frames = 0;
	natural epochs = 1;
	if (getInt(configs, "frames", lines, &frames) != SUCCESS) {
		fprintf(stderr,"ERROR: Invalid number of frames\n");
		help();
		exit(0);
	}

	if (getInt(configs, "epochs", lines, &epochs) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid number of epochs\n");
	}

	dataset->config.nsamples = frames * epochs;


	if (getString(configs, "WeightsOutFile", lines, &dataset->config.weightsoutfile) != SUCCESS) {
		fprintf(stderr,"ERROR: Invalid weights out file\n");
		help();
		exit(0);
	}

	if (getString(configs, "SphereFile", lines, &dataset->config.sphereoutfile) != SUCCESS) {
		fprintf(stderr,"ERROR: Invalid sphere out file\n");
		help();
		exit(0);
	}

	if (getBool(configs, "sphering", lines, &dataset->config.sphering)  == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid sphering flag\n");
	}

	if (getBool(configs, "bias", lines, &dataset->config.biasing)  == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid bias flag\n");
	}


	if (getInt(configs, "extended", lines, &dataset->config.extblocks)  == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid extended blocks number\n");
	}

	dataset->config.extended = (dataset->config.extblocks != 0);

	if (getInt(configs, "pca", lines, &dataset->config.pca)  == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid pca number\n");
	}

	if (dataset->config.pca != 0) {
		fprintf(stderr,"WARNING: PCA currently not supported, will continue with pca = 0\n");
		dataset->config.pca = 0;
	}

	if (getString(configs, "WeightsInFile", lines, &dataset->config.weightsinfile) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid initial weights file\n");
	}

	if (getString(configs, "ActivationsFile", lines, &dataset->config.activationsfile) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid activations output file\n");
	}

	if (getString(configs, "BiasFile", lines, &dataset->config.biasfile) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid bias output file\n");
	}

	if (getString(configs, "SignFile", lines, &dataset->config.signfile) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid sings output file\n");
	}

	if (getReal(configs, "lrate", lines, &dataset->config.lrate) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid initial lrate\n");
	}

	if (getInt(configs, "blocksize", lines, &dataset->config.block) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid block size\n");
	}

	if (getReal(configs, "stop", lines, &dataset->config.nochange) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid stop change\n");
	}

	if (getInt(configs, "maxsteps", lines, &dataset->config.maxsteps) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid max steps\n");
	}

	if (getBool(configs, "posact", lines, &dataset->config.posact) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid posact flag\n");
	}

	if (getReal(configs, "annealstep", lines, &dataset->config.annealstep) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid anneal step\n");
	}

	if (getReal(configs, "annealdeg", lines, &dataset->config.annealdeg) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid angle for annealing\n");
	}

	if (getReal(configs, "momentum", lines, &dataset->config.momentum) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid momentum\n");
	}

	if (getVerbose(configs, "verbose", lines, &dataset->config.verbose) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid verbose value\n");
	}

	if (getInt(configs, "seed", lines, &dataset->config.seed) == ERRORINVALIDPARAM) {
		fprintf(stderr,"ERROR: Invalid seed value\n");
	}

	DPRINTF(2, "Config file parsed correctly\n");
	printConfig(dataset);
	for (i = 0; i < lines; i++) {
		if (configs[i] != NULL) {
			free(configs[i]);
		}
	}

	free(configs);
	free(buffer);
	return SUCCESS;

}

/*
 * Inits configuration for dataset. Should be called AFTER loading data.
 */
void initDefaultConfig(eegdataset_t *set) {
	set->config.datafile = NULL;
	set->config.nchannels = 0;
	set->config.nsamples = 0;
	set->config.weightsoutfile = NULL;
	set->config.sphereoutfile = NULL;

	set->config.sphering = DEFAULT_SPHERING;
	set->config.biasing = DEFAULT_BIASING;
	set->config.extblocks = DEFAULT_EXTBLOCKS;
	set->config.pca = DEFAULT_PCA;

	set->config.weightsinfile = NULL;
	set->config.lrate = 0.0;
	set->config.block = 0;
	set->config.nochange = DEFAULT_STOP;
	set->config.maxsteps = DEFAULT_MAXSTESPS;
	set->config.posact = DEFAULT_POSACT;
	set->config.annealstep = 0.0f;
	set->config.annealdeg = DEFAULT_ANNEALDEG;
	set->config.momentum = DEFAULT_MOMENTUM;


	set->config.activationsfile = NULL;
	set->config.biasfile = NULL;
	set->config.signfile = NULL;

	set->config.nsub = DEFAULT_NSUB;
	set->config.pdfsize = MAX_PDFSIZE;
	set->config.urextblocks = DEFAULT_UREXTBLOCKS;
	set->config.signsbias = DEFAULT_SIGNSBIAS;
	set->config.extended = DEFAULT_EXTENDED;
	set->config.verbose = DEFAULT_VERBOSE;
	set->config.seed = (int)time(NULL);

	set->nchannels = 0;
	set->nsamples = 0;
	set->devicePointer = NULL;
	set->sphere = NULL;
	set->pitch = 0;
	set->data = NULL;
	set->spitch = 0;
	set->weights = NULL;
	set->wpitch = 0;
	set->h_weights = NULL;
	set->bias = NULL;
	set->signs = NULL;

}

void checkDefaultConfig(eegdataset_t *set) {

	if (set->config.lrate == 0) set->config.lrate = DEFAULT_LRATE(set->config.nchannels);
	if (set->config.block == 0) set->config.block = DEFAULT_BLOCK(set->config.nsamples);
	if (set->config.annealstep == 0.0) {
		set->config.annealstep = (set->config.extended) ? DEFAULT_EXTANNEAL : DEFAULT_ANNEALSTEP;
	}
}
