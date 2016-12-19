/*	
 *	Copyright (C) 2011, Federico Raimondo (fraimondo@dc.uba.ar)
 *	
 *	This file is part of Cudaica.
 *
 *  Cudaica is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  Cudaica is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with Cudaica.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <loader.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <error.h>
#include <preprocess.h>
#include <common.h>
#include <device.h>

#include <errno.h>

/*
 * Loads data from file into host memory
 * 
 * src: data file
 * rows: number of rows
 * cols: number of cols
 * dst: return variable with the data on memory
 */
error dataload(char* src, natural rows, natural cols, real** dst) {
	int fd = open(src, O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "Error opening data file (%d) %s - %s\n", errno, strerror(errno), src);
		return ERRORNOFILE;
	}
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		fprintf(stderr, "Error stating data file %s\n", src);
		return ERRORNOFILE;
	}
	void *mmaping = mmap (0, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	size_t map_size = sb.st_size;
	float *matriz = (float*)mmaping;
	if (matriz == MAP_FAILED) {
		fprintf(stderr, "Error mapping data file %s\n", src);
		return ERRORNOFILE;
	}
	DPRINTF(2, "Matrix mapped at %p\n", matriz);
	DPRINTF(2, "dataload from %p (%d x %d) to dataset\n", matriz, rows, cols);
	size_t size = sizeof(real) * rows * cols;
	real* newdata = (real*)malloc(size);
	for (int i = 0; i < rows*cols; i++) {
		newdata[i] = (real)matriz[i];
	}
	*dst = newdata;
	if (close(fd) == -1) {
		fprintf(stderr, "Error closing data file %d\n",fd);
	}
	if (munmap (mmaping, map_size) == -1) {
		fprintf(stderr, "Error unmapping data file at %p with size %lu\n", mmaping, map_size);
	}
	return SUCCESS;

}


/*
 * Prints dataset info
 */ 
void printDatasetInfo(eegdataset_t* dataset) {
	fprintf(stdout, "====================================\n");
	fprintf(stdout, "            Dataset info            \n");
	fprintf(stdout, "                                    \n");
	fprintf(stdout, " Channels: %d\n", dataset->nchannels);	
	fprintf(stdout, " Samples: %d\n", dataset->nsamples);	
	fprintf(stdout, " Elements: %d\n", dataset->nsamples * dataset->nchannels);
	fprintf(stdout, " Device pointer: %p\n", dataset->devicePointer);
	fprintf(stdout, " Pitch: %lu\n", dataset->pitch);
	fprintf(stdout, " Data: %p\n", dataset->data);
	fprintf(stdout, " Sphere pointer: %p\n", dataset->sphere);
	fprintf(stdout, " Sphere Pitch: %lu\n", dataset->spitch);
	fprintf(stdout, " Weights pointer: %p\n", dataset->weights);
	fprintf(stdout, " Weights Pitch: %lu\n", dataset->wpitch);
	fprintf(stdout, "====================================\n");

}

/*
 * Loads data and weights from the files in the dataset
 */ 
error loadEEG(eegdataset_t *dataset) {
	int nchannels = dataset->config.nchannels;
	int nsamples = dataset->config.nsamples;
	dataset->nchannels = 0;
	dataset->nsamples = 0;
	dataset->devicePointer = NULL;
	dataset->sphere = NULL;
	dataset->pitch = 0;
	dataset->data = NULL;
	dataset->weights = NULL;
	dataset->bias = NULL;
	dataset->signs = NULL;
	dataset->wpitch = 0;
	dataset->spitch = 0;
	
	/*
	 * Load data file
	 */ 
	error err = dataload(dataset->config.datafile, nsamples, nchannels, &dataset->data);
	if (err != SUCCESS) {
		fprintf(stderr, "Error loading data file %s\n", dataset->config.datafile);
		return err;
	}
	
	/*
	 * Load weights file
	 */ 
	if (dataset->config.weightsinfile != NULL) {
		err = dataload(dataset->config.weightsinfile, nchannels, nchannels, &dataset->h_weights);
		if (err != SUCCESS) {
			fprintf(stderr, "Error loading weights file %s\n", dataset->config.weightsinfile);
			return err;
		}
	} else {
		dataset->h_weights = NULL;
	}
	
	dataset->nsamples = dataset->config.nsamples;
	dataset->nchannels = dataset->config.nchannels;
	
	return SUCCESS;
}


/*
 * Saves the data from the dataset into the corresponding files
 */ 
error saveEEG(eegdataset_t *dataset) {
	DPRINTF(1, "Saving dataset results\n");
	if (dataset->weights != NULL) {
		DPRINTF(1, "Saving weights results in %s\n", dataset->config.weightsoutfile);
		dev_matwrite(dataset->config.weightsoutfile, dataset->nchannels, dataset->nchannels, dataset->weights, dataset->wpitch);
	}
	if (dataset->sphere != NULL) {
		DPRINTF(1, "Saving sphere results in %s\n", dataset->config.sphereoutfile);
		dev_matwrite(dataset->config.sphereoutfile, dataset->nchannels, dataset->nchannels, dataset->sphere, dataset->spitch);
	}
	if (dataset->bias != NULL && dataset->config.biasfile != NULL) {
		DPRINTF(1, "Saving bias results in %s\n", dataset->config.biasfile);
		dev_matwrite(dataset->config.biasfile, 1, dataset->nchannels, dataset->bias, dataset->nchannels * sizeof(real));
	}
	
	if (dataset->signs != NULL && dataset->config.signfile != NULL) {
		DPRINTF(1, "Saving signs results in %s\n", dataset->config.signfile);
		dev_matwriteInt(dataset->config.signfile, 1, dataset->nchannels, dataset->signs, dataset->nchannels * sizeof(real));
	}
	
	
	return SUCCESS;
}


/*
 * Deletes the dataset
 */  
error freeEEG(eegdataset_t *dataset) {
	if (dataset->h_weights != NULL) free(dataset->h_weights);
	if (dataset->weights != NULL) HANDLE_ERROR(cudaFree(dataset->weights));
	if (dataset->sphere != NULL) HANDLE_ERROR(cudaFree(dataset->sphere));
	if (dataset->signs != NULL) HANDLE_ERROR(cudaFree(dataset->signs));
	if (dataset->bias != NULL) HANDLE_ERROR(cudaFree(dataset->bias));
	if (dataset->data != NULL) free(dataset->data);
	free(dataset);
	return SUCCESS;
}
