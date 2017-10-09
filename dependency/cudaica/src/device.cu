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

#include <stdio.h>
#include <device.h>
#include "config.h"
#include <common.h>
#include <error.h>

#ifdef __cplusplus
extern "C" {
#endif

void printCapabilities(cudaDeviceProp* properties);
device_t gpu;

#ifdef __cplusplus
}
#endif

/*
 * Selects the specified cuda device
 * 
 * deviceNum: number of the desired device
 */ 

error selectDevice(natural deviceNum, natural verbose) {
	cudaGetDeviceCount((int*)&gpu.deviceCount);
	natural numdev = 0;
	cudaDeviceProp deviceProp;
	if (verbose != 0) {
		fprintf(stdout, "=====================\n");
		fprintf(stdout, "List of cuda devices:\n");
		fprintf(stdout, "=====================\n\n");
	}
	for (numdev = 0; numdev < gpu.deviceCount; ++numdev) {
		HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, numdev));
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
			DPRINTF(1,"Device %d does not support CUDA\n", numdev);
		} else {
			DPRINTF(1,"Device %d supports CUDA\n", numdev);
			if (verbose != 0) {
				printf("Device: %d\n", numdev);
				printCapabilities(&deviceProp);
			}
		}
		
	}
	
	fprintf(stdout, "\n\nSelecting device %d", deviceNum);
	HANDLE_ERROR(cudaSetDevice(deviceNum));
	HANDLE_ERROR(cudaDeviceReset());
	HANDLE_ERROR(cudaGetDeviceProperties(&gpu.deviceProp, deviceNum));
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

	gpu.device = deviceNum;
	
	if (gpu.deviceProp.major == 2) {
		gpu.nthreads = 32;
	} else {
		gpu.nthreads = 8;
	}
	
	
	
	
	fprintf(stdout, " Success!\n");
	DPRINTF(1, "Number of threads per GPU in this device: %d\n", gpu.nthreads);
	recalcFreeMem(128);
	if (verbose != 0) {
		fprintf(stdout, "=====================\n\n");
	}
	return SUCCESS;
}

/*
 * Return the maximum number of threads supported by the device
 */ 
natural getMaxThreads() {
	return gpu.nthreads;
}

/*
 * Return the maximum number of blocks supported by the device
 */ 
natural getMaxBlocks() {
	return gpu.deviceProp.maxGridSize[0] > MAX_CUDA_BLOCKS ? MAX_CUDA_BLOCKS : gpu.deviceProp.maxGridSize[0];
}

/*
 * Gets real free mem by trying to malloc.
 * TODO: This is really bad, it doesn't work all the time and it needs an amount of elements by column
 * 
 */
void recalcFreeMem(size_t elemspercolumn) {
	size_t mfree;
	size_t step;
	size_t total;
	HANDLE_ERROR(cudaMemGetInfo(&mfree, &total));
	DPRINTF(1, "Informed free memory %lu - total %lu\n", mfree, total);
	void* ptr;
	step = mfree/2;

	size_t ncols = elemspercolumn;
	size_t nrows = (mfree/sizeof(real))/ncols;
	step = nrows /2;
	size_t pitch;
	while (step > MAX_MEM_THRESHOLD) {
		DPRINTF(2,"Trying with %lu rows and %lu cols ", nrows, ncols);
		if (cudaMallocPitch(&ptr, &pitch, ncols*sizeof(real), nrows) != cudaSuccess) {
			DPRINTF(2, "FAILED!\n");
			nrows -= step;
		} else {
			DPRINTF(2, "OK!\n");
			nrows += step;
			cudaFree(ptr);
		}
		step /= 2;
	}
	nrows -= step*2;
	mfree = ncols * sizeof(real) * nrows;
	DPRINTF(1,"Max linear free mem = %lu (%lu, %lu)\n", mfree, nrows, ncols);
	fprintf(stdout,"Max linear free mem = %lu bytes (aprox %lu samples, %lu channels)\n", mfree, nrows, ncols);
	gpu.currentFreeMem = mfree;
	gpu.neededReservedMem = RESERVED_MEM_BYTES;
	ResetError();
}

/*
 * Returns the amount of memory available to use by data
 * TODO: Fix
 */ 
size_t getFreeMem() {
	DPRINTF(1, "Available Mem %lu (free %lu - res %lu)\n", (gpu.currentFreeMem - gpu.neededReservedMem), gpu.currentFreeMem, gpu.neededReservedMem);
	if (gpu.currentFreeMem < gpu.neededReservedMem) {
		fprintf(stderr, "Not enough mem!\n");
		exit(-1);
	}
	return gpu.currentFreeMem - gpu.neededReservedMem;
}

/*
 * Prints the device capabilities
 */ 
void printCapabilities(cudaDeviceProp* properties) {
	fprintf(stdout, "CUDA Device capabilities:\n");
	fprintf(stdout, "	Name: %s\n", properties->name);
	fprintf(stdout, "	Global Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Mem: %lu\n", properties->totalGlobalMem);
	fprintf(stdout, "	Mem per Block: %lu\n", properties->sharedMemPerBlock);
	fprintf(stdout, "	Regs per Block: %d\n", properties->regsPerBlock);
	fprintf(stdout, "	Warp size: %d\n", properties->warpSize);
	fprintf(stdout, "	Mem pitch: %lu\n", properties->memPitch);
	fprintf(stdout, "	Max Threads per Block: %d\n", properties->maxThreadsPerBlock);
	fprintf(stdout, "	Max Threads Dim: %d x %d x %d\n", 
		properties->maxThreadsDim[0], 
		properties->maxThreadsDim[1], 
		properties->maxThreadsDim[2]);
	fprintf(stdout, "	Max Grid Size: %d x %d x %d\n", 
		properties->maxGridSize[0],
		properties->maxGridSize[1],
		properties->maxGridSize[2]);
	fprintf(stdout, "	Total Const Mem: %lu\n", properties->totalConstMem);
	fprintf(stdout, "	Major: %d\n", properties->major);
	fprintf(stdout, "	Minor: %d\n", properties->minor);
	fprintf(stdout, "	Clock Rate: %d\n", properties->clockRate);
	fprintf(stdout, "	Texture Alignment: %lu\n", properties->textureAlignment);
	fprintf(stdout, "	Device Overlap: %d\n", properties->deviceOverlap);
	fprintf(stdout, "	Multiprocessor Count: %d\n", properties->multiProcessorCount);
	fprintf(stdout, "	Kernel Timeout Enabled: %d\n", properties->kernelExecTimeoutEnabled);
	fprintf(stdout, "	Integrated: %d\n", properties->integrated);
	fprintf(stdout, "	Can Map host mem: %d\n", properties->canMapHostMemory);
	fprintf(stdout, "	Compute mode: %d\n", properties->computeMode);
	fprintf(stdout, "	Concurrent kernels: %d\n", properties->concurrentKernels);
	fprintf(stdout, "	ECC Enabled: %d\n", properties->ECCEnabled);
	fprintf(stdout, "	PCI Bus ID: %d\n", properties->pciBusID);
	fprintf(stdout, "	PCI Device ID: %d\n", properties->pciDeviceID);
	fprintf(stdout, "	TCC Driver: %d\n", properties->tccDriver);
}


/*
 * Saves the data from the device into the host 
 */ 
void saveData(eegdataset_t *set){
	DPRINTF(1, "Saving data from set\n");
	real* datastart = set->data;
	size_t spitch = set->nchannels * sizeof(real);
	DPRINTF(2, "cudaMemcpy2d to %p with pitch %lu from %p with pitch %lu and width %lu bytes and height %u rows\n", datastart,  spitch,  set->devicePointer, set->pitch, spitch, set->nsamples);
	HANDLE_ERROR(cudaMemcpy2D(datastart, spitch, set->devicePointer, set->pitch, spitch, set->nsamples, cudaMemcpyDeviceToHost));
}

/*
 * Frees device memory
 */ 
void freeDeviceMem(eegdataset_t *set) {
	if (set->devicePointer != NULL) {
		DPRINTF(1, "Freeing matrix in device memory\n");
		HANDLE_ERROR(cudaFree(set->devicePointer));
		set->devicePointer = NULL;
	}	
}

/*
 * Loads data to device memory.
 */
error loadToDevice(eegdataset_t *set) {
	DPRINTF(1, "Loading dataset to device memory samples %d channels %d\n", set->nsamples, set->nchannels);
	
	void * ptr = NULL;
	if (set->devicePointer == NULL) {
		cudaError_t err;
		size_t width = set->nchannels * sizeof(real);
		size_t height = set->nsamples;
		DPRINTF(1, "cudaMallocPitch width %lu bytes, height %lu rows\n", width, height);
		size_t pitch;
		err = cudaMallocPitch(&ptr, &pitch, width, height);
		DPRINTF(1, "cudaMallocPitch result %p with pitch %lu\n", ptr, pitch);
		if ( err != cudaSuccess) {
			return ERRORNODEVICEMEM;
		}
		set->pitch = pitch;
		DPRINTF(1, "cudaMallocPitch pointer %p, pitch %lu\n", ptr, pitch);
		set->devicePointer = ptr;
	}

	real* datastart = set->data;
	size_t spitch = set->nchannels * sizeof(real);
	
	DPRINTF(2, "cudaMemcpy2d to %p with pitch %lu from %p with pitch %lu and width %lu bytes and height %u rows\n", set->devicePointer, set->pitch, datastart, spitch, spitch, set->nsamples);
	HANDLE_ERROR(cudaMemcpy2D(set->devicePointer, set->pitch, datastart, spitch, spitch, set->nsamples, cudaMemcpyHostToDevice));
	DPRINTF(1, "Dataset loaded! samples %d channels %d\n", set->nsamples, set->nchannels);
	return SUCCESS;
	
}


error loadWeightsFromDevice(eegdataset_t *set, char* data) {
	DPRINTF(2, "cudaMemcpy2d to %p with pitch %lu from %p with pitch %lu and width %lu bytes and height %u rows\n", data, set->nchannels * 8, set->weights, set->wpitch, set->nchannels * 8, set->nchannels);
	HANDLE_ERROR(cudaMemcpy2D(data, set->nchannels * 8, set->weights, set->wpitch, set->nchannels * 8, set->nchannels, cudaMemcpyDeviceToHost));
	DPRINTF(1, "Weights copied!\n");
	return SUCCESS;
}
error loadSphereFromDevice(eegdataset_t *set, char* data) {
	DPRINTF(2, "cudaMemcpy2d to %p with pitch %lu from %p with pitch %lu and width %lu bytes and height %u rows\n", data, set->nchannels * 8, set->weights, set->wpitch, set->nchannels * 8, set->nchannels);
	HANDLE_ERROR(cudaMemcpy2D(data, set->nchannels * 8, set->sphere, set->spitch, set->nchannels * 8, set->nchannels, cudaMemcpyDeviceToHost));
	DPRINTF(1, "Sphere copied!\n");
	return SUCCESS;
}

error loadDataFromDevice(eegdataset_t *set, char* data) {
	DPRINTF(2, "cudaMemcpy2d to %p with pitch %lu from %p with pitch %lu and width %lu bytes and height %u rows\n", data, set->nchannels * 8, set->weights, set->wpitch, set->nchannels * 8, set->nchannels);
	HANDLE_ERROR(cudaMemcpy2D(data, set->nchannels * 8, set->devicePointer, set->pitch, set->nchannels * 8, set->nsamples, cudaMemcpyDeviceToHost));
	DPRINTF(1, "Data copied!\n");
	return SUCCESS;
}



/*
 * Saves the data from the device into the host 
 */ 
void saveFromDevice(eegdataset_t *set) {
	saveData(set);
}


