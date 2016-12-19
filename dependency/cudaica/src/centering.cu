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
#include <centering.h>
#include <error.h>
#include <common.h>
#include <device.h>


/*
 * Magic:
 * When needed, blocks may increase this variable.
 * When this reaches gridDim.x, then last block has finished.
 */
#ifndef BLOCKS_FINISHED
#define BLOCKS_FINISHED
__device__ unsigned int blocksFinished;
__shared__ bool isLastBlockFinished; //If true, then last block has finished.
#endif

/*
 * Computes the sums for each channel and then divides by the number of samples.
 * sums = sum(channels(data))/samples
 *
 * Should be launched with N blocks of channels threads
 *
 * data: matrix
 * channels: number of channels
 * samples: number of samples
 * pitch: matrix row size in bytes
 * sums: output matrix (must be at least blocks by channels)
 * sumspitch: sums row size in bytes
 */
__global__ void getMean(real* data, natural channels, natural samples, size_t pitch, real* sums, size_t sumspitch) {
	float sum = 0.0;
	size_t colwidth = pitch/sizeof(real);
	size_t sumcolwidth = sumspitch/sizeof(real);
	int count = samples / gridDim.x;	// Process a fraction of a column
	int i = count * blockIdx.x;			// Starts when it should
	int end = count * (blockIdx.x + 1);	// Ends when the next starts
	if (blockIdx.x == gridDim.x -1) {	// If its the last, finish
		end = samples;
	}
	for (; i < end; i++) {
		sum += data[(i*colwidth) + threadIdx.x];
	}
	sums[blockIdx.x * sumcolwidth + threadIdx.x] = sum;

	if (threadIdx.x == 0) {
		natural value = atomicInc(&blocksFinished, gridDim.x);
		isLastBlockFinished = (value == gridDim.x-1);
	}

	__syncthreads();
	if (isLastBlockFinished) {
		sum = 0.0;
		for (i = 0; i < gridDim.x; i++) {
			sum += sums[threadIdx.x + i * sumcolwidth];
		}
		sums[threadIdx.x] = sum/samples;
		if (threadIdx.x == 0) {
			blocksFinished = 0;
		}
	}
}

/*
 * Centers data by substracting the mean value from means vector
 * data = data - mean
 *
 * Should be launched with N blocks of channels threads
 *
 * data: matrix
 * channels: number of channels
 * samples: number of samples
 * pitch: matrix row size in bytes
 * means: vector of means
 */
__global__ void subMean(real* data, natural channels, natural samples, size_t pitch, real* means) {
	int colwidth = pitch/sizeof(real);
	real mean = means[threadIdx.x];
	int count = samples / gridDim.x;		// Process a fraction of a column
	int i = count * blockIdx.x;				// Starts when it should
	int end = count * (blockIdx.x + 1);		// Ends when the next starts
	if (blockIdx.x == gridDim.x -1) {		// If its the last, finish
		end = samples;
	}
	for (; i < end; i++) {
		data[(i*colwidth) + threadIdx.x] -= mean;
	}
}

/*
 * Centers a dataset.
 *
 * Computes the mean of each channel and substracts it.
 *
 * set: the dataset to be centered
 */
void centerData(eegdataset_t *set) {
	DPRINTF(1, "Centering dataset channels %d, samples %d\n", set->nchannels, set->nsamples);
	real *sums;
	size_t sumspitch;
	natural nthreads = set->nchannels;
	natural nblocks = getMaxBlocks();
	DPRINTF(2, "cudaMallocPitch %lu x %d for sums\n", set->nchannels * sizeof(real), nblocks);
	HANDLE_ERROR(cudaMallocPitch(&sums, &sumspitch, set->nchannels * sizeof(real), nblocks));
	real *data = (real*)set->devicePointer;

	DPRINTF(2, "Getting channels mean\n");
	getMean<<<nblocks,nthreads>>>(data, set->nchannels, set->nsamples, set->pitch, sums, sumspitch);
	CHECK_ERROR();

	DPRINTF(2, "Substracting mean to data\n");
	subMean<<<nblocks,nthreads>>>(data, set->nchannels, set->nsamples, set->pitch, sums);
	CHECK_ERROR();
	DPRINTF(1, "Centering dataset finished! channels %d, samples %d\n", set->nchannels, set->nsamples);
	HANDLE_ERROR(cudaFree(sums));

}
