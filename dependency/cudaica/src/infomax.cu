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

#include <infomax.h>
#include <stdio.h>
#include <error.h>
#include <common.h>
#include <device.h>
#include <r250.h>
#include <cublas_v2.h>

#define ERASE_STRING "\r"

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

//extern "C" {
__device__ unsigned int weights_blowup;
//}
__device__ real sums[MAX_CHANNELS*MAX_CHANNELS];



/*************************
 * ICA Initializing steps
 *************************/




 /*
  * Initializes momentum needed variables
  * Should be launched with channels blocks x channels threads
  */
 __global__ void initprvweights(real * weights, size_t wpitch, real * prevweights, size_t prevweightspitch, real * prevwtschange, size_t prevwtschangepitch) {

	size_t wcolwidth = wpitch/sizeof(real);
	size_t prvwtscolwidth = prevweightspitch/sizeof(real);
	size_t prvwtschgcolwidth = prevwtschangepitch/sizeof(real);
	prevweights[threadIdx.x + blockIdx.x * prvwtscolwidth] = weights[threadIdx.x + blockIdx.x * wcolwidth];
	prevwtschange[threadIdx.x + blockIdx.x * prvwtschgcolwidth ] = 0.0;

}

 /*
  * Initializes needed variables
  * Should be launched with channels blocks x channels threads
  */
__global__ void initChxChMatrixes(real * weights, real* startweights, real* oldweights, size_t wpitch, size_t startwpitch, size_t oldwpitch, int initweights) {

	size_t wcolwidth = wpitch/sizeof(real);
	size_t startwcolwidth = startwpitch/sizeof(real);
	size_t oldwcolwidth = oldwpitch/sizeof(real);
	real value = 0;
	if (initweights) {
		value = (threadIdx.x == blockIdx.x ? 1.0 : 0.0);
		weights[threadIdx.x + blockIdx.x * wcolwidth] = value;
	} else {
		value = weights[threadIdx.x + blockIdx.x * wcolwidth];
	}
	startweights[threadIdx.x + blockIdx.x * startwcolwidth] = value;
	oldweights[threadIdx.x + blockIdx.x * oldwcolwidth ] = value;

}

/*
 * Initializes all channel's sized vectors.
 *
 * Should be launched with b blocks and t threads where n*t >= channels
 */
__global__ void initChannelsVectors(real * bias, natural biasing, int* signs, real*oldkk, natural extended, natural nsub, natural channels) {
	if (biasing) {
		bias[threadIdx.x] = 0.0;
	}
	if (extended) {
		signs[threadIdx.x] = (threadIdx.x < nsub) ? 1 : 0;
		oldkk[threadIdx.x] = 0.0;
	}
}





/*
 * Performs:
 * u = weigths * data randomly permuted;
 * if (biasing) u += bias;
 * if (! extended)
 * 	y = -tanh(u/2)
 * else
 * 	y = tanh(u)
 *
 *
 * Should be launched with block blocks and channels threads
 */
extern __shared__ real sample[];
__global__ void step1(
	natural channels,
	natural extended,
	natural t,
	real *weights,
	real *data,
	real *u,
	real *y,
	natural * dataperm,
	natural biasing,
	real * bias,
	size_t wpi,
	size_t dpi,
	size_t upi,
	size_t ypi
	) {
	int i = 0;
	size_t colwidth = dpi/sizeof(real);
	size_t wcolwidth = wpi/sizeof(real);
	size_t ucolwidth = upi/sizeof(real);
	size_t ycolwidth = ypi/sizeof(real);
	natural swap = dataperm[t + blockIdx.x];
	//TODO: This makes	four 32B accesses instead of two 64B access. Why?
	real thissample = data[swap * colwidth + threadIdx.x];	//Copy data into local memory.
	sample[threadIdx.x] = thissample;
	__syncthreads();

	real value = 0.0;
	for (i = 0; i < channels; i++) {
		value += weights[threadIdx.x + wcolwidth * i] * sample[i];
	}
	if (biasing) {
		value += bias[threadIdx.x];
	}

	u[blockIdx.x * ucolwidth + threadIdx.x] = value;
	if (! extended) {
		y[blockIdx.x * ycolwidth + threadIdx.x] = -tanh(value/2.0);
	} else {
		y[blockIdx.x * ycolwidth + threadIdx.x] = tanh(value);
	}
}

/*
 * Performs:
 * if !extended
 * 	bsum = sum(channels(y))
 * else
 * 	bsum = -2*sum(channels(y))
 *  if (signs[i] != -) -y[i];
 *
 * Should be launched with MAX_MULTIPROCESSORS block of channel threads
 */

__global__ void step2(
	natural block,
	natural extended,
	natural channels,
	int *signs,
	real *y,
	real *bsum,
	size_t ypitch,
	int biasing
) {
	size_t ycolwidth = ypitch/sizeof(real);
	natural count = block / gridDim.x;	//Each block iterates count samples
	natural start = blockIdx.x * count;	//Each block starts where previous finished;
	natural end = ((blockIdx.x + 1) * count) -1; //Each block ends one before the next
	if ((blockIdx.x +1) == gridDim.x) {
		end = block -1; //If last block, it ends
	}
	int i = start;
	real sum = 0.0f;
	natural invert = 0;
	if (extended) {
		invert = signs[threadIdx.x];
	}
	for (i = start; i <= end; i++) {
		if (biasing) sum += y[i * ycolwidth + threadIdx.x];
		if (invert) y[i * ycolwidth + threadIdx.x] = -y[i * ycolwidth + threadIdx.x];
	}
	if (biasing) sums[blockIdx.x * MAX_CHANNELS + threadIdx.x] = sum;

	if (biasing) {
		if (threadIdx.x == 0) {
			natural value = atomicInc(&blocksFinished, gridDim.x);
			isLastBlockFinished = (value == gridDim.x-1);
		}
		__syncthreads();
		if (isLastBlockFinished) {
			sum = 0.0f;
			for (i = 0; i < gridDim.x; i++) {
				sum += sums[threadIdx.x + i * MAX_CHANNELS];
			}
			if (!extended) {
				bsum[threadIdx.x] = sum;
			} else {
				bsum[threadIdx.x] = -2*sum;
			}
			if (threadIdx.x == 0) {
				blocksFinished = 0;
			}
		}
	}
}

extern __shared__ real uchannel[];

/*
 * Step 3
 * Computes:
 * if (!extended)
 * 	yu = y * u'
 * else
 * 	yu = -y*u' - (u * u')
 * fi
 *  yu =+ I(BLOCK);
 *
 * Should be launched with channels block of channels threads
 */
__global__ void step3(
	natural extended,
	natural channels,
	natural block,
	real *u,
	real *y,
	real *yu,
	size_t upitch,
	size_t ypitch,
	size_t yupitch
	) {

	int i = 0;
	real sum = 0.0f;
	size_t ucolwidth = upitch/sizeof(real);
	size_t ycolwidth = ypitch/sizeof(real);
	size_t yucolwidth = yupitch/sizeof(real);
	int start = threadIdx.x;
	int end = block;

	/*
	 * Copy channel into shared memory
	 * for 32 bits broadcast access
	 */
	for (i = start; i < end; i += blockDim.x) {
		uchannel[i] = u[blockIdx.x + ucolwidth * i];
	}
	__syncthreads();

	if (!extended) {
		for (i = 0; i < block; i++) {
			sum += uchannel[i] * y[threadIdx.x + ycolwidth *i];
		}
		if (threadIdx.x == blockIdx.x) {
			sum += block;
		}
		yu[threadIdx.x + yucolwidth * blockIdx.x] = sum; //stores again in column major order
	} else {
		for (i = 0; i < block; i++) {
			sum += uchannel[i] * u[threadIdx.x + ucolwidth *i];
		}

		real sum2 = 0.0;
		for (i = 0; i < block; i++) {
			sum2 += -y[threadIdx.x + ycolwidth * i] * uchannel[i];
		}
		if (threadIdx.x == blockIdx.x) {
			sum2 += block;
		}
		yu[threadIdx.x + yucolwidth * blockIdx.x] = sum2-sum; //stores again in column major order
	}


}

/*
 * Step 4
 * Computes:
 *
 * weigths = lrate * yu * weights + weights
 * if (biasing) bias = lrate * bsum + bias
 * if (momentum > 0.0) {
 * 		weights = weights + momentum  * prevweights
 * 		prevwtchange = weights - prevweights
 * 		prevweights = weights
 * }
 *
 * Should be launched with channel blocks and channel threads
 */
 extern __shared__ real wchannel[];
 __global__ void step4(
	real lrate,
	natural channels,
	natural biasing,
	real* bsum,
	real* bias,
	real* yu,
	real* weights,
	size_t yupitch,
	size_t wpitch,
	real * prevweights,
	size_t prevweightspitch,
	real * prevwtchange,
	size_t prevwtchangepitch,
	real v_momentum
 ) {
	size_t wcolwidth = wpitch/sizeof(real);
	size_t yucolwidth = yupitch/sizeof(real);
	size_t pwcolwidth = prevweightspitch/sizeof(real);
	size_t pwchangecolwidth = prevwtchangepitch/sizeof(real);

	int i = 0;

   /*
	* Copy weigths column into shared memory
	* for 32 bits broadcast access
	*/
	wchannel[threadIdx.x] = weights[blockIdx.x * wcolwidth + threadIdx.x];
	__syncthreads();
	real sum = 0.0;
	for (i = 0; i < channels; i++) {
		sum += yu[threadIdx.x + yucolwidth * i] * wchannel[i];
	}

	sum *= lrate;
	sum += wchannel[threadIdx.x];

	weights[threadIdx.x + blockIdx.x * wcolwidth] = sum;

	if (v_momentum > 0.0) {
		weights[threadIdx.x + blockIdx.x * wcolwidth] = weights[threadIdx.x + blockIdx.x * wcolwidth] * v_momentum + prevweights[threadIdx.x + blockIdx.x * pwcolwidth];
		prevwtchange[threadIdx.x + blockIdx.x * pwchangecolwidth] = weights[threadIdx.x + blockIdx.x * wcolwidth] - prevweights[threadIdx.x + blockIdx.x * pwcolwidth];
		prevweights[threadIdx.x + blockIdx.x * pwcolwidth] = weights[threadIdx.x + blockIdx.x * wcolwidth];
		sum = weights[threadIdx.x + blockIdx.x * wcolwidth];
	}

	if (sum > MAX_WEIGHT) weights_blowup = 1;

	/*
	 * Only block 0 calculates
	 * if (biasing) bias = lrate * bsum + bias
	 */
	if (blockIdx.x == 0) {
		if (biasing) {
			bias[threadIdx.x] += lrate * bsum[threadIdx.x];
		}
	}
 }


/*
 * PDF
 * Computes:
 * tmp = weigths * sample
 * kk[i] = (sum(tmp^4) * pdfsize/ sum(tmp^2)^2) -3
 *
 * distintos = #(signs != oldsigns)
 * signs = kk[i] < - signsbias
 *
 * Should be launched with pdfsize blocks of channel threads
 */
__device__ unsigned int distintos;
__global__ void pdf(
	real* data,
	natural channels,
	real * weights,
	natural * pdfperm,
	natural pdfsize,
	natural piter,
	int * signs,
	//int * oldsigns,
	real signsbias,
	real* kk,
	size_t dpitch,
	size_t wpitch,
	size_t kkpitch,
	real * old_kk,
	real extmomentum
) {
	real sum = 0.0;
	real sum2 = 0.0;
	int i = 0;
	size_t dcolwidth = dpitch / sizeof(real);
	size_t wcolwidth = wpitch / sizeof(real);
	size_t kkcolwidth = kkpitch /sizeof(real);

	int swap = blockIdx.x;
		if (pdfperm) {
			swap = pdfperm[piter * pdfsize + blockIdx.x];
		}
		sample[threadIdx.x] = data[threadIdx.x + swap * dcolwidth];


		__syncthreads();


		for (i = 0; i < channels; i++) {
			sum += weights[threadIdx.x + i * wcolwidth] * sample[i];
		}

		sum = sum * sum;
		kk[blockIdx.x * kkcolwidth + threadIdx.x] = sum;
		sum = sum * sum;
		kk[(pdfsize + blockIdx.x) * kkcolwidth + threadIdx.x] = sum;

		if (threadIdx.x == 0) {
			natural value = atomicInc(&blocksFinished, gridDim.x);
			isLastBlockFinished = (value == gridDim.x-1);
		}
		__syncthreads();
		if (isLastBlockFinished) {
			sum = 0.0;
			for (i = 0; i < pdfsize; i++) {
				sum += kk[threadIdx.x + i * kkcolwidth];
				sum2 += kk[threadIdx.x + (i + pdfsize) * kkcolwidth];
			}
			sum2 = (sum2 * pdfsize / (sum * sum)) - 3.0;
			if (extmomentum > 0.0) {
				real okk = old_kk[threadIdx.x];
				sum2 = (1.0 - extmomentum) * sum2 + extmomentum * okk;

			}
			int sign = (sum2 < (-signsbias));
			if (sign != signs[threadIdx.x]) {
				atomicInc(&distintos, gridDim.x);
			}
			signs[threadIdx.x] = sign;

			kk[threadIdx.x] = sum2;
			old_kk[threadIdx.x] = sum2;
			if (threadIdx.x == 0) {
				blocksFinished = 0;
			}
		}
}

/*
 * calcDelta
 * Calculates DELTA from WEIGHTS and OLDWEIGHTS
 * Should be launched with channel blocks of channel threads
 */
__global__ void calcDelta(
	natural channels,
	real * delta,
	real * weights,
	real * oldweights,
	size_t deltapitch,
	size_t wpitch,
	size_t oldwpitch
	) {

	size_t dcolwidth = deltapitch / sizeof(real);
	size_t wcolwidth = wpitch / sizeof(real);
	size_t oldwcolwidth = oldwpitch /sizeof(real);
	delta[threadIdx.x + blockIdx.x * dcolwidth] = weights[threadIdx.x + blockIdx.x * wcolwidth] - oldweights[threadIdx.x + blockIdx.x * oldwcolwidth];
}

/*
 * dotProductSame
 * Calulates sum(elem(matrix)^2);
 * Should be launched with 1 block of channel threads
 */
extern __shared__ real matrixsums[];
__device__ real dotResult;
__global__ void dotProductSame(
	natural channels,
	real* matrix,
	size_t pitch
	) {
	size_t colwidth = pitch/sizeof(real);
	int i = 0;
	real sum = 0.0;
	real elem = 0.0;
	for (i = 0; i < channels; i++) {
		elem = matrix[threadIdx.x + i * colwidth];
		sum += elem * elem;
	}
	matrixsums[threadIdx.x] = sum;
	__syncthreads();
	if (threadIdx.x == 0) {
		sum = 0;
		for (i = 0; i < channels; i++) {
			sum += matrixsums[i];
		}
		dotResult = sum;
	}
}

/*
 * dotProduct
 * Calulates A * B as vectors;
 * Should be launched with 1 block of channel threads
 */
__global__ void dotProduct(
	natural channels,
	real* matrixa,
	real * matrixb,
	size_t apitch,
	size_t bpitch
	) {
	size_t acolwidth = apitch/sizeof(real);
	size_t bcolwidth = bpitch/sizeof(real);
	int i = 0;
	real sum = 0.0;
	for (i = 0; i < channels; i++) {
		sum += matrixa[threadIdx.x + i * acolwidth] * matrixb[threadIdx.x + i * bcolwidth];
	}
	matrixsums[threadIdx.x] = sum;
	__syncthreads();
	if (threadIdx.x == 0) {
		sum = 0;
		for (i = 0; i < channels; i++) {
			sum += matrixsums[i];
		}
		dotResult = sum;
	}
}

void initperm(size_t samples, natural* perm, natural *hostperm) {
	natural i = 0;
	for (i = 0; i < samples; i++) {
		hostperm[i] = i;
	}
	DPRINTF(1, "Using permutations\n");
	natural temp;
	natural swap;
	for (i = samples; i > 0; i--) {
		swap = r250() %i;

		if ((i-1) != swap) {
			temp = hostperm[swap];
			hostperm[swap] = hostperm[i-1];
			hostperm[i-1] = temp;
		}
	}
	HANDLE_ERROR(cudaMemcpy(perm, hostperm, samples*sizeof(natural), cudaMemcpyHostToDevice));
}


__global__ void addEye(real *y, size_t inc, real value) {
	if (threadIdx.x == blockIdx.x) {
		y[threadIdx.x + blockIdx.x * inc] += value;
	}
}

__global__ void matScale(real *a, natural arowsize, real *b, natural browsize, real *c, natural crowsize, real scale) {
	c[threadIdx.x + blockIdx.x * crowsize] = a[threadIdx.x + blockIdx.x * arowsize] + scale * b[threadIdx.x + blockIdx.x * browsize];
}

__global__ void getBlowup(real *weights, natural wrowsize) {
	if (weights[threadIdx.x + blockIdx.x * wrowsize] > MAX_WEIGHT) atomicInc(&weights_blowup, threadIdx.x + blockIdx.x * wrowsize);
}


void infomax(eegdataset_t *dataset) {
	/*
	* Configuration variables
	*/
	natural nsub;
	natural extended;
	natural verbose;
	natural biasing;
	natural channels;
	natural samples;
	natural pdfsize;
	natural urextblocks;
	natural extblocks;
	natural block;
	natural t;
	real lrate;
	real signsbias;
	real annealdeg;
	real annealstep;
	real * data;
	real nochange;
	size_t pitch;
	size_t ypitch;
	size_t upitch;
	size_t yupitch;
	size_t wpitch;
	size_t tmpwpitch;
	size_t startwpitch;
	size_t oldwpitch;
	size_t kkpitch;
	size_t oldkkpitch;
	size_t deltapitch;
	size_t olddeltapitch;

	/*
	 * Load config
	 */
	nsub = dataset->config.nsub;
	extended = dataset->config.extended;
	biasing = dataset->config.biasing;
	channels = dataset->nchannels;
	samples = dataset->nsamples;
	pdfsize = dataset->config.pdfsize;
	urextblocks = dataset->config.urextblocks;
	extblocks = dataset->config.extblocks;
	block = dataset->config.block;
	t = 0;
	data = (real*)dataset->devicePointer;
	lrate = dataset->config.lrate;
	signsbias = dataset->config.signsbias;
	annealstep = dataset->config.annealstep;
	annealdeg = dataset->config.annealdeg;
	nochange = dataset->config.nochange;
	pitch = dataset->pitch;
	verbose = dataset->config.verbose;
	int maxsteps = dataset->config.maxsteps;
	real momentum = dataset->config.momentum;
	if (verbose != 0) {

		fprintf(stdout, "*********************************\n");
		fprintf(stdout, "      Infomax configuration      \n");
		fprintf(stdout, "*********************************\n");
		fprintf(stdout, "  channels %d\n", channels);
		fprintf(stdout, "  samples %d\n", samples);
		fprintf(stdout, "  biasing %d\n", biasing);
		fprintf(stdout, "  extblocks %d\n", extblocks);
		fprintf(stdout, "  lrate %.16f\n", lrate);
		fprintf(stdout, "  block %d\n", block);
		fprintf(stdout, "  nochange %.16f\n", nochange);
		fprintf(stdout, "  maxsteps %d\n", maxsteps);
		fprintf(stdout, "  annealstep %.16f\n", annealstep);
		fprintf(stdout, "  annealdeg %.16f\n", annealdeg);
		fprintf(stdout, "  momentum %.16f\n", momentum);

		fprintf(stdout, "  nsub %d\n", nsub);
		fprintf(stdout, "  pdfsize %d\n", pdfsize);
		fprintf(stdout, "  urextblocks %d\n", urextblocks);
		fprintf(stdout, "  signsbias %.16f\n", signsbias);
		fprintf(stdout, "  extended %d\n", extended);


		fprintf(stdout, "  t %d\n", t);
		fprintf(stdout, "  data %p\n", data);
		fprintf(stdout, "  pitch %lu\n", pitch);
		fprintf(stdout, "*********************************\n");
	}


	DPRINTF(1, "Running with random seed %d\n", dataset->config.seed);
	r250_init(dataset->config.seed);

	/*
	 * Variables for CUBLAS
	 */
	real alpha = 1.0, beta = 0.0, gamma = -1.0;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
	cublasOperation_t transn = CUBLAS_OP_N;
	cublasOperation_t transt = CUBLAS_OP_T;
	cublasHandle_t handle;
	int inc = 1;

	HANDLE_CUBLAS_ERROR(cublasCreate(&handle));


	int	zero = 0;
	natural nchannels = channels;
	natural nsamples = dataset->nsamples;
	size_t chxch = nchannels * nchannels * sizeof(real);
	size_t ch = nchannels * sizeof(real);
	size_t intsamples = nsamples * sizeof(natural);

	natural h_weights_blowup = 0;
	natural	blockno = 1;
	natural pleft = nsamples;
	natural piter = 0;
	natural signcount = 0;
	real angledelta = 0.0f;
	real h_change = 0.0f;
	real h_oldchange = 0.0f;
	real epsilon = 0.0f;

	natural chxchblocks = nchannels;
	natural chxchthreads = nchannels;

	natural chthreads = getMaxThreads();

	natural * dataperm = NULL;
	natural * h_dataperm = NULL;
	real * weights = NULL;
	real * oldweights = NULL;
	real * startweights = NULL;
	real * tmpweights = NULL;
	real * bias = NULL;
	real * bsum = NULL;
	int * signs = NULL;
	natural * pdfperm = NULL;
	natural * h_pdfperm = NULL;
	real * kk = NULL;
	real * oldkk = NULL;
	real * u = NULL;
	real * y = NULL;
	real * yu = NULL;
	real * delta = NULL;
	real * olddelta = NULL;

	real extmomentum = DEFAULT_EXTMOMENTUM;
	real * prevweights = NULL;
	real * prevwtschange = NULL;
	size_t prevweightspitch = 0;
	size_t prevwtschangepitch = 0;


	/*
	 * Progressbar
	 */
#define UICOLS 80
	 char uiprog[UICOLS+1];
	 uiprog[UICOLS] = 0;
	 int uicurrent = 0;

	/*
	 * Permutation vector
	 */
	DPRINTF(2, "cudaMalloc %lu bytes for permutation vector (dataperm)\n", intsamples);
	HANDLE_ERROR(cudaMalloc(&dataperm, intsamples));
	h_dataperm = (natural*)malloc(intsamples);

	DPRINTF(2, "Pointer address in device: %p\n", dataperm);
	initperm(nsamples, (natural*) dataperm, h_dataperm);

	/*
	 * ch x ch matrixes
	 */

	DPRINTF(2, "cudaMalloc %lu bytes for weights (weights)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&weights, &wpitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", weights);

	DPRINTF(2, "cudaMalloc %lu bytes for tmpweights (tmpweights)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&tmpweights, &tmpwpitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", tmpweights);


	DPRINTF(2, "cudaMalloc %lu bytes for old weights (oldweights)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&oldweights, &oldwpitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", oldweights);

	DPRINTF(2, "cudaMalloc %lu bytes for start weights (startweights)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&startweights, &startwpitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", startweights);

	DPRINTF(2, "cudaMalloc %lu bytes for delta (delta)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&delta, &deltapitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", delta);

	DPRINTF(2, "cudaMalloc %lu bytes for old delta (olddelta)\n", chxch);
	HANDLE_ERROR(cudaMallocPitch(&olddelta, &olddeltapitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", olddelta);

	if (momentum > 0) {
		DPRINTF(2, "cudaMalloc %lu bytes for prevweights (prevweights)\n", chxch);
		HANDLE_ERROR(cudaMallocPitch(&prevweights, &prevwtschangepitch, nchannels * sizeof(real), nchannels));
		DPRINTF(2, "Pointer address in device: %p\n", prevweights);

		DPRINTF(2, "cudaMalloc %lu bytes for prevwtschange (prevwtschange)\n", chxch);
		HANDLE_ERROR(cudaMallocPitch(&prevwtschange, &prevwtschangepitch, nchannels * sizeof(real), nchannels));
		DPRINTF(2, "Pointer address in device: %p\n", prevwtschange);

		initprvweights<<<chxchblocks, chxchthreads>>>(weights, wpitch, prevweights, prevweightspitch, prevwtschange, prevwtschangepitch);
		CHECK_ERROR();
	}

	int initweights = 1;
	if (dataset->h_weights != NULL) {
		HANDLE_ERROR(cudaMemcpy2D(weights, wpitch, dataset->h_weights, nchannels * sizeof(real),  nchannels * sizeof(real), nchannels, cudaMemcpyHostToDevice));
		initweights = 0;
	}

	initChxChMatrixes<<<chxchblocks, chxchthreads>>>(weights, startweights, oldweights, wpitch,	startwpitch, oldwpitch, initweights);

	CHECK_ERROR();

	/*
	 * 1 x ch vectors
	 */
	if (biasing) {
		DPRINTF(2, "cudaMalloc %lu bytes for bias (bias)\n", ch);
		HANDLE_ERROR(cudaMalloc(&bias, ch));
		DPRINTF(2, "Pointer address in device: %p\n", bias);

		DPRINTF(2, "cudaMalloc %lu bytes for bias sums (bsum)\n", ch);
		HANDLE_ERROR(cudaMalloc(&bsum,nchannels * sizeof(real)));
		DPRINTF(2, "Pointer address in device: %p\n", bsum);
	}
	if (extended) {
		DPRINTF(2, "cudaMalloc %lu bytes for signs (signs)\n", ch);
		HANDLE_ERROR(cudaMalloc(&signs, ch));
		DPRINTF(2, "Pointer address in device: %p\n", signs);

		if (pdfsize > nsamples) {
			pdfsize = nsamples;
		}

		DPRINTF(2, "cudaMalloc %lu bytes for PDF permutation (pdfperm)\n", nsamples * sizeof(natural));
		HANDLE_ERROR(cudaMalloc(&pdfperm, nsamples * sizeof(natural)));
		h_pdfperm = (natural*)malloc(nsamples * sizeof(natural));
		DPRINTF(2, "Pointer address in device: %p\n", pdfperm);
		initperm(nsamples, (natural*) pdfperm, h_pdfperm);

		DPRINTF(2, "cudaMalloc %lu bytes for kurtosis estimation (kk)\n", nchannels * sizeof(real) * 2 * pdfsize);
		HANDLE_ERROR(cudaMallocPitch(&kk, &kkpitch, nchannels * sizeof(real), 2*pdfsize));
		DPRINTF(2, "Pointer address in device: %p\n", kk);

		DPRINTF(2, "cudaMalloc %lu bytes for old kurtosis estimation (oldkk)\n", ch);
		HANDLE_ERROR(cudaMallocPitch(&oldkk, &oldkkpitch, nchannels * sizeof(real), 1));
		DPRINTF(2, "Pointer address in device: %p\n", oldkk);
	}
	initChannelsVectors<<<1, chxchthreads>>>(bias, biasing, signs, oldkk, extended, nsub, channels);
	CHECK_ERROR();

	/*
	 * Alloc mem for other structures
	 */
	DPRINTF(2, "cudaMalloc %lu bytes for auxiliar matrix (u)\n", nchannels * sizeof(real) * block);
	HANDLE_ERROR(cudaMallocPitch(&u, &upitch, nchannels * sizeof(real), block));
	DPRINTF(2, "Pointer address in device: %p\n", u);

	DPRINTF(2, "cudaMalloc %lu bytes for auxiliar matrix (y)\n", nchannels * sizeof(real) * block);
	HANDLE_ERROR(cudaMallocPitch(&y, &ypitch, nchannels * sizeof(real), block));
	DPRINTF(2, "Pointer address in device: %p\n", y);

	DPRINTF(2, "cudaMalloc %lu bytes for auxiliar matrix (yu)\n", nchannels * sizeof(real) * block);
	HANDLE_ERROR(cudaMallocPitch(&yu, &yupitch, nchannels * sizeof(real), nchannels));
	DPRINTF(2, "Pointer address in device: %p\n", yu);

	urextblocks = extblocks;

	time_t start, stepstart, stepend, end;
	clock_t clockstart, clockstepstart, clockstepend, clockend;
	clock_t dif, hour, min, sec;
	clockstart = clock();
	time (&start);

	int step = 0;
	int numblocks = 0;
	numblocks = nsamples/block;
	int uiblockno = 1;

	while (step < maxsteps) {
		initperm(nsamples, (unsigned int*) dataperm, h_dataperm);
		uicurrent = 0;
		if (verbose > 1) {
			for (int i = 0; i < UICOLS; i++) {
				uiprog[i] = ' ';
			}
			uiblockno = 1;
			uiprog[uicurrent] = '>';

		}

		DPRINTF(3, "Will run for %i blocks\n", numblocks);

		time(&stepstart);
		clockstepstart = clock();
		for (t = 0; t < nsamples - block && !h_weights_blowup; t += block) {
			DPRINTF(3, "Starting step\n", numblocks);
			DPRINTF(3, "Step 1\n", numblocks);
			step1<<<block, nchannels, nchannels*sizeof(real), 0>>>(channels, extended, t, weights, data, u, y, dataperm, biasing, bias, wpitch, pitch, upitch, ypitch);
			CHECK_ERROR();
			DPRINTF(3, "Step 1 end\n", numblocks);
			if (extended || biasing) {
				DPRINTF(3, "Step 2\n");
				natural n_max_multi = block < MAX_MULTIPROCESSORS ? 1 : MAX_MULTIPROCESSORS;
				step2<<<n_max_multi, nchannels>>>(block, extended, channels, signs, y, bsum, ypitch, biasing);
				CHECK_ERROR();
				DPRINTF(3, "Step 2 end\n");
			}

/*
 * 			STEP 3 changed for CUBLAS function calls
			step3<<<nchannels, nchannels, block*sizeof(real), 0>>>(extended, channels, block, u, y, yu, upitch, ypitch, yupitch);
			CHECK_ERROR();
*/
			DPRINTF(3, "Step 3\n");
			if (! extended) {
				HANDLE_CUBLAS_ERROR(cublas(gemm)(handle, transn, transt, nchannels, nchannels, block, &alpha, y, ypitch/sizeof(real), u, upitch/sizeof(real), &beta, yu, yupitch/sizeof(real)));
			} else {
				HANDLE_CUBLAS_ERROR(cublas(syrk)(handle, uplo, transn, nchannels, block, &alpha, u, upitch/sizeof(real), &beta, yu, yupitch/sizeof(real)));
				unsigned long src;
				unsigned long dst;
				src = (unsigned long)yu;
				dst = src;

				src += ((nchannels-2) * yupitch) + (nchannels-1)*sizeof(real);
				dst += ((nchannels-1) * yupitch) + (nchannels-2)*sizeof(real);
				for (int i=1 ; i<nchannels ; i++) {
					HANDLE_CUBLAS_ERROR(cublas(copy)(handle, i, (real*)dst, yupitch/sizeof(real), (real*)src, 1));
					src -= yupitch+sizeof(real);
					dst -= yupitch+sizeof(real);
				}
				/* Compute: -y * u' -u*u' */
				HANDLE_CUBLAS_ERROR(cublas(gemm)(handle, transn, transt, nchannels, nchannels, block, &gamma, y, ypitch/sizeof(real), u, upitch/sizeof(real), &gamma, yu, yupitch/sizeof(real)));
			}
			addEye<<<nchannels, nchannels>>>(yu, yupitch/sizeof(real), block);
				CHECK_ERROR();
			DPRINTF(3, "Step 3 end\n");
			DPRINTF(3, "Step 4\n", numblocks);
			HANDLE_ERROR(cudaMemcpy2D(tmpweights, tmpwpitch, weights, wpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));

			HANDLE_CUBLAS_ERROR(cublas(gemm)(handle, transn,transn,nchannels,nchannels,nchannels,&lrate,yu,yupitch/sizeof(real),tmpweights,tmpwpitch/sizeof(real),&alpha,weights,wpitch/sizeof(real)));

			if (bias) {
				HANDLE_CUBLAS_ERROR(cublas(axpy)(handle, nchannels,&lrate,bsum,inc,bias,inc));
			}

/******************************** Add momentum ********************************/
			if (momentum > 0.0) {
				HANDLE_CUBLAS_ERROR(cublas(axpy)(handle, chxch, &momentum,prevwtschange,inc,weights,inc));
				matScale<<<nchannels, nchannels>>>(weights, wpitch/sizeof(real), prevweights, prevweightspitch/sizeof(real), prevwtschange, prevwtschangepitch/sizeof(real), -1.0);
				CHECK_ERROR();

				HANDLE_ERROR(cudaMemcpy2D(prevweights, prevweightspitch, weights, wpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));

			}
			DPRINTF(3, "Step 4 end\n", numblocks);


			/*
			 * Wait till step 4 finishes and check weigths_blowup;
			 */
			getBlowup<<<nchannels, nchannels>>>(weights, wpitch/sizeof(real));
			CHECK_ERROR();
			HANDLE_ERROR(cudaThreadSynchronize());
			HANDLE_ERROR(cudaMemcpyFromSymbol(&h_weights_blowup, SYMBOL(weights_blowup), sizeof(h_weights_blowup), 0, cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpyToSymbol(SYMBOL(weights_blowup), &zero, sizeof(zero), 0, cudaMemcpyHostToDevice));
#ifdef DEBUG
			if (h_weights_blowup) {
				dprintf(1, "HOST BLOWUP!\n");
			}
#endif

			if (extended && ! h_weights_blowup && extblocks > 0 && blockno%extblocks ==0) {
				DPRINTF(3, "PDF\n");
				if (pdfperm && pleft < pdfsize) {
					initperm(nsamples, pdfperm, h_pdfperm);
					piter = 0;
					pleft = nsamples;
				}
				int h_distintos = 0;
				HANDLE_ERROR(cudaMemcpyToSymbol(SYMBOL(distintos), &h_distintos, sizeof(h_distintos)));

				/*
				 * PDF
				 */
				DPRINTF(3,"Launching PDF with %d blocks, %d threads, %lu shared mem, data=%p, nchannels=%d, w=%p, pdfperm=%p, pdfsize=%d, piter=%d, signs=%p, signsbias=%f, pitch=%d, wpitch=%d, kkpitch=%d, kk=%p, oldkk=%p, extmomentum=%f\n", pdfsize, nchannels, (nchannels+2)*sizeof(real),data, nchannels, weights, pdfperm, pdfsize, piter, signs, signsbias, pitch, wpitch, kkpitch, kk, oldkk, extmomentum);
				pdf<<<pdfsize, nchannels, (nchannels+2) * sizeof(real), 0>>>(data, nchannels, weights, pdfperm, pdfsize, piter, signs, signsbias, kk, pitch, wpitch, kkpitch, oldkk, extmomentum);
				CHECK_ERROR();
				HANDLE_ERROR(cudaThreadSynchronize());
				HANDLE_ERROR(cudaMemcpyFromSymbol(&h_distintos, SYMBOL(distintos), sizeof(h_distintos)));
				DPRINTF(3, "PDF end\n");
				if (!h_distintos) signcount++;
				else signcount = 0;
				DPRINTF(3, "Signcount %d - distintos %d\n", signcount, h_distintos);
				if (signcount >= SIGNCOUNT_THRESHOLD) {
					extblocks = (int)(extblocks * SIGNCOUNT_STEP);
					signcount = 0;
				}
				piter++;
				pleft -= pdfsize;

			}
			if (verbose > 1) {
				if (((uiblockno * UICOLS) / numblocks) > uicurrent) {
					uiprog[uicurrent] = '=';
					if (uicurrent < UICOLS-1) {
						uiprog[++uicurrent] = '>';
					}
					printf(ERASE_STRING);
					printf("Step %d [%s]  [block %d of %d]", step+1, uiprog, uiblockno , numblocks);
					fflush(stdout);
				}
				uiblockno++;
			}
			blockno++;

		}
		if (!h_weights_blowup) {
			step ++;
			angledelta = 0.0;
			calcDelta<<<nchannels, nchannels, 0, 0>>>(nchannels, delta, weights, oldweights, deltapitch, wpitch, oldwpitch);
			CHECK_ERROR();
			dotProductSame<<<1, nchannels, nchannels * sizeof(real), 0>>>(nchannels, delta, deltapitch);
			CHECK_ERROR();
			HANDLE_ERROR(cudaThreadSynchronize());
			HANDLE_ERROR(cudaMemcpyFromSymbol(&h_change, SYMBOL(dotResult), sizeof(h_change)));
			time(&stepend);
			clockstepend = clock();
			dif = difftime(stepend,stepstart);
			hour = dif/3600;
			min = dif/60 % 60;
			sec = dif % 60;

			if (step > 2) {
				dotProduct<<<1, nchannels, nchannels * sizeof(real), 0>>>(nchannels, delta, olddelta, deltapitch, olddeltapitch);
				CHECK_ERROR();
				HANDLE_ERROR(cudaThreadSynchronize());
				HANDLE_ERROR(cudaMemcpyFromSymbol(&epsilon, SYMBOL(dotResult), sizeof(epsilon)));
				angledelta = acos(epsilon/sqrt(h_change*h_oldchange));
				if (verbose != 0) {
					if (verbose > 1) {
						printf("\n");
					}
					printf("Step %d - epsilon: %7.9f, lrate %7.9f, wchange %7.9f, angledelta %4.1f deg",step, epsilon, lrate,h_change,DEGCONST*angledelta);
				} else {
					printf("%d ", step);
				}


			} else {
				if (verbose != 0) {
					if (verbose > 1) {
						printf("\n");
					}
					printf("Step %d - lrate %7.9f, wchange %7.9f",step,lrate,h_change);
				} else {
					printf("%d ", step);
				}
			}
			if (verbose != 0) {
			printf(" - Elapsed time: %lu tics = %lu h %lu m %lu s\n",
				clockstepend - clockstepstart, hour, min, sec);
			}
		} else {
			if (verbose != 0) {
				printf(ERASE_STRING);
				printf("Step %d [ BLOWUP! ]\n", step+1);
			} else {
				printf("Step %d [ BLOWUP! ]", step+1);
			}
			step = 0;
			h_change = nochange;
			h_weights_blowup = 0;
			blockno = 1;
			extblocks = urextblocks;
			lrate = lrate * DEFAULT_RESTART_FAC;
			HANDLE_ERROR(cudaMemcpy2D(weights, wpitch, startweights, startwpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemset2D(delta, deltapitch, 0, nchannels * sizeof(real), nchannels));
			HANDLE_ERROR(cudaMemset2D(olddelta, olddeltapitch, 0, nchannels * sizeof(real), nchannels));
			initChannelsVectors<<<1, chxchthreads>>>(bias, biasing, /*oldsigns,*/ signs, oldkk, extended, nsub, channels);
			CHECK_ERROR();

			if (momentum > 0.0) {
				HANDLE_ERROR(cudaMemcpy2D(oldweights, oldwpitch, startweights, startwpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
				HANDLE_ERROR(cudaMemcpy2D(prevweights, prevweightspitch, startweights, startwpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
				HANDLE_ERROR(cudaMemset2D(prevwtschange, prevwtschangepitch, 0, nchannels * sizeof(real), nchannels));
			}


			if (lrate > MIN_LRATE) {
				if (verbose != 0) {
					printf("Lowering learning rate to %g and starting again.\n",lrate);
				}
			} else {
				printf("QUITTING - weight matrix may not be invertible!\n");
				exit(1);
			}
		}
		HANDLE_ERROR(cudaMemcpy2D(oldweights, oldwpitch, weights, wpitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));

		if (DEGCONST*angledelta > annealdeg) {
			HANDLE_ERROR(cudaMemcpy2D(olddelta, olddeltapitch, delta, deltapitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
			lrate = lrate*annealstep;
			h_oldchange = h_change;
		} else {
			if (step == 1) {
				HANDLE_ERROR(cudaMemcpy2D(olddelta, olddeltapitch, delta, deltapitch, nchannels * sizeof(real), nchannels, cudaMemcpyDeviceToDevice));
				h_oldchange = h_change;
			}
		}

		if (step > 2 && h_change < nochange) {
			step = maxsteps;
		} else {
			if (h_change > DEFAULT_BLOWUP) {
				lrate = lrate*DEFAULT_BLOWUP_FAC;
			}
		}
	}

	time (&end);
	clockend = clock();
	dif = difftime(end,start);
	hour = dif/3600;
	min = dif/60 % 60;
	sec = dif % 60;
	printf("\nElapsed Infomax time (%d c %d s %d min) %lu secs: %lu tics = %lu h %lu m %lu s\n", nchannels, nsamples, nsamples/(512*60), dif, clockend-clockstart, hour, min, sec);

	clockend = clock();

	if (dataperm) HANDLE_ERROR(cudaFree(dataperm));
	dataset->weights = weights;
	dataset->wpitch = wpitch;
	if (bias) dataset->bias = bias;
	if (signs) dataset->signs = signs;
	if (oldweights) HANDLE_ERROR(cudaFree(oldweights));
	if (startweights) HANDLE_ERROR(cudaFree(startweights));
	if (bsum) HANDLE_ERROR(cudaFree(bsum));
	if (pdfperm) HANDLE_ERROR(cudaFree(pdfperm));
	if (kk) HANDLE_ERROR(cudaFree(kk));
	if (oldkk) HANDLE_ERROR(cudaFree(oldkk));
	if (u) HANDLE_ERROR(cudaFree(u));
	if (y) HANDLE_ERROR(cudaFree(y));
	if (yu) HANDLE_ERROR(cudaFree(yu));

	HANDLE_CUBLAS_ERROR(cublasDestroy(handle));

}
