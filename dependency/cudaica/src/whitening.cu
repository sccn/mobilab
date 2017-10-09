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
#include <whitening.h>
#include <error.h>
#include <common.h>
#include <device.h>
#include <cblas.h>

/*
 * Multiplies sphere matrix by data
 * Should be launched with samples blocks and channels threads
 *
 * sphere: sphere matrix
 * spitch: sphere matrix row size in bytes
 * data: data matrix
 * pitch: data matrix row size in bytes
 * channles: number of channels
 */
extern __shared__ real sample[];
__global__ void multbySphere(real *sphere, size_t spitch, real* data, size_t pitch, natural channels) {
	int colwidth = pitch/sizeof(real);
	int scolwidth = spitch/sizeof(real);
	sample[threadIdx.x]  = data[blockIdx.x * colwidth + threadIdx.x];
	__syncthreads();

	real value = 0.0f;
	int i = 0;
	for (i = 0; i < channels; i++) {
		value += sphere[threadIdx.x + scolwidth * i] * sample[i];
	}
	data[blockIdx.x * colwidth + threadIdx.x] = value;
}

/*
 *	[v d] = eig(cov(data'))
 *   sphere = v * d^(-1) * v'
 *  Taken from Efficient Independent Component Analysis on a GPU
 */
void whiten(eegdataset_t *set) {
	DPRINTF(1,"Whitening dataset\n");
	real *spherematrix;
	size_t spitch;
	DPRINTF(2, "cudaMallocPitch %d rows of %lu bytes for sphere matrix\n", set->nchannels, set->nchannels * sizeof(real));
	HANDLE_ERROR(cudaMallocPitch(&spherematrix, &spitch, set->nchannels * sizeof(real), set->nchannels));

	if (set->config.sphering == 2 || (set->config.sphering == 0 && set->config.weightsinfile != NULL)) {
		eye<<<set->nchannels, set->nchannels>>>(spherematrix, spitch);
		CHECK_ERROR();
		set->spitch = spitch;
		set->sphere = spherematrix;
		return;
	}

	int n = set->nsamples;
	int m = set->nchannels;

	real alpha = 1.0/(real)(n-1);
	real beta = 0.0;
	int info = 0;

	int nb = 8;//ilaenv_(&ispec,name,opts,&m,&na,&na,&na); Segfaults
	int i, im, lwork = (nb+2)*m, inc = 1, mxm = m*m;

	char uplo='U', transn='N', jobz='V';
	real *host_sphe = (real*)malloc(m*m*sizeof(real));
	real *host_eigv = (real*)malloc(m*m*sizeof(real));
	real *host_eigd = (real*)malloc(m*sizeof(real));
	int  *host_ipiv = (int*)malloc(m*sizeof(int));
	real *host_work = (real*)malloc(lwork*sizeof(real));
	real *host_data = set->data;

	dsyrk_(&uplo,&transn,&m,&n,&alpha,host_data,&m,&beta,host_sphe,&m);
	dsyev_(&jobz,&uplo,&m,host_sphe,&m,host_eigd,host_work,&lwork,&info);
	for (i=0,im=0 ; i<m ; i++,im+=m)
		dcopy_(&m,&host_sphe[im],&inc,&host_eigv[i],&m);
	dcopy_(&mxm,host_eigv,&inc,host_sphe,&inc);
	for (i=0 ; i<m ; i++) {
		host_eigd[i] = 0.5 * sqrt(host_eigd[i]);
		dscal_(&m,&host_eigd[i],&host_eigv[i],&m);

	}
	dgesv_(&m,&m,host_eigv,&m,host_ipiv,host_sphe,&m,&info);

	HANDLE_ERROR(cudaMemcpy2D(spherematrix, spitch, host_sphe, set->nchannels*sizeof(real), set->nchannels*sizeof(real), set->nchannels,  cudaMemcpyHostToDevice));
	free(host_sphe);
	free(host_work);
	free(host_ipiv);
	free(host_eigd);
	free(host_eigv);

	natural nthreads = set->nchannels;
	natural nblocks = set->nsamples > MAX_CUDA_BLOCKS ? MAX_CUDA_BLOCKS : set->nsamples;
	natural start = 0;
	for (start = 0; start < set->nsamples; start += nblocks) {
		if (nblocks > (set->nsamples - start)) nblocks = (set->nsamples - start);
		DPRINTF(3, "Calling multBySphere with %d blocks, %d threads, src %p, size (%d x %d), pitch %lu starting at offset %d\n", nblocks, nthreads, set->devicePointer, set->nsamples, set->nchannels, set->pitch, start);
		multbySphere<<<nblocks, nthreads, set->nchannels * sizeof(real), 0>>>(spherematrix, spitch, (real*)set->devicePointer + (start * set->pitch/sizeof(real)), set->pitch, set->nchannels);
		CHECK_ERROR();
	}
	if (set->config.sphering == 1) {
		set->spitch = spitch;
		set->sphere = spherematrix;
	} else if (set->config.sphering == 0) {
		if (set->config.weightsinfile == NULL) {
			set->weights = spherematrix;
			set->wpitch = spitch;
			HANDLE_ERROR(cudaMallocPitch(&set->sphere, &set->spitch, set->nchannels * sizeof(real), set->nchannels));
			eye<<<set->nchannels, set->nchannels>>>(set->sphere, set->spitch);
			CHECK_ERROR();
		}
	}
}
