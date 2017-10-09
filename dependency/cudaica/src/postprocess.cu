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
#include <postprocess.h>
#include <error.h>
#include <common.h>
#include <cblas.h>

/*
 * This methods were extracted from original Infomax
 */


/*************** Orient components toward positive activations ****************/
/* Project data using trsf. Negate components and their corresponding weights */
/* to assure positive RMS. Returns projections in proj.                       */
/*                                                                            */
/* data: double array [m,n] (input)                                           */
/* trsf: double array [m,m] (input/output)                                    */
/* m:    natural (input)                                                      */
/* n:    natural (input)                                                      */
/* proj: double array [m,n] (output)                                          */

void posact(real *data, real *trsf, integer m, integer n, real *proj) {
	char trans='N';
	real alpha = 1.0;
	real beta = 0.0;
	real posrms, negrms;
	natural pos, neg, i, j;

	dgemm_(&trans,&trans,&m,&n,&m,&alpha,trsf,&m,data,&m,&beta,proj,&m);

	printf("Inverting negative activations: ");
	for (i=0 ; i<m ; i++) {
		posrms = 0.0; negrms = 0.0;
		pos = 0; neg = 0;
		for (j=i ; j<m*n ; j+=m)
			if (proj[j] >= 0) {
				posrms += proj[j]*proj[j];
				pos++;
			}
			else {
				negrms += proj[j]*proj[j];
				neg++;
			}

		if (negrms*(real)pos > posrms*(real)neg) {
			printf("-");
			for (j=i ; j<m*n ; j+=m) proj[j] = -proj[j];
			for (j=i ; j<m*m ; j+=m) trsf[j] = -trsf[j];
		}
		printf("%d ",(int)(i+1));
	}
	printf("\n");
}

/******************* Project data using a general projection ******************/
/* Project data using trsf. Return projections in proj.                       */
/*                                                                            */
/* data: double array [m,n] (input)                                           */
/* trsf: double array [m,m] (input)                                           */
/* m:    natural (input)                                                      */
/* n:    natural (input)                                                      */
/* proj: double array [m,n] (output)                                          */

void geproj(real *data, real *trsf, integer m, integer n, real *proj) {
	real alpha = 1.0, beta = 0.0;
	char trans='N';

	dgemm_(&trans,&trans,&m,&n,&m,&alpha,trsf,&m,data,&m,&beta,proj,&m);
}


/****************** Sort data according to projected variance *****************/
/* Compute back-projected variances for each component based on inverse of    */
/* weights and sphere or pseudoinverse of weights, sphere, and eigv. Reorder  */
/* data and weights accordingly. Also if not NULL, reorder bia and signs.    */
/*                                                                            */
/* data:    double array [m,n] (input/output)                                 */
/* weights: double array [m,m] (input/output)                                 */
/* sphere:  double array [m,m] (input)                                        */
/* eigv:    double array [m,k] (input) or NULL                                */
/* bias:    double array [m] (input/output) or NULL                           */
/* signs:   integer array [m] (input/output) or NULL                          */
/* m:       integer (input)                                                   */
/* n:       integer (input)                                                   */
/* k:       integer (input)                                                   */

typedef struct {
    integer    idx;
    real val;
} idxelm;

int compar(const void *x, const void *y) {
	if (((idxelm*)x)->val < ((idxelm*)y)->val) return 1;
	if (((idxelm*)x)->val > ((idxelm*)y)->val) return -1;
	return 0;
}

void varsort(real *data, real *weights, real *sphere, real *eigv, real *bias, integer *signs, integer m, integer n, integer k) {
	real alpha = 1.0, beta = 0.0;
	integer i, j, l, jm, ik, info = 0;
	char transn='N', transt='T', uplo='U', side = 'R';


	integer nb = 8; //ilaenv_(&ispec,name,opts,&m,&na,&na,&na); segfaults!
	integer itmp, lwork = m*nb, inc = 1;
	real act, dtmp, *wcpy;


	integer    *ipiv = (integer*)malloc(m*sizeof(integer));
	real *work = (real*)malloc(lwork*sizeof(real));
	real *winv = (real*)malloc(m*k*sizeof(real));
	real *sum  = (real*)malloc(n*sizeof(real));
	idxelm  *meanvar = (idxelm*)malloc(m*sizeof(idxelm));

	if (eigv) {
/* Compute pseudoinverse of weights*sphere*eigv */
		wcpy = (real*)malloc(m*m*sizeof(real));
		dsymm_(&side,&uplo,&m,&m,&alpha,sphere,&m,weights,&m,&beta,wcpy,&m);
		dgemm_(&transn,&transt,&m,&k,&m,&alpha,wcpy,&m,eigv,&k,&beta,weights,&m);

		dgetrf_(&m,&m,wcpy,&m,ipiv,&info);
		dgetri_(&m,wcpy,&m,ipiv,work,&lwork,&info);
		dgemm_(&transn,&transn,&k,&m,&m,&alpha,eigv,&k,wcpy,&m,&beta,winv,&k);
		free(wcpy);
	}
	else {
/* Compute inverse of weights*sphere */
		dsymm_(&side,&uplo,&m,&m,&alpha,sphere,&m,weights,&m,&beta,winv,&m);
		dgetrf_(&m,&m,winv,&m,ipiv,&info);
		dgetri_(&m,winv,&m,ipiv,work,&lwork,&info);
	}

/* Compute mean variances for back-projected components */
	for (i=0 ; i<m*k ; i++) winv[i] = winv[i]*winv[i];

	for (i=0,ik=0 ; i<m ; i++,ik+=k) {
		for (j=0,jm=0 ; j<n ; j++,jm+=m) {
			sum[j] = 0;
			act = data[i+jm]*data[i+jm];
			for(l=0 ; l<k ; l++) sum[j] += act*winv[l+ik];
		}

		meanvar[i].idx = i;
		meanvar[i].val = dsum_(&n,sum,&inc) / (real)((m*n)-1);

	}


/* Sort meanvar */
	qsort(meanvar,m,sizeof(idxelm),compar);

	printf("Permuting the activation wave forms ...\n");

/* Perform in-place reordering of weights, data, bias, and signs */
	for (i=0 ; i<m-1 ; i++) {
		j = meanvar[i].idx;
		if (i != j) {
			dswap_(&k,&weights[i],&m,&weights[j],&m);
			dswap_(&n,&data[i],&m,&data[j],&m);

			if (bias) {
				dtmp = bias[i];
				bias[i] = bias[j];
				bias[j] = dtmp;
			}

			if (signs) {
				itmp = signs[i];
				signs[i] = signs[j];
				signs[j] = itmp;
			}

			for (l=i+1 ; i!=meanvar[l].idx ; l++);
			meanvar[l].idx = j;
		}
		printf("%d ",(int)(i+1));
	}
	printf("\n");

	free(ipiv);
	free(work);
	free(winv);
	free(sum);
	free(meanvar);
}


/*
 * Post process data
 * TODO: do it in cuda.
 */
void postprocess(eegdataset_t *set) {

	integer ncomps = set->nchannels;
	integer datalength = set->nsamples;
	real * dataB = (real*)malloc(ncomps*datalength*sizeof(real));
	real * weights = (real*)malloc(ncomps*ncomps*sizeof(real));

	HANDLE_ERROR(cudaMemcpy2D(weights, ncomps*sizeof(real), set->weights, set->wpitch, ncomps*sizeof(real), ncomps, cudaMemcpyDeviceToHost));

	if (set->config.posact) {
		posact(set->data,weights,ncomps,datalength,dataB);
	} else {
		geproj(set->data,weights,ncomps,datalength,dataB);
	}

	free(set->data);
	set->data = dataB;

	printf("Sorting components in descending order of mean projected variance ...\n");
	real * sphere = (real*)malloc(ncomps*ncomps*sizeof(real));
	real * bias = NULL;
	integer * signs = NULL;

	HANDLE_ERROR(cudaMemcpy2D(sphere, ncomps*sizeof(real), set->sphere, set->spitch, ncomps*sizeof(real), ncomps, cudaMemcpyDeviceToHost));
	if (set->bias != NULL) {
		bias = (real*)malloc(ncomps*sizeof(real));
		HANDLE_ERROR(cudaMemcpy(bias, set->bias, ncomps*sizeof(real), cudaMemcpyDeviceToHost));
	}
	if (set->signs != NULL) {
		signs = (integer*)malloc(ncomps*sizeof(integer));
		HANDLE_ERROR(cudaMemcpy(signs, set->signs, ncomps*sizeof(integer), cudaMemcpyDeviceToHost));
	}
	varsort(set->data,weights,sphere,NULL,bias,signs,ncomps,datalength, ncomps);
	HANDLE_ERROR(cudaMemcpy2D(set->weights, set->wpitch, weights, ncomps*sizeof(real), ncomps*sizeof(real), ncomps, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy2D(set->sphere, set->spitch, sphere, ncomps*sizeof(real), ncomps*sizeof(real), ncomps, cudaMemcpyHostToDevice));
	if (set->bias != NULL) {
		HANDLE_ERROR(cudaMemcpy(set->bias, bias, ncomps*sizeof(real), cudaMemcpyHostToDevice));
		free(bias);
	}
	if (set->signs != NULL) {
		HANDLE_ERROR(cudaMemcpy(set->signs, signs, ncomps*sizeof(integer), cudaMemcpyHostToDevice));
		free(signs);
	}

	free(sphere);




}
