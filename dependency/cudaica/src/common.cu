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
#include <config.h>
#include <common.h>
#include <device.h>
#include <error.h>

#define MAX_DIMENSION(a) ( a > 512 ? 512 : a)

/*
 * Zeroes a matrix.
 * Should dispatch rows blocks of cols threads
 *
 * data: matrix
 * pitch: matrix row size in bytes
 */
__global__ void zeroMatrix(real* data, size_t pitch) {
	data[threadIdx.x + blockIdx.x * pitch/sizeof(real)] = 0.0;
}

 /*
  * Identity matrix
  * Should be launched with channels blocks x channels threads
  *
  * data: matrix
  * pitch: matrix row size in bytes
  */
__global__ void eye(real * data, size_t pitch) {

	size_t colwidth = pitch/sizeof(real);
	real value = (threadIdx.x == blockIdx.x ? 1.0 : 0.0);

	data[threadIdx.x + blockIdx.x * colwidth] = value;
}


/*
 * Memory mappings functions
 */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
void *mapmalloc(size_t size) {
	void * base;
#ifdef DARWIN
	base = malloc(size);

#else
	int     fd;


	fd = open("/dev/zero",O_RDWR);
	base = mmap(NULL,(size_t)(size),PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
	if (base == MAP_FAILED) {
		printf("Oh dear, something went wrong with mmap()! %s\n", strerror(errno));
	}
	close(fd);
#endif
	return base;
}

void mapfree(void *addr, size_t size) {
#ifdef DARWIN
	free(addr);
#else
	munmap(addr,size);
#endif
}

/*
 *
 * Read a total of size floting point values from the file specified by
 * fname to the device memory.
 *
 * fname: file name to be read
 * size: numer of bytes in the file
 * mat: output (should have enough size)
 */
void dev_matread(char *fname, int rows, int cols, real *mat, size_t pitch) {
	FILE *file = fopen(fname,"rb");
	real *buffer;
	float *floatbuffer;
	int items;
	int size = rows * cols;
	if (!file) {
		printf("open failed\n");
		exit (0);
	}

	floatbuffer = (float*)mapmalloc(size*sizeof(float));
	buffer = (real*)mapmalloc(size*sizeof(real));


	items = (int)fread(floatbuffer,sizeof(float),size,file);
	if (items != size) {
		printf("invalid number of elements\n");
		exit (0);
	}
	for (int i = 0; i < size; i++){
		buffer[i] = (real) floatbuffer[i];
	}

	HANDLE_ERROR(cudaMemcpy2D(mat, pitch, buffer, cols*sizeof(real), cols*sizeof(real), rows, cudaMemcpyHostToDevice));

	mapfree(buffer,size*sizeof(real));
	mapfree(floatbuffer,size*sizeof(float));

	fclose(file);
}


/*
 *
 * Read a total of size integer values from the file specified by fname to the device memory.
 *
 * fname: file name to be read
 * size: numer of bytes in the file
 * mat: output (should have enough size)
 */
void dev_matreadInt(char *fname, int rows, int cols, int *mat, size_t pitch) {
	FILE *file = fopen(fname,"rb");
	int *buffer;
	int items;
	int size = rows * cols;
	if (!file) {
		printf("open failed\n");
		exit (0);
	}

 	buffer = (int*)mapmalloc(size*sizeof(int));

	items = (int)fread(buffer,sizeof(int),size,file);
	if (items != size) {
		printf("invalid number of elements\n");
		exit (0);
	}

	HANDLE_ERROR(cudaMemcpy2D(mat, pitch, buffer, cols*sizeof(int), cols*sizeof(int), rows, cudaMemcpyHostToDevice));

	mapfree(buffer,size*sizeof(int));

	fclose(file);
}

/*
 * Write a total of size floting point values from matrix in the device memory to the
 * file.
 *
 * fname: file name to be written to
 * rows: number of rows in the matrix
 * cols: number of rows in the matrix
 * mat: matrix
 * pitch: matrix row size in bytes
 */
void dev_matwrite(char *fname, int rows, int cols, real *mat, size_t pitch) {
	FILE *file = fopen(fname,"wb");
	real *buffer;
	float *floatbuffer;
	int items;
	int size = rows * cols;
	if (!file) {
		printf("open failed\n");
		exit (0);
	}
	buffer = (real*)mapmalloc(size*sizeof(real));
	DPRINTF(2, "Copying %d by %d cols from %p to %p\n", rows, cols, mat, buffer);
	HANDLE_ERROR(cudaMemcpy2D(buffer, cols*sizeof(real), mat, pitch, cols*sizeof(real), rows, cudaMemcpyDeviceToHost));
	floatbuffer = (float*)mapmalloc(size*sizeof(float));
	for (int i = 0; i < size; i++){
		floatbuffer[i] = (float) buffer[i];
	}
	items = (int)fwrite(floatbuffer,sizeof(float),size,file);
	if (items != size) {
		printf("invalid number of elements\n");
		exit (0);
	}

	mapfree(floatbuffer, size*sizeof(float));
	mapfree(buffer,size*sizeof(real));


	fclose(file);
}


/*
 * Write a total of size integer values from matrix in the device memory to the
 * file.
 *
 * fname: file name to be written to
 * rows: number of rows in the matrix
 * cols: number of rows in the matrix
 * mat: matrix
 * pitch: matrix row size in bytes
 */
void dev_matwriteInt(char *fname, int rows, int cols, int *mat, size_t pitch) {
	FILE *file = fopen(fname,"wb");
	int *buffer;
	int items;
	int size = rows * cols;
	if (!file) {
		printf("open failed\n");
		exit (0);
	}
	buffer = (int*)mapmalloc(size*sizeof(int));
	HANDLE_ERROR(cudaMemcpy2D(buffer, cols*sizeof(int), mat, pitch, cols*sizeof(int), rows, cudaMemcpyDeviceToHost));
	items = (int)fwrite(buffer,sizeof(int),size,file);
	if (items != size) {
		printf("invalid number of elements\n");
		exit (0);
	}


	mapfree(buffer,size*sizeof(int));


	fclose(file);
}


/*
 * Write a total of size unsigned integer values from matrix in the device memory to the
 * file.
 *
 * fname: file name to be written to
 * rows: number of rows in the matrix
 * cols: number of rows in the matrix
 * mat: matrix
 * pitch: matrix row size in bytes
 */
void dev_matwriteNat(char *fname, int rows, int cols, natural *mat, size_t pitch) {
	FILE *file = fopen(fname,"wb");
	natural *buffer;
	int items;
	int size = rows * cols;
	if (!file) {
		printf("open failed\n");
		exit (0);
	}
	buffer = (natural*)mapmalloc(size*sizeof(natural));
	HANDLE_ERROR(cudaMemcpy2D(buffer, cols*sizeof(natural), mat, pitch, cols*sizeof(natural), rows, cudaMemcpyDeviceToHost));
	items = (natural)fwrite(buffer,sizeof(natural),size,file);
	if (items != size) {
		printf("invalid number of elements\n");
		exit (0);
	}


	mapfree(buffer,size*sizeof(int));


	fclose(file);
}



void printVector(real* data, natural size) {
	int j = 0;
	for (j = 0; j < size; j++) {
		printf("%d = %f\n", j, data[j]);
	}
}

void dev_printVector(real* data, natural size, natural max) {
	real* host = (real*) malloc(size*sizeof(real));
	HANDLE_ERROR(cudaMemcpy(host, data, size*sizeof(real), cudaMemcpyDeviceToHost));
	printVector(host, max);
	free(host);
}

/*
 * Calculates greatest common divisor between u and v
 */
unsigned int gcd(natural u, natural v) {
	int shift;
	if (u == 0 || v == 0)
		return u | v;
	for (shift = 0; ((u | v) & 1) == 0; ++shift) {
		u >>= 1;
		v >>= 1;
	}
	while ((u & 1) == 0)
		u >>= 1;
	do {
		while ((v & 1) == 0)
			v >>= 1;
		if (u < v) {
			v -= u;
		} else {
			natural diff = u - v;
			u = v;
			v = diff;
		}
		v >>= 1;
	} while (v != 0);

	return u << shift;
 }


/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

real dsum_(integer *n, real *dx, integer *incx) {

	/* System generated locals */
	//~ integer i__1, i__2;
	real ret_val, d__1, d__2, d__3, d__4, d__5, d__6;

	/* Local variables */
	static integer i, m;
	static real dtemp;
	static integer nincx, mp1;

#define DX(I) dx[(I)-1]


	ret_val = 0.;
	dtemp = 0.;
	if (*n <= 0 || *incx <= 0) {
		return ret_val;
	}
	if (*incx == 1) {
		goto L20;
	}

/* code for increment not equal to 1 */

	nincx = *n * *incx;
	for (i = 1; *incx < 0 ? i >= nincx : i <= nincx; i += *incx) {
		dtemp += (d__1 = DX(i), d__1);
	}
	ret_val = dtemp;
	return ret_val;
L20:
	m = *n % 6;
	if (m == 0) {
		goto L40;
	}
	for (i = 1; i <= m; ++i) {
		dtemp += (d__1 = DX(i), d__1);
	}
	if (*n < 6) {
		goto L60;
	}
L40:
	mp1 = m + 1;
	for (i = mp1; i <= *n; i += 6) {
		dtemp = dtemp + (d__1 = DX(i), d__1) + (d__2 = DX(i + 1),
		d__2) + (d__3 = DX(i + 2), d__3) + (d__4 = DX(i + 3),
		d__4) + (d__5 = DX(i + 4), d__5) + (d__6 = DX(i + 5)
		, d__6);
	}
L60:
	ret_val = dtemp;
	return ret_val;
} /* dsum_ */
