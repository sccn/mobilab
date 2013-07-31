#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"

/* Bspline transformation grid function
 * function [Error,ErrorGradient]=bspline_error_2d_double(Ox,Oy,I1,I2,dx,dy)
 * 
 * Ox, Oy are the grid points coordinates
 * I1 is the moving input image
 * I2 is the static input image
 * dx and dy are the spacing of the b-spline knots
 *
 * Error: The squared pixel distance error (SSD)
 * ErrorGradient: The error gradient from the grid control points
 *
 * This function is an implementation of the b-spline registration
 * algorithm in "D. Rueckert et al. : Nonrigid Registration Using Free-Form 
 * Deformations: Application to Breast MR Images".
 * 
 * We used "Fumihiko Ino et al. : a data distrubted parallel algortihm for 
 * nonrigid image registration" for the correct formula's, because 
 * (most) other papers contain errors. 
 *
 * Function is written by D.Kroon University of Twente (July 2009)
 */

voidthread transformvolume_error_gray(double **Args) {

    double *Bu, *Bv, *dxa, *dya, *ThreadID, *Ox, *Oy, *I1, *I2, *ThreadOut;
    double *Isize_d, *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[2]={0,0};
	int Onumel;
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    /* Location of pixel which will be come the current pixel */
    double Tlocalx, Tlocaly;
    /* Variables to store 1D index */
    int indexO, indexI;
    /* Grid distance */
    int dx,dy; 
    /* X,Y coordinates of current pixel */
    int x,y;
    /* B-spline variablesl */
    int u_index=0, v_index=0, i, j;
    /* temporary value */
	double val;
    /* Count number of pixels used for error for normalization */
	int err_pixelc=0;
    /* Look up tables index */
	int *u_index_array, *i_array, *v_index_array, *j_array;
    /*  B-Spline loop variabels */
    int l,m;
    /* current accumlated image error */
	double err=0;
    /* Current voxel/pixel */
    double Ipixel;
    /* Split input into variables */
    Bu=Args[0]; Bv=Args[1];
    Isize_d=Args[2]; Osize_d=Args[3];
    ThreadOut=Args[4];
	dxa=Args[5]; dya=Args[6];
    ThreadID=Args[7]; ThreadOffset=(int) ThreadID[0]; 
    Ox=Args[8]; Oy=Args[9];
    I1=Args[10]; I2=Args[11];
	Nthreadsd=Args[12];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; Isize[1] = (int)Isize_d[1];
    Osize[0] = (int)Osize_d[0]; Osize[1] = (int)Osize_d[1]; 
    Onumel=Osize[0]*Osize[1];
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
	/*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=ThreadOffset; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}
   
    /*  Loop through all image pixel coordinates */
    for (y=ThreadOffset; y<Isize[1]; y=y+Nthreads)
    {
  		v_index=v_index_array[y]; j=j_array[y];
        for (x=0; x<Isize[0]; x++)
        {
            /* Calculate the index needed to loop up the B-spline values. */
			u_index=u_index_array[x]; i=i_array[x];
            /*  This part calculates the coordinates of the pixel */
            /*  which will be transformed to the current x,y pixel. */
            Tlocalx=0; Tlocaly=0;
            for(l=0; l<4; l++)
            {
				if(((i+l)>=0)&&((i+l)<Osize[0]))
				{
					for(m=0; m<4; m++)
					{    
						 if(((j+m)>=0)&&((j+m)<Osize[1]))
						 {
							  indexO=(i+l)+(j+m)*Osize[0];
							  val=Bu[l+u_index]*Bv[m+v_index];
							  Tlocalx+=val*Ox[indexO]; Tlocaly+=val*Oy[indexO];
						 }
					}
				}
            }            
            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            Ipixel = interpolate_2d_double_gray(Tlocalx, Tlocaly, Isize, I1,false,false);
            err+=pow2(I2[indexI]-Ipixel);
            err_pixelc++;
        }
    }
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
	/* Free memory index look up tables */
   	free(u_index_array);
	free(i_array);
	free(v_index_array);
	free(j_array);
	
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}

 voidthread transformvolume_error_color(double **Args) {
    double *Bu, *Bv, *dxa, *dya, *ThreadID, *Ox, *Oy, *I1, *I2, *ThreadOut;
    double *Isize_d, *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[2]={0,0};
	int Onumel;
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
   /* Location of pixel which will be come the current pixel */
    double Tlocalx;
    double Tlocaly;
    /* Variables to store 1D index */
    int indexO;
    int indexI;
    /* Grid distance */
    int dx,dy; 
    /* X,Y coordinates of current pixel */
    int x,y;
    /* B-spline variablesl */
    int u_index=0; 
    int v_index=0;
    int i, j;
	/* temporary value */
	double val;
    /* Count number of pixels used for error for normalization */
	int err_pixelc=0;
	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
    /*  B-Spline loop variabels */
    int l,m;
	/* current accumlated image error */
	double err=0;
	/* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    /* RGB index offsets */
    int index_rgb[3]={0,0,0};
  
	/* Split input into variables */
    Bu=Args[0]; Bv=Args[1];
    Isize_d=Args[2]; Osize_d=Args[3];
    ThreadOut=Args[4];
	dxa=Args[5]; dya=Args[6];
    ThreadID=Args[7]; ThreadOffset=(int) ThreadID[0]; 
    Ox=Args[8]; Oy=Args[9];
    I1=Args[10]; I2=Args[11];
	Nthreadsd=Args[12];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; Isize[1] = (int)Isize_d[1];
    Osize[0] = (int)Osize_d[0]; Osize[1] = (int)Osize_d[1]; 
    Onumel=Osize[0]*Osize[1];
	
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
	/*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=ThreadOffset; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}

     /* Make RGB index offsets */
    index_rgb[0]=0; index_rgb[1]=Isize[0]*Isize[1]; index_rgb[2]=2*index_rgb[1];
 
    /*  Loop through all image pixel coordinates */
    for (y=ThreadOffset; y<Isize[1]; y=y+Nthreads)
    {
  		v_index=v_index_array[y]; j=j_array[y];
        for (x=0; x<Isize[0]; x++)
        {
            /* Calculate the index needed to loop up the B-spline values. */
			u_index=u_index_array[x]; i=i_array[x];
            /*  This part calculates the coordinates of the pixel */
            /*  which will be transformed to the current x,y pixel. */
            Tlocalx=0; Tlocaly=0;
            for(l=0; l<4; l++)
            {
				if(((i+l)>=0)&&((i+l)<Osize[0]))
				{
					for(m=0; m<4; m++)
					{    
						 if(((j+m)>=0)&&((j+m)<Osize[1]))
						 {
							  indexO=(i+l)+(j+m)*Osize[0];
							  val=Bu[l+u_index]*Bv[m+v_index];
							  Tlocalx+=val*Ox[indexO];
							  Tlocaly+=val*Oy[indexO];
						 }
					}
				}
            }            
            
            /* interpolate the intensities */
            interpolate_2d_double_color(Ipixel,Tlocalx, Tlocaly, Isize, I1,false,false);
                        
            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            err+=pow2(I2[indexI+index_rgb[0]]-Ipixel[0]);
			err+=pow2(I2[indexI+index_rgb[1]]-Ipixel[1]);
			err+=pow2(I2[indexI+index_rgb[2]]-Ipixel[2]);
			err_pixelc++;
        }
    }
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
	/* Free memory index look up tables */
   	free(u_index_array);
	free(i_array);
	free(v_index_array);
	free(j_array);
	
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}

voidthread transformvolume_gradient_gray(double **Args) {
    double *Bu, *Bv, *dxa, *dya, *ThreadID, *Ox, *Oy, *I1, *I2, *ThreadOut;
    double *Isize_d, *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[2]={0,0};
	int Onumel;
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads offset */
	int ThreadOffset;
	/* index offsets */
    int offset1, offset2;	
    /* Location of pixel which will be come the current pixel */
    double Tlocalx, Tlocaly;
	double TlocalGradx[16];
	double TlocalGrady[16];
	/* Gradient control grids */
	double *OxGrad, *OyGrad;
    /* Variables to store 1D index */
    int indexO, indexI;
    /* Grid distance */
    int dx,dy; 
     /* loop throught the colors r,g,b */
    int rgb=0;
    /* X,Y coordinates of current pixel */
    int x,y;
	/* The control points which influence a pixel */
	int nCPinvolved; 
	int CPinvolved[16];
	int *membergrid;
	/* Step finite difference */
	double step=0.01;
    /* B-spline variablesl */
    int u_index=0, v_index=0, i, j;
	/* Loop variables */
	int ix, iy, jx, jy, k1, k2, k;
	/* index offset in grid gradient */
	int IndexOffGrad[16];
    /* Count number of pixels used for error (gradient) for normalization */
	int err_pixelc=0;
    int *err_grad_pixelc;
    /* temporary value */
	double val;
    /* current pixel error */
    double current_pixelerr;
	/* Look up tables index */
	int *u_index_array, *i_array, *v_index_array, *j_array;
    /*  B-Spline loop variabels */
    int l,m;
    /* current accumlated image error / error gradient */
	double err=0;
	double *err_gradientx, *err_gradienty;
    /* Current voxel/pixel */
    double Ipixel;
    /* Split input into variables */
    Bu=Args[0]; Bv=Args[1];
    Isize_d=Args[2]; Osize_d=Args[3];
    ThreadOut=Args[4];
	dxa=Args[5]; dya=Args[6];
    ThreadID=Args[7]; ThreadOffset=(int) ThreadID[0]; 
    Ox=Args[8]; Oy=Args[9];
    I1=Args[10]; I2=Args[11];
	Nthreadsd=Args[12];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; Isize[1] = (int)Isize_d[1];
    Osize[0] = (int)Osize_d[0]; Osize[1] = (int)Osize_d[1]; 
    Onumel=Osize[0]*Osize[1];
	
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
	/*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=ThreadOffset; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}
	
	/* Initialize gradient error storage */
	err_gradientx=(double*)malloc(Onumel*sizeof(double));
	err_gradienty=(double*)malloc(Onumel*sizeof(double));
	err_grad_pixelc=(int*)malloc(Onumel*sizeof(int));
    for(k1=0; k1<Onumel; k1++) { err_gradientx[k1]=0; err_gradienty[k1]=0; err_grad_pixelc[k1]=0; }
	
	/* Make the grids for the finite difference gradient*/
	OxGrad=(double*)malloc(16*Onumel*sizeof(double));
	OyGrad=(double*)malloc(16*Onumel*sizeof(double));
	membergrid=(int*)malloc(Onumel*sizeof(int));
    
    /* Copy the current grid to all gradient grid arrays */
	for(k1=0; k1<16; k1++) {
		IndexOffGrad[k1]=k1*Onumel; k=IndexOffGrad[k1];
		for(k2=0; k2<Onumel; k2++) {OxGrad[k2+k]=Ox[k2]; OyGrad[k2+k]=Oy[k2]; }
	}		
	/* Move every 4th node in the grid arrays */
	for (iy=0; iy<4; iy++) {
		for (ix=0; ix<4; ix++) {
            k2=ix+iy*4;
			k=IndexOffGrad[k2];
			for (jy=iy; jy<Osize[1]; jy+=4) {
				for (jx=ix; jx<Osize[0]; jx+=4) {
					k1=jx+jy*Osize[0];
                    OxGrad[k1+k]=Ox[k1]+step; OyGrad[k1+k]=Oy[k1]+step;
					membergrid[k1]=k2;
				}
			}
		}
	}

    /*  Loop through all image pixel coordinates */
    for (y=ThreadOffset; y<Isize[1]; y=y+Nthreads)
    {
  		v_index=v_index_array[y]; j=j_array[y];
        for (x=0; x<Isize[0]; x++)
        {
            /* Calculate the index needed to loop up the B-spline values.  */
			u_index=u_index_array[x]; i=i_array[x];
            /*  This part calculates the coordinates of the pixel  */
            /*  which will be transformed to the current x,y pixel.  */
            Tlocalx=0; Tlocaly=0; for(k1=0; k1<16; k1++) { TlocalGradx[k1]=0; TlocalGrady[k1]=0;}
          
            nCPinvolved=0;
     
			for(l=0; l<4; l++)
            {
				if(((i+l)>=0)&&((i+l)<Osize[0]))
				{
					for(m=0; m<4; m++)
					{    
						 if(((j+m)>=0)&&((j+m)<Osize[1]))
						 {
							  indexO=(i+l)+(j+m)*Osize[0];
							  CPinvolved[nCPinvolved]=indexO; nCPinvolved++;
							  val=Bu[l+u_index]*Bv[m+v_index];
							  Tlocalx+=val*Ox[indexO]; Tlocaly+=val*Oy[indexO];
							  for(k1=0; k1<16; k1++) {
								TlocalGradx[k1]+=val*OxGrad[indexO+IndexOffGrad[k1]]; 
								TlocalGrady[k1]+=val*OyGrad[indexO+IndexOffGrad[k1]];
							  }
						 }
					}
				}
            }            
		
            /* Set the current pixel value  */
            indexI=mindex2(x,y,Isize[0]);
            Ipixel = interpolate_2d_double_gray(Tlocalx, Tlocaly, Isize, I1,false,false);
            current_pixelerr=pow2(I2[indexI]-Ipixel);
            err+=current_pixelerr;
	        err_pixelc++;
     		for(k1=0; k1<nCPinvolved; k1++) 
			{
                indexO=CPinvolved[k1];
				k=membergrid[indexO];
                Ipixel = interpolate_2d_double_gray(TlocalGradx[k], Tlocaly, Isize, I1,false,false);
				err_gradientx[indexO]+=pow2(I2[indexI]-Ipixel)-current_pixelerr; 
                Ipixel = interpolate_2d_double_gray(Tlocalx, TlocalGrady[k], Isize, I1,false,false);
				err_gradienty[indexO]+=pow2(I2[indexI]-Ipixel)-current_pixelerr; 
                err_grad_pixelc[indexO]++;
			}
	    }
    }
       
	/* Return error outputs */
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
	offset1=ThreadOffset*(2*Onumel); offset2=offset1+Onumel;
	for(j=0; j<Onumel; j++) {
		ThreadOut[Nthreads+j+offset1]=err_gradientx[j]/(EPS+(double)err_grad_pixelc[j]); 
		ThreadOut[Nthreads+j+offset2]=err_gradienty[j]/(EPS+(double)err_grad_pixelc[j]); 
	}

            
    /* Empty arrays made with Malloc */
	free(err_gradientx);
	free(err_gradienty);
	free(OxGrad);
	free(OyGrad);
	free(u_index_array);
	free(v_index_array);
	free(i_array);
	free(j_array);
		
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}



voidthread transformvolume_gradient_color(double **Args) {
     double *Bu, *Bv, *dxa, *dya, *ThreadID, *Ox, *Oy, *I1, *I2, *ThreadOut;
    double *Isize_d, *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[2]={0,0};
	int Onumel;
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads offset */
	int ThreadOffset;
	/* index offsets */
    int offset1, offset2;	
    /* Location of pixel which will be come the current pixel */
    double Tlocalx, Tlocaly;
	double TlocalGradx[16];
	double TlocalGrady[16];
	/* Gradient control grids */
	double *OxGrad, *OyGrad;
    /* Variables to store 1D index */
    int indexO, indexI;
    /* Grid distance */
    int dx,dy; 
     /* loop throught the colors r,g,b */
    int rgb=0;
    /* X,Y coordinates of current pixel */
    int x,y;
	/* The control points which influence a pixel */
	int nCPinvolved; 
	int CPinvolved[16];
	int *membergrid;
	/* Step finite difference */
	double step=0.01;
    /* B-spline variablesl */
    int u_index=0, v_index=0, i, j;
	/* Loop variables */
	int ix, iy, jx, jy, k1, k2, k;
	/* index offset in grid gradient */
	int IndexOffGrad[16];
    /* Count number of pixels used for error (gradient) for normalization */
	int err_pixelc=0;
    int *err_grad_pixelc;
    /* temporary value */
	double val;
    /* current pixel error */
    double current_pixelerr,current_pixelerr2;
	/* Look up tables index */
	int *u_index_array, *i_array, *v_index_array, *j_array;
    /*  B-Spline loop variabels */
    int l,m;
    /* current accumlated image error / error gradient */
	double err=0;
	double *err_gradientx, *err_gradienty;
	/* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    /* RGB index offsets */
    int index_rgb[3]={0,0,0};
	
    /* Split input into variables */
    Bu=Args[0]; Bv=Args[1];
    Isize_d=Args[2]; Osize_d=Args[3];
    ThreadOut=Args[4];
	dxa=Args[5]; dya=Args[6];
    ThreadID=Args[7]; ThreadOffset=(int) ThreadID[0]; 
    Ox=Args[8]; Oy=Args[9];
    I1=Args[10]; I2=Args[11];
	Nthreadsd=Args[12];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; Isize[1] = (int)Isize_d[1];
    Osize[0] = (int)Osize_d[0]; Osize[1] = (int)Osize_d[1]; 
    Onumel=Osize[0]*Osize[1];
	
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
	/* Make RGB index offsets */
    index_rgb[0]=0; index_rgb[1]=Isize[0]*Isize[1]; index_rgb[2]=2*index_rgb[1];
	
	/*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=ThreadOffset; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}
	
	/* Initialize gradient error storage */
	err_gradientx=(double*)malloc(Onumel*sizeof(double));
	err_gradienty=(double*)malloc(Onumel*sizeof(double));
	err_grad_pixelc=(int*)malloc(Onumel*sizeof(int));
    for(k1=0; k1<Onumel; k1++) { err_gradientx[k1]=0; err_gradienty[k1]=0; err_grad_pixelc[k1]=0; }
	
	/* Make the grids for the finite difference gradient*/
	OxGrad=(double*)malloc(16*Onumel*sizeof(double));
	OyGrad=(double*)malloc(16*Onumel*sizeof(double));
	membergrid=(int*)malloc(Onumel*sizeof(int));
    
    /* Copy the current grid to all gradient grid arrays */
	for(k1=0; k1<16; k1++) {
		IndexOffGrad[k1]=k1*Onumel; k=IndexOffGrad[k1];
		for(k2=0; k2<Onumel; k2++) {OxGrad[k2+k]=Ox[k2]; OyGrad[k2+k]=Oy[k2]; }
	}		
	/* Move every 4th node in the grid arrays */
	for (iy=0; iy<4; iy++) {
		for (ix=0; ix<4; ix++) {
            k2=ix+iy*4;
			k=IndexOffGrad[k2];
			for (jy=iy; jy<Osize[1]; jy+=4) {
				for (jx=ix; jx<Osize[0]; jx+=4) {
					k1=jx+jy*Osize[0];
                    OxGrad[k1+k]=Ox[k1]+step; OyGrad[k1+k]=Oy[k1]+step;
					membergrid[k1]=k2;
				}
			}
		}
	}

    /*  Loop through all image pixel coordinates */
    for (y=ThreadOffset; y<Isize[1]; y=y+Nthreads)
    {
  		v_index=v_index_array[y]; j=j_array[y];
        for (x=0; x<Isize[0]; x++)
        {
            /* Calculate the index needed to loop up the B-spline values.  */
			u_index=u_index_array[x]; i=i_array[x];
            /*  This part calculates the coordinates of the pixel  */
            /*  which will be transformed to the current x,y pixel.  */
            Tlocalx=0; Tlocaly=0; for(k1=0; k1<16; k1++) { TlocalGradx[k1]=0; TlocalGrady[k1]=0;}
          
            nCPinvolved=0;
     
			for(l=0; l<4; l++)
            {
				if(((i+l)>=0)&&((i+l)<Osize[0]))
				{
					for(m=0; m<4; m++)
					{    
						 if(((j+m)>=0)&&((j+m)<Osize[1]))
						 {
							  indexO=(i+l)+(j+m)*Osize[0];
							  CPinvolved[nCPinvolved]=indexO; nCPinvolved++;
							  val=Bu[l+u_index]*Bv[m+v_index];
							  Tlocalx+=val*Ox[indexO]; Tlocaly+=val*Oy[indexO];
							  for(k1=0; k1<16; k1++) {
								TlocalGradx[k1]+=val*OxGrad[indexO+IndexOffGrad[k1]]; 
								TlocalGrady[k1]+=val*OyGrad[indexO+IndexOffGrad[k1]];
							  }
						 }
					}
				}
            }          
			
            /* interpolate the intensities */
            interpolate_2d_double_color(Ipixel,Tlocalx, Tlocaly, Isize, I1,false,false);
                       
            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            current_pixelerr= pow2(I2[indexI+index_rgb[0]]-Ipixel[0]);
			current_pixelerr+=pow2(I2[indexI+index_rgb[1]]-Ipixel[1]);
			current_pixelerr+=pow2(I2[indexI+index_rgb[2]]-Ipixel[2]);
			err+=current_pixelerr;
	        err_pixelc++;
     		for(k1=0; k1<nCPinvolved; k1++) 
			{
                indexO=CPinvolved[k1];
				k=membergrid[indexO];
                interpolate_2d_double_color(Ipixel,TlocalGradx[k], Tlocaly, Isize, I1,false,false);	
				current_pixelerr2= pow2(I2[indexI+index_rgb[0]]-Ipixel[0]);
				current_pixelerr2+=pow2(I2[indexI+index_rgb[1]]-Ipixel[1]);
				current_pixelerr2+=pow2(I2[indexI+index_rgb[2]]-Ipixel[2]);
				err_gradientx[indexO]+=current_pixelerr2-current_pixelerr; 
                interpolate_2d_double_color(Ipixel,Tlocalx, TlocalGrady[k], Isize, I1,false,false);
				current_pixelerr2= pow2(I2[indexI+index_rgb[0]]-Ipixel[0]);
				current_pixelerr2+=pow2(I2[indexI+index_rgb[1]]-Ipixel[1]);
				current_pixelerr2+=pow2(I2[indexI+index_rgb[2]]-Ipixel[2]);
				err_gradienty[indexO]+=current_pixelerr2-current_pixelerr; 
                err_grad_pixelc[indexO]++;
			}
	    }
    }
       
	/* Return error outputs */
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
	offset1=ThreadOffset*(2*Onumel); offset2=offset1+Onumel;
	for(j=0; j<Onumel; j++) {
		ThreadOut[Nthreads+j+offset1]=err_gradientx[j]/(EPS+(double)err_grad_pixelc[j]); 
		ThreadOut[Nthreads+j+offset2]=err_gradienty[j]/(EPS+(double)err_grad_pixelc[j]); 
	}

            
    /* Empty arrays made with Malloc */
	free(err_gradientx);
	free(err_gradienty);
	free(OxGrad);
	free(OyGrad);
	free(u_index_array);
	free(v_index_array);
	free(i_array);
	free(j_array);
		
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}


/* The matlab mex function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    /* Ox and Oy are the grid points */
    /* Zo is the input image */
    /* Zi is the transformed image */
    /* dx and dy are the spacing of the b-spline knots */
    double *Ox,*Oy, *I1, *I2, *dxa, *dya, *E, *Egradient, *ThreadOut;
    mxArray *matlabCallOut[1]={0};
    mxArray *matlabCallIn[1]={0};
    double *Nthreadsd;
    int Nthreads;
	/* Finite difference step size */
	double step=0.01;
    /* index offsets */
	int offset1, offset2;
	
    /* double pointer array to store all needed function variables) */
    double ***ThreadArgs;
    double **ThreadArgs1;
    
	/* Handles to the worker threads */
		ThreadHANDLE *ThreadList;

	
    /* ID of Threads */
    double **ThreadID;              
    double *ThreadID1;
    
	/* Dims outputs */
	const int dims_error[2]={1,1};
	int dims_error_gradient[3]={1,1,2};
	
	/* Size of input image */
    double Isize_d[3]={0,0,0};
    const mwSize *dims;
        
    /* Size of grid */
    mwSize  Osizex, Osizey;
    double Osize_d[2]={0,0};
    
    /* B-spline variablesl */
    double u,v;
    int u_index=0; 
    int v_index=0;
    double *Bu, *Bv;

    /* Loop variables  */
    int i,j;
    
    /* X,Y coordinates of current pixel */
    int x,y;
    /* Grid distance */
    int dx,dy; 
    
  /* Check for proper number of arguments. */
  if(nrhs!=6) {
    mexErrMsgTxt("Six inputs are required.");
  }
    
  /* Get the sizes of the grid */
  Osizex = (mwSize)mxGetM(prhs[0]);  
  Osizey = (mwSize)mxGetN(prhs[0]);
    
  /* Get the sizes of the image */
  dims = mxGetDimensions(prhs[2]);   
  Isize_d[0] = (double)dims[0]; Isize_d[1] = (double)dims[1]; 
  /* Detect if color image */
  if(mxGetNumberOfDimensions(prhs[2])>2) { Isize_d[2]=(double)3; } else { Isize_d[2]=1; }
  
  /* Create image matrix for the Error return argument  */  
  plhs[0] = mxCreateNumericArray(2, dims_error, mxDOUBLE_CLASS, mxREAL);
  if(nlhs>1) 
  {
    dims_error_gradient[0]=Osizex; dims_error_gradient[1]=Osizey;
    /* Error Gradient needed */
	plhs[1] = mxCreateNumericArray(3, dims_error_gradient, mxDOUBLE_CLASS, mxREAL);
  }
    
  /* Assign pointers to each input. */
  Ox=mxGetPr(prhs[0]);
  Oy=mxGetPr(prhs[1]);
  I1=mxGetPr(prhs[2]);
  I2=mxGetPr(prhs[3]);
  dxa=mxGetPr(prhs[4]);
  dya=mxGetPr(prhs[5]);
  
  /* Get the spacing of the uniform b-spline grid */
  dx=(int)dxa[0]; dy=(int)dya[0];
  
  /* Get number of allowed threads */
  mexCallMATLAB(1, matlabCallOut, 0, matlabCallIn, "maxNumCompThreads");
  Nthreadsd=mxGetPr(matlabCallOut[0]);
  Nthreads=(int)Nthreadsd[0];
  /* Reserve room for handles of threads in ThreadList  */

	ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof( ThreadHANDLE ));

  ThreadID = (double **)malloc( Nthreads* sizeof(double *) );
  ThreadArgs = (double ***)malloc( Nthreads* sizeof(double **) );
  if(nlhs==1){	ThreadOut = (double *)malloc(Nthreads* sizeof(double) ); }
  else { ThreadOut = (double *)malloc(Nthreads*(1+Osizex*Osizey*2)*sizeof(double) );  }
  
  /* Assign pointer to output. */
  E = mxGetPr(plhs[0]);
  if(nlhs>1) { Egradient = mxGetPr(plhs[1]); }
  
  /*  Make polynomial look up tables   */
  Bu=malloc(dx*4*sizeof(double));
  Bv=malloc(dy*4*sizeof(double));
  for (x=0; x<dx; x++)
  {
    u=(x/(double)dx)-floor(x/(double)dx);
    Bu[mindex2c(0,x,4,dx)] = pow((1-u),3)/6;
    Bu[mindex2c(1,x,4,dx)] = ( 3*pow(u,3) - 6*pow(u,2) + 4)/6;
    Bu[mindex2c(2,x,4,dx)] = (-3*pow(u,3) + 3*pow(u,2) + 3*u + 1)/6;
    Bu[mindex2c(3,x,4,dx)] = pow(u,3)/6;
  }
  
  for (y=0; y<dy; y++)
  {
    v=(y/(double)dy)-floor(y/(double)dy);
    Bv[mindex2c(0,y,4,dy)] = pow((1-v),3)/6;
    Bv[mindex2c(1,y,4,dy)] = ( 3*pow(v,3) - 6*pow(v,2) + 4)/6;
    Bv[mindex2c(2,y,4,dy)] = (-3*pow(v,3) + 3*pow(v,2) + 3*v + 1)/6;
    Bv[mindex2c(3,y,4,dy)] = pow(v,3)/6;
  }
  
  Osize_d[0]=Osizex;  Osize_d[1]=Osizey;
  
  /* Reserve room for 16 function variables(arrays)   */
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (double *)malloc( 1* sizeof(double) );
    ThreadID1[0]=i;
    ThreadID[i]=ThreadID1;  

    /*  Make Thread Structure  */
    ThreadArgs1 = (double **)malloc( 13 * sizeof( double * ) );  
    ThreadArgs1[0]=Bu;
    ThreadArgs1[1]=Bv;
    ThreadArgs1[2]=Isize_d;
    ThreadArgs1[3]=Osize_d;
    ThreadArgs1[4]=ThreadOut;
    ThreadArgs1[5]=dxa;
    ThreadArgs1[6]=dya;
    ThreadArgs1[7]=ThreadID[i];
    ThreadArgs1[8]=Ox;
    ThreadArgs1[9]=Oy;
    ThreadArgs1[10]=I1;
	ThreadArgs1[11]=I2;
	ThreadArgs1[12]=Nthreadsd;
    ThreadArgs[i]=ThreadArgs1;
  

		if(nlhs==1){
			if(Isize_d[2]>1) {
				StartThread(ThreadList[i], &transformvolume_error_color, ThreadArgs[i])
			}
			else {
				StartThread(ThreadList[i], &transformvolume_error_gray, ThreadArgs[i])

			}
		}
		else{
			if(Isize_d[2]>1) {
						StartThread(ThreadList[i], &transformvolume_gradient_color, ThreadArgs[i])
			}
			else {
						StartThread(ThreadList[i], &transformvolume_gradient_gray, ThreadArgs[i])
			}
		}

  }
  
 for (i=0; i<Nthreads; i++) { WaitForThreadFinish(ThreadList[i]); }


  /* Add accumlated error of all threads */
  E[0]=0; for (i=0; i<Nthreads; i++) { E[0]+=ThreadOut[i]; } E[0]/=Nthreads;

  if(nlhs>1)
  {
		for (i=0; i<Nthreads; i++) 
		{ 
			offset1=i*(2*Osizex*Osizey);
            offset2=offset1+(Osizex*Osizey);
			for(j=0; j<Osizex*Osizey; j++)
			{
				Egradient[j]+=ThreadOut[Nthreads+j+offset1]/step;
				Egradient[j+(Osizex*Osizey)]+=ThreadOut[Nthreads+j+offset2]/step;
			}
		}

        for(j=0; j<2*Osizex*Osizey; j++)
    	{
            Egradient[j]/=Nthreads;
        }
  }
  
  for (i=0; i<Nthreads; i++) 
  { 
    free(ThreadArgs[i]);
    free(ThreadID[i]);
  }

  free(ThreadArgs);
  free(ThreadID );
  free(ThreadList);
  
  free(Bu);
  free(Bv);
}
        

