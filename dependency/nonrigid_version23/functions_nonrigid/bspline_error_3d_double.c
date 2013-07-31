#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"
  

/* 3D Bspline transformation grid function
 * function [Error,ErrorGradient]=bspline_error_3d_double(Ox,Oy,Iz,I1,I2,dx,dy,dz)
 * 
 * Ox, Oy, Oz are the grid points coordinates
 * Vin is input image, Vout the transformed output image
 * dx, dy and dz are the spacing of the b-spline knots
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

voidthread transformvolume_error(double **Args) {
    double *Bu, *Bv, *Bw, *ThreadOut;
    double *dxa, *dya, *dza, *ThreadID, *Ox, *Oy, *Oz, *I1, *I2;
    double *Isize_d;
    double *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[3]={0,0,0};
    int Onumel;
	double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    /* Location of pixel which will be come the current pixel */
    double Tlocalx, Tlocaly, Tlocalz;
    /* Variables to store 1D index */
    int indexO;
    int indexI;
	/* Grid distance */
    int dx,dy,dz; 
    /* X,Y,Z coordinates of current pixel */
    int x,y,z;
    /* B-spline variables */
    int u_index=0, v_index=0, w_index=0, i, j, k;
	/* temporary value */
	double val;
	/* Count number of pixels used for error for normalization */
	int err_pixelc=0;
	double err=0;
    /* Current voxel/pixel */
    double Ipixel;
    /* Split input into variables */
   	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
	int *w_index_array, *k_array;
	/*  B-Spline loop variabels */
    int l,m,n;
    /* Split input into variables */
    Bu=Args[0]; Bv=Args[1]; Bw=Args[2];
    Isize_d=Args[3]; Osize_d=Args[4];
    ThreadOut=Args[5];
    dxa=Args[6]; dya=Args[7]; dza=Args[8];
    ThreadID=Args[9]; ThreadOffset=(int) ThreadID[0];
    Ox=Args[10]; Oy=Args[11]; Oz=Args[12];
    I1=Args[13];  I2=Args[14];
    Nthreadsd=Args[15];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 
    Osize[0] = (int)Osize_d[0]; 
    Osize[1] = (int)Osize_d[1]; 
    Osize[2] = (int)Osize_d[2]; 
    Onumel = Osize[0]*Osize[1]*Osize[2];
	 
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0]; dz=(int)dza[0];

    /*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	w_index_array= (int*)malloc(Isize[2]* sizeof(int));
	k_array= (int*)malloc(Isize[2]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=0; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}
	for (z=ThreadOffset; z<Isize[2]; z++) {
	    w_index_array[z]=(z%dz)*4; k_array[z]=(int)floor((double)z/dz); 
	}

	/*  Loop through all image pixel coordinates */
     for (z=ThreadOffset; z<Isize[2]; z=z+Nthreads)
     {
		w_index=w_index_array[z]; k=k_array[z];
	    for (y=0; y<Isize[1]; y++)
        {
			v_index=v_index_array[y]; j=j_array[y];
            for (x=0; x<Isize[0]; x++)
            {
				u_index=u_index_array[x]; i=i_array[x];
                /*  This part calculates the coordinates of the pixel */
                /*  which will be transformed to the current x,y pixel. */
                Tlocalx=0; Tlocaly=0; Tlocalz=0;
                for(l=0; l<4; l++)
                {
                    if(((i+l)>=0)&&((i+l)<Osize[0]))
                    {
                        for(m=0; m<4; m++)
                        {   
                            if(((j+m)>=0)&&((j+m)<Osize[1]))
                            {
                                for(n=0; n<4; n++)
                                {       
                                    if(((k+n)>=0)&&((k+n)<Osize[2]))
                                    {
                                         indexO=(i+l)+(j+m)*Osize[0]+(k+n)*Osize[0]*Osize[1];
                                       	 val=Bu[l+u_index]*Bv[m+v_index]*Bw[n+w_index];
										 Tlocalx+=val*Ox[indexO];
                                         Tlocaly+=val*Oy[indexO]; 
                                         Tlocalz+=val*Oz[indexO];
                                    }
                                }
                            }
                        }
                    }
                }            
                /* Set the current pixel value */
                indexI=x+y*Isize[0]+z*Isize[0]*Isize[1];
			    Ipixel = interpolate_3d_double_gray(Tlocalx, Tlocaly, Tlocalz, Isize, I1,false,false); 
			    err+=pow2(I2[indexI]-Ipixel);
				err_pixelc++;
            }
        }
    }    
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
   
    /* Free memory index look up tables */
   	free(u_index_array);
	free(i_array);
	free(v_index_array);
	free(j_array);
	free(w_index_array);
	free(k_array);
	
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}

voidthread transformvolume_gradient(double **Args) {
    double *Bu, *Bv, *Bw, *dxa, *dya, *dza, *ThreadID, *Ox, *Oy, *Oz, *I1, *I2, *ThreadOut;
    double *Isize_d, *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[3]={0,0,0};
	int Onumel;
    double *Nthreadsd;
    int Nthreads;
    /* Multiple thread offset */
    int ThreadOffset;
	/* index offsets */
	int offset1, offset2, offset3;	
    /* Location of pixel which will be come the current pixel */
    double Tlocalx, Tlocaly, Tlocalz;
	double TlocalGradx[64];
	double TlocalGrady[64];
	double TlocalGradz[64];
	/* Gradient control grids */
	double *OxGrad, *OyGrad, *OzGrad;
    /* Variables to store 1D index */
    int indexO, indexI;
    /* Grid distance */
    int dx,dy,dz; 
    /* X,Y coordinates of current pixel */
    int x,y,z;
	/* The control points which influence a pixel */
	int nCPinvolved; 
	int CPinvolved[64];
	int *membergrid;
	/* Step finite difference */
	double step=0.01;
    /* B-spline variablesl */
    int u_index=0, v_index=0,  w_index=0, i, j, k;
	/* Loop variables */
	int ix, iy, iz, jx, jy, jz, k1, k2, k3;
	/* index offset in grid gradient */
	int IndexOffGrad[64];
    /* Count number of pixels used for error (gradient) for normalization */
	int err_pixelc=0;
    int *err_grad_pixelc;
    /* temporary value */
	double val;
    /* current pixel error */
    double current_pixelerr;
	/* Look up tables index */
	int *u_index_array, *i_array, *v_index_array, *j_array, *w_index_array, *k_array;
    /*  B-Spline loop variabels */
    int l,m, n;
    /* current accumlated image error / error gradient */
	double err=0;
	double *err_gradientx, *err_gradienty, *err_gradientz;
    /* Current voxel/pixel */
    double Ipixel;
	
	/* Split input into variables */
    Bu=Args[0]; Bv=Args[1]; Bw=Args[2];
    Isize_d=Args[3]; Osize_d=Args[4];
    ThreadOut=Args[5];
    dxa=Args[6]; dya=Args[7]; dza=Args[8];
    ThreadID=Args[9]; ThreadOffset=(int) ThreadID[0];
    Ox=Args[10]; Oy=Args[11]; Oz=Args[12];
    I1=Args[13];  I2=Args[14];
    Nthreadsd=Args[15];  Nthreads=(int)Nthreadsd[0];

    Isize[0] = (int)Isize_d[0]; Isize[1] = (int)Isize_d[1]; Isize[2] = (int)Isize_d[2]; 
    Osize[0] = (int)Osize_d[0]; Osize[1] = (int)Osize_d[1]; Osize[2] = (int)Osize_d[2]; 
    Onumel = Osize[0]*Osize[1]*Osize[2];
	 
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0]; dz=(int)dza[0];

    /*  Calculate the indexs need to look up the B-spline values. */
	u_index_array= (int*)malloc(Isize[0]* sizeof(int));
	i_array= (int*)malloc(Isize[0]* sizeof(int));
	v_index_array= (int*)malloc(Isize[1]* sizeof(int));
	j_array= (int*)malloc(Isize[1]* sizeof(int));
	w_index_array= (int*)malloc(Isize[2]* sizeof(int));
	k_array= (int*)malloc(Isize[2]* sizeof(int));
	for (x=0; x<Isize[0]; x++) {
        u_index_array[x]=(x%dx)*4; /* Already multiplied by 4, because it specifies the y dimension */
        i_array[x]=(int)floor((double)x/dx); /*  (first row outside image against boundary artefacts)  */
	}
	for (y=0; y<Isize[1]; y++) {
	    v_index_array[y]=(y%dy)*4; j_array[y]=(int)floor((double)y/dy);
	}
	for (z=ThreadOffset; z<Isize[2]; z++) {
	    w_index_array[z]=(z%dz)*4; k_array[z]=(int)floor((double)z/dz); 
	}
	
	/* Initialize gradient error storage */
	err_gradientx=(double*)malloc(Onumel*sizeof(double));
	err_gradienty=(double*)malloc(Onumel*sizeof(double));
	err_gradientz=(double*)malloc(Onumel*sizeof(double));
	
	err_grad_pixelc=(int*)malloc(Onumel*sizeof(int));
    for(k1=0; k1<Onumel; k1++)	{ err_gradientx[k1]=0; err_gradienty[k1]=0; err_gradientz[k1]=0; err_grad_pixelc[k1]=0; }
	
	/* Make the grids for the finite difference gradient*/
	OxGrad=(double*)malloc(64*Onumel*sizeof(double));
	OyGrad=(double*)malloc(64*Onumel*sizeof(double));
	OzGrad=(double*)malloc(64*Onumel*sizeof(double));
	membergrid=(int*)malloc(Onumel*sizeof(int));
    
    /* Copy the current grid to all gradient grid arrays */
	for(k1=0; k1<64; k1++) {
		IndexOffGrad[k1]=k1*Onumel;
		k3=IndexOffGrad[k1];
		for(k2=0; k2<Onumel; k2++) 
		{
			OxGrad[k2+k3]=Ox[k2]; 
			OyGrad[k2+k3]=Oy[k2]; 
			OzGrad[k2+k3]=Oz[k2]; 
		}
	}		
	
	/* Move every 4th node in the grid arrays */
	for (iz=0; iz<4; iz++) {
		for (iy=0; iy<4; iy++) {
			for (ix=0; ix<4; ix++) {
				k2=ix+iy*4+iz*16;
				k3=IndexOffGrad[k2];
				for (jz=iz; jz<Osize[2]; jz+=4) {
					for (jy=iy; jy<Osize[1]; jy+=4) {
						for (jx=ix; jx<Osize[0]; jx+=4) {
							k1=jx+jy*Osize[0]+jz*Osize[0]*Osize[1];
							OxGrad[k1+k3]=Ox[k1]+step;
							OyGrad[k1+k3]=Oy[k1]+step;
							OzGrad[k1+k3]=Oz[k1]+step;
							membergrid[k1]=k2;
						}
					}
				}
			}
		}
	}

	/*  Loop through all image pixel coordinates */
     for (z=ThreadOffset; z<Isize[2]; z=z+Nthreads)
     {
		w_index=w_index_array[z]; k=k_array[z];
	    for (y=0; y<Isize[1]; y++)
        {
			v_index=v_index_array[y]; j=j_array[y];
            for (x=0; x<Isize[0]; x++)
            {
				u_index=u_index_array[x]; i=i_array[x];
                /*  This part calculates the coordinates of the pixel */
                /*  which will be transformed to the current x,y pixel. */
                Tlocalx=0; Tlocaly=0; Tlocalz=0; nCPinvolved=0;
				for(k1=0; k1<64; k1++) { TlocalGradx[k1]=0; TlocalGrady[k1]=0; TlocalGradz[k1]=0;}
				
                for(l=0; l<4; l++)
                {
                    if(((i+l)>=0)&&((i+l)<Osize[0]))
                    {
                        for(m=0; m<4; m++)
                        {   
                            if(((j+m)>=0)&&((j+m)<Osize[1]))
                            {
                                for(n=0; n<4; n++)
                                {       
                                    if(((k+n)>=0)&&((k+n)<Osize[2]))
                                    {
                                        indexO=(i+l)+(j+m)*Osize[0]+(k+n)*Osize[0]*Osize[1];
                                       	val=Bu[l+u_index]*Bv[m+v_index]*Bw[n+w_index];
										Tlocalx+=val*Ox[indexO];
                                        Tlocaly+=val*Oy[indexO]; 
                                        Tlocalz+=val*Oz[indexO];
										CPinvolved[nCPinvolved]=indexO; nCPinvolved++;
										for(k1=0; k1<64; k1++) {
											TlocalGradx[k1]+=val*OxGrad[indexO+IndexOffGrad[k1]]; 
											TlocalGrady[k1]+=val*OyGrad[indexO+IndexOffGrad[k1]];
											TlocalGradz[k1]+=val*OzGrad[indexO+IndexOffGrad[k1]];
										}
									}
								}
							}
						}
					}
				}
            
				/* Set the current pixel value  */
				  /* Set the current pixel value */
                indexI=x+y*Isize[0]+z*Isize[0]*Isize[1];
			    Ipixel = interpolate_3d_double_gray(Tlocalx, Tlocaly, Tlocalz, Isize, I1,false,false); 
			    current_pixelerr=pow2(I2[indexI]-Ipixel);
				err+=current_pixelerr;
				err_pixelc++;
				for(k1=0; k1<nCPinvolved; k1++) 
				{
					indexO=CPinvolved[k1];
					k3=membergrid[indexO];
					Ipixel = interpolate_3d_double_gray(TlocalGradx[k3], Tlocaly, Tlocalz, Isize, I1,false,false); 
					err_gradientx[indexO]+=pow2(I2[indexI]-Ipixel)-current_pixelerr; 
					Ipixel = interpolate_3d_double_gray(Tlocalx, TlocalGrady[k3], Tlocalz, Isize, I1,false,false); 
					err_gradienty[indexO]+=pow2(I2[indexI]-Ipixel)-current_pixelerr; 
					Ipixel = interpolate_3d_double_gray(Tlocalx, Tlocaly, TlocalGradz[k3], Isize, I1,false,false); 
					err_gradientz[indexO]+=pow2(I2[indexI]-Ipixel)-current_pixelerr; 
					err_grad_pixelc[indexO]++;
				}
			}
	    }
    }
       
	/* Return error outputs */
	ThreadOut[ThreadOffset]=err/(EPS+(double)err_pixelc);
	
	offset1=ThreadOffset*(3*Onumel); 
	offset2=offset1+Onumel;
	offset3=offset2+Onumel;
	for(j=0; j<Onumel; j++) {
		ThreadOut[Nthreads+j+offset1]=err_gradientx[j]/(EPS+(double)err_grad_pixelc[j]); 
		ThreadOut[Nthreads+j+offset2]=err_gradienty[j]/(EPS+(double)err_grad_pixelc[j]); 
		ThreadOut[Nthreads+j+offset3]=err_gradientz[j]/(EPS+(double)err_grad_pixelc[j]); 
	}


            
    /* Empty arrays made with Malloc */
	free(err_gradientx);
	free(err_gradienty);
	free(err_gradientz);
	free(OxGrad);
	free(OyGrad);
	free(OzGrad);
	free(u_index_array);
	free(v_index_array);
	free(w_index_array);
	free(i_array);
	free(j_array);
	free(k_array);
		
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

    /* nx and ny are the number of grid points (inside the image) */
    double *Ox,*Oy,*Oz,*I1,*I2,*dxa, *dya,*dza, *E, *Egradient, *ThreadOut;
    mxArray *matlabCallOut[1]={0};
    mxArray *matlabCallIn[1]={0};
    double *Nthreadsd;
    int Nthreads;
	/* Finite difference step size */
	double step=0.01;
	/* index offsets */
	int offset1, offset2, offset3;
	/* Dims outputs */
	const int dims_error[2]={1,1};
	int dims_error_gradient[4]={1,1,1,3};
	/* double pointer array to store all needed function variables) */
    double ***ThreadArgs;
    double **ThreadArgs1;
    /* Handles to the worker threads */
		ThreadHANDLE *ThreadList;
    
    /* ID of Threads */
    double **ThreadID;              
    double *ThreadID1;
    
    /* Size of input image */
    mwSize  Isizex, Isizey, Isizez;
    double Isize_d[3]={0,0,0};
    const mwSize *dims;

    /* Size of grid */
    mwSize  Osizex, Osizey, Osizez;
	int Onumel;
    double Osize_d[3]={0,0,0};
   
    /* B-spline variablesl */
    double u,v,w;
    int u_index=0; 
    int v_index=0;
    int w_index=0;
    
    double *Bu, *Bv, *Bw;
    
    /* Loop variables  */
    int i,j;
	/* Grid distance */
    int dx,dy,dz; 
    /* X,Y,Z coordinates of current pixel */
    int x,y,z;
        
  /* Check for proper number of arguments. */
  if(nrhs!=8) {
    mexErrMsgTxt("Eight inputs are required.");
  }
 
  /* Get the sizes of the grid */
  dims = mxGetDimensions(prhs[0]);   
  Osizex = dims[0]; 
  Osizey = dims[1];
  Osizez = dims[2];
  Onumel = Osizex*Osizey*Osizez;
   
  /* Create image matrix for the return arguments with the size of input image  */  
  dims = mxGetDimensions(prhs[3]);  
  Isizex = dims[0]; 
  Isizey = dims[1];
  Isizez = dims[2];
  
  /* Create image matrix for the Error return argument  */  
  plhs[0] = mxCreateNumericArray(2, dims_error, mxDOUBLE_CLASS, mxREAL);
  if(nlhs>1) 
  {
    dims_error_gradient[0]=Osizex; dims_error_gradient[1]=Osizey; dims_error_gradient[2]=Osizez;
    /* Error Gradient needed */
	plhs[1] = mxCreateNumericArray(4, dims_error_gradient, mxDOUBLE_CLASS, mxREAL);
  }
  
  /* Assign pointers to each input. */
  Ox=(double *)mxGetData(prhs[0]);
  Oy=(double *)mxGetData(prhs[1]);
  Oz=(double *)mxGetData(prhs[2]);
  I1=(double *)mxGetData(prhs[3]);
  I2=(double *)mxGetData(prhs[4]);
  dxa=(double *)mxGetData(prhs[5]);
  dya=(double *)mxGetData(prhs[6]);
  dza=(double *)mxGetData(prhs[7]);
   
  /* Get the spacing of the uniform b-spline grid */
  dx=(int)dxa[0]; dy=(int)dya[0]; dz=(int)dza[0]; 
  
  /* Get number of allowed threads */
  mexCallMATLAB(1, matlabCallOut, 0, matlabCallIn, "maxNumCompThreads");
  Nthreadsd=mxGetPr(matlabCallOut[0]);
  Nthreads=(int)Nthreadsd[0];
  
    /* Reserve room for handles of threads in ThreadList  */
		ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof( ThreadHANDLE ));

  ThreadID = (double **)malloc( Nthreads* sizeof(double *) );
  ThreadArgs = (double ***)malloc( Nthreads* sizeof(double **) );
  if(nlhs==1){	ThreadOut = (double *)malloc(Nthreads* sizeof(double) ); }
  else { ThreadOut = (double *)malloc(Nthreads*(1+Onumel*3)*sizeof(double) );  }

  /* Assign pointer to output. */
  E = mxGetPr(plhs[0]);
  if(nlhs>1) { Egradient = mxGetPr(plhs[1]); }
  
   /*  Make polynomial look up tables   */
  Bu=malloc(dx*4*sizeof(double));
  Bv=malloc(dy*4*sizeof(double));
  Bw=malloc(dz*4*sizeof(double));
  for (x=0; x<dx; x++)
  {
    u=((double)x/(double)dx)-floor((double)x/(double)dx);
    Bu[mindex2(0,x,4)] = (double)pow((1-u),3)/6;
    Bu[mindex2(1,x,4)] = (double)( 3*pow(u,3) - 6*pow(u,2) + 4)/6;
    Bu[mindex2(2,x,4)] = (double)(-3*pow(u,3) + 3*pow(u,2) + 3*u + 1)/6;
    Bu[mindex2(3,x,4)] = (double)pow(u,3)/6;
  }
  
  for (y=0; y<dy; y++)
  {
    v=((double)y/(double)dy)-floor((double)y/(double)dy);
    Bv[mindex2(0,y,4)] = (double)pow((1-v),3)/6;
    Bv[mindex2(1,y,4)] = (double)( 3*pow(v,3) - 6*pow(v,2) + 4)/6;
    Bv[mindex2(2,y,4)] = (double)(-3*pow(v,3) + 3*pow(v,2) + 3*v + 1)/6;
    Bv[mindex2(3,y,4)] = (double)pow(v,3)/6;
  }

  for (z=0; z<dz; z++)
  {
    w=((double)z/(double)dz)-floor((double)z/(double)dz);
    Bw[mindex2(0,z,4)] = (double)pow((1-w),3)/6;
    Bw[mindex2(1,z,4)] = (double)( 3*pow(w,3) - 6*pow(w,2) + 4)/6;
    Bw[mindex2(2,z,4)] = (double)(-3*pow(w,3) + 3*pow(w,2) + 3*w + 1)/6;
    Bw[mindex2(3,z,4)] = (double)pow(w,3)/6;
  }

  Isize_d[0]=(double)Isizex;  Isize_d[1]=(double)Isizey; Isize_d[2]=(double)Isizez;
  Osize_d[0]=(double)Osizex;  Osize_d[1]=(double)Osizey; Osize_d[2]=(double)Osizez;
  
 /* Reserve room for 16 function variables(arrays)   */
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (double *)malloc( 1* sizeof(double) );
    ThreadID1[0]=(double)i;
    ThreadID[i]=ThreadID1;  
	
    /*  Make Thread Structure  */
    ThreadArgs1 = (double **)malloc( 16 * sizeof( double * ) );  
    ThreadArgs1[0]=Bu;
    ThreadArgs1[1]=Bv;
	ThreadArgs1[2]=Bw;
	ThreadArgs1[3]=Isize_d;
    ThreadArgs1[4]=Osize_d;
    ThreadArgs1[5]=ThreadOut;
    ThreadArgs1[6]=dxa;
    ThreadArgs1[7]=dya;
    ThreadArgs1[8]=dza;
    ThreadArgs1[9]=ThreadID[i];
    ThreadArgs1[10]=Ox;
    ThreadArgs1[11]=Oy;
	ThreadArgs1[12]=Oz;
	ThreadArgs1[13]=I1;
	ThreadArgs1[14]=I2;
	ThreadArgs1[15]=Nthreadsd;
    ThreadArgs[i]=ThreadArgs1;
       

		if(nlhs==1){
		   StartThread(ThreadList[i], &transformvolume_error, ThreadArgs[i])
		}
		else{
			StartThread(ThreadList[i], &transformvolume_gradient, ThreadArgs[i])
		}
  }
  
 for (i=0; i<Nthreads; i++) { WaitForThreadFinish(ThreadList[i]); }


  /* Add accumlated error of all threads */
  E[0]=0; for (i=0; i<Nthreads; i++) { E[0]+=ThreadOut[i]; } E[0]/=Nthreads;

  if(nlhs>1)
  {
		for (i=0; i<Nthreads; i++) 
		{ 
			offset1=i*(3*Onumel);
            offset2=offset1+Onumel;
			offset3=offset2+Onumel;
			for(j=0; j<Onumel; j++)
			{
				Egradient[j]+=ThreadOut[Nthreads+j+offset1]/step;
				Egradient[j+Onumel]+=ThreadOut[Nthreads+j+offset2]/step;
				Egradient[j+2*Onumel]+=ThreadOut[Nthreads+j+offset3]/step;
			}
		}
        for(j=0; j<3*Onumel; j++)
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
  free(Bw);
  
}
        

