#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"

/* 3D Bspline transformation grid function
 * function [Vout,Tx,Ty,Tz]t=bspline_transform_3d_float(Ox,Oy,Oz,Vin,dx,dy,dz,mode)
 * 
 * Ox, Oy, Oz are the grid points coordinates
 * Vin is input image, Vout the transformed output image
 * dx, dy and dz are the spacing of the b-spline knots
 * mode: If 0: linear interpolation and outside pixels set to nearest pixel
 *          1: linear interpolation and outside pixels set to zero
 *          2: cubic interpolation and outsite pixels set to nearest pixel
 *          3: cubic interpolation and outside pixels set to zero
 *
 * Iout: The transformed image
 * Tx: The transformation field in x direction
 * Ty: The transformation field in y direction
 * Tz: The transformation field in y direction
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


voidthread transformvolume(double **Args) {
    double *Bu, *Bv, *Bw, *Iout, *Tx, *Ty, *Tz, *moded;
    double *dxa, *dya, *dza, *ThreadID, *Ox, *Oy, *Oz, *Iin;
    double *Isize_d;
    double *Osize_d;
    double *nlhs_d;
    int mode=0;
    int Isize[3]={0,0,0};
    int Osize[3]={0,0,0};
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    /* Location of pixel which will be come the current pixel */
    double Tlocalx;
    double Tlocaly;
    double Tlocalz;
    /* Cubic and outside black booleans */
    bool black, cubic;
    /* Variables to store 1D index */
    int indexO;
    int indexI;
    /* Grid distance */
    int dx,dy,dz; 
    /* X,Y,Z coordinates of current pixel */
    int x,y,z;
    /* B-spline variables */
    int u_index=0, v_index=0, w_index=0;
    int i, j, k;
	/* temporary value */
	double val;
   	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
	int *w_index_array, *k_array;
	/*  B-Spline loop variabels */
    int l,m,n;
    int nlhs=0;
    /* Split input into variables */
    Bu=Args[0];
    Bv=Args[1];
    Bw=Args[2];
    Isize_d=Args[3];
    Osize_d=Args[4];
    Iout=Args[5];
    Tx=Args[6];
    Ty=Args[7];
    Tz=Args[8];
    dxa=Args[9];
    dya=Args[10];
    dza=Args[11];
    ThreadID=Args[12];
    Ox=Args[13];
    Oy=Args[14];
    Oz=Args[15];
    Iin=Args[16];
    nlhs_d=Args[17];
    moded=Args[18]; mode=(int) moded[0];
    Nthreadsd=Args[19];  Nthreads=(int)Nthreadsd[0];
       
    if(mode==0||mode==2){ black = false; } else { black = true; }
    if(mode==0||mode==1){ cubic = false; } else { cubic = true; }
	
    nlhs=(int)nlhs_d[0];
    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 
    Osize[0] = (int)Osize_d[0]; 
    Osize[1] = (int)Osize_d[1]; 
    Osize[2] = (int)Osize_d[2]; 
    
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0]; dz=(int)dza[0];
    
    ThreadOffset=(int) ThreadID[0];

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

                Iout[indexI]=interpolate_3d_double_gray(Tlocalx, Tlocaly, Tlocalz, Isize, Iin,cubic,black); 
            
                /*  Store transformation field */
                if(nlhs>1) { Tx[indexI]=Tlocalx-(double)x; }
                if(nlhs>2) { Ty[indexI]=Tlocaly-(double)y; }
                if(nlhs>3) { Tz[indexI]=Tlocalz-(double)z; }
            }
        }
    }    
   
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

/* The matlab mex function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    /* Ox and Oy are the grid points */
    /* Zo is the input image */
    /* Zi is the transformed image */

    /* nx and ny are the number of grid points (inside the image) */
    double *Ox,*Oy,*Oz,*Iin, *dxa, *dya,*dza,*Iout, *Tx,*Ty,*Tz,*moded;
    mxArray *matlabCallOut[1]={0};
    mxArray *matlabCallIn[1]={0};
    double *Nthreadsd;
    int Nthreads;
	/* double pointer array to store all needed function variables) */
    double ***ThreadArgs;
    double **ThreadArgs1;
    /* Handles to the worker threads */
		ThreadHANDLE *ThreadList;
 

    /* ID of Threads */
    double **ThreadID;              
    double *ThreadID1;
    
    double nlhs_d[1]={0};
    
    /* Size of input image */
    mwSize  Isizex, Isizey, Isizez;
    double Isize_d[3]={0,0,0};
    const mwSize *dims;


    /* Size of grid */
    mwSize  Osizex, Osizey, Osizez;
    double Osize_d[3]={0,0,0};
   
    /* B-spline variablesl */
    double u,v,w;
    int u_index=0; 
    int v_index=0;
    int w_index=0;
    
    double *Bu, *Bv, *Bw;
    
	/* Loop variable  */
	int i;
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
  /* Create image matrix for the return arguments with the size of input image  */  
  dims = mxGetDimensions(prhs[3]);  
  Isizex = dims[0]; 
  Isizey = dims[1];
  Isizez = dims[2];
  

  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 
  if(nlhs>1) { plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); }
  if(nlhs>2) { plhs[2] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); }
  if(nlhs>3) { plhs[3] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); }
  
  
  /* Assign pointers to each input. */
  Ox=(double *)mxGetData(prhs[0]);
  Oy=(double *)mxGetData(prhs[1]);
  Oz=(double *)mxGetData(prhs[2]);
  Iin=(double *)mxGetData(prhs[3]);
  dxa=(double *)mxGetData(prhs[4]);

  dya=(double *)mxGetData(prhs[5]);
  dza=(double *)mxGetData(prhs[6]);
  moded=(double *)mxGetData(prhs[7]);
   
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
  

  /* Assign pointer to output. */
  Iout = (double *)mxGetData(plhs[0]);
  if(nlhs>1) { Tx =(double *)mxGetData(plhs[1]); }
  if(nlhs>2) { Ty =(double *)mxGetData(plhs[2]); }
  if(nlhs>3) { Tz =(double *)mxGetData(plhs[3]); }
  
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
  
  nlhs_d[0]=(double)nlhs;
  
 /* Reserve room for 16 function variables(arrays)   */
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (double *)malloc( 1* sizeof(double) );
    ThreadID1[0]=(double)i;
    ThreadID[i]=ThreadID1;  
	
    /*  Make Thread Structure  */
    ThreadArgs1 = (double **)malloc( 20* sizeof( double * ) );  
	ThreadArgs1[0]=Bu;
	ThreadArgs1[1]=Bv;
	ThreadArgs1[2]=Bw;
	ThreadArgs1[3]=Isize_d;
	ThreadArgs1[4]=Osize_d;
	ThreadArgs1[5]=Iout;
	ThreadArgs1[6]=Tx;
	ThreadArgs1[7]=Ty;
	ThreadArgs1[8]=Tz;
	ThreadArgs1[9]=dxa;
	ThreadArgs1[10]=dya;
	ThreadArgs1[11]=dza;
	ThreadArgs1[12]=ThreadID[i];
	ThreadArgs1[13]=Ox;
	ThreadArgs1[14]=Oy;
	ThreadArgs1[15]=Oz;
	ThreadArgs1[16]=Iin;
	ThreadArgs1[17]=nlhs_d;
	ThreadArgs1[18]=moded;
	ThreadArgs1[19]=Nthreadsd;
    ThreadArgs[i]=ThreadArgs1;

	 StartThread(ThreadList[i], &transformvolume, ThreadArgs[i])

  }

for (i=0; i<Nthreads; i++) { WaitForThreadFinish(ThreadList[i]); }


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
        

