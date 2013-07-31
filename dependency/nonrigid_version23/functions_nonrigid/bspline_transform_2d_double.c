#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"

/* Bspline transformation grid function
 * function [Iout,Tx,Ty]=bspline_transform_2d_double(Ox,Oy,Iin,dx,dy,mode)
 * 
 * Ox, Oy are the grid points coordinates
 * Iin is input image, Iout the transformed output image
 * dx and dy are the spacing of the b-spline knots
 * mode: If 0: linear interpolation and outside pixels set to nearest pixel
 *          1: linear interpolation and outside pixels set to zero
 *          2: cubic interpolation and outsite pixels set to nearest pixel
 *          3: cubic interpolation and outside pixels set to zero
 *
 * Iout: The transformed image
 * Tx: The transformation field in x direction
 * Ty: The transformation field in y direction
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

voidthread transformvolume_color(double **Args) {
    double *Bu, *Bv, *Iout, *Tx, *Ty, *dxa, *dya, *ThreadID, *Ox, *Oy, *Iin, *moded;
    double *Isize_d;
    double *Osize_d;
    double *nlhs_d;
    int Isize[3]={0,0,0};
    int mode=0;
    int Osize[2]={0,0};
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
    
     /* loop throught the colors r,g,b */
    int rgb=0;
    
    /* X,Y coordinates of current pixel */
    int x,y;
    
    /* B-spline variablesl */
    int u_index=0; 
    int v_index=0;
    int i, j;
    
    /* temporary value */
	double val;
	
	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
	
    /*  B-Spline loop variabels */
    int l,m;
    int nlhs=0;
    
    /* Cubic and outside black booleans */
    bool black, cubic;

    /* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    
    /* RGB index offsets */
    int index_rgb[3]={0,0,0};
            
    /* Split input into variables */
    Bu=Args[0];
    Bv=Args[1];
    Isize_d=Args[2];
    Osize_d=Args[3];
    Iout=Args[4];
    Tx=Args[5];
    Ty=Args[6];
    dxa=Args[7];
    dya=Args[8];
    ThreadID=Args[9];
    Ox=Args[10];
    Oy=Args[11];
    Iin=Args[12];
    nlhs_d=Args[13];
    moded=Args[14]; mode=(int) moded[0];
    Nthreadsd=Args[15];  Nthreads=(int)Nthreadsd[0];
    
	if((mode==0)||(mode==2)){ black = false; } else { black = true; }
    if((mode==0)||(mode==1)){ cubic = false; } else { cubic = true; }
    
    nlhs=(int)nlhs_d[0];
    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 
    
    Osize[0] = (int)Osize_d[0]; 
    Osize[1] = (int)Osize_d[1]; 
    
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
  
    ThreadOffset=(int) ThreadID[0];
    
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
            interpolate_2d_double_color(Ipixel,Tlocalx, Tlocaly, Isize, Iin,cubic,mode);
                        
            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            Iout[indexI+index_rgb[0]]=Ipixel[0];
            Iout[indexI+index_rgb[1]]=Ipixel[1];
            Iout[indexI+index_rgb[2]]=Ipixel[2];
                        
            /*  Store transformation field */
            if(nlhs>1) { Tx[indexI]=Tlocalx-(double)x; }
            if(nlhs>2) { Ty[indexI]=Tlocaly-(double)y; }
        }
    }
    
	free(u_index_array);
	free(v_index_array);
	free(i_array);
	free(j_array);
	
    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
	EndThread;
}

voidthread transformvolume_gray(double **Args) {
    double *Bu, *Bv, *Iout, *Tx, *Ty, *dxa, *dya, *ThreadID, *Ox, *Oy, *Iin, *moded;
    double *Isize_d;
    double *Osize_d;
    double *nlhs_d;
    int Isize[3]={0,0,0};
    int mode=0;
    int Osize[2]={0,0};
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
    
     /* loop throught the colors r,g,b */
    int rgb=0;
    
    /* X,Y coordinates of current pixel */
    int x,y;
    
    /* B-spline variablesl */
    int u_index=0; 
    int v_index=0;
    int i, j;
    
    /* temporary value */
	double val;
	
	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
    
    /*  B-Spline loop variabels */
    int l,m;
    int nlhs=0;
   
    /* Cubic and outside black booleans */
    bool black, cubic;
    
    /* Current voxel/pixel */
    double Ipixel[3]={0,0,0};
    
    /* Split input into variables */
    Bu=Args[0];
    Bv=Args[1];
    Isize_d=Args[2];
    Osize_d=Args[3];
    Iout=Args[4];
    Tx=Args[5];
    Ty=Args[6];
    dxa=Args[7];
    dya=Args[8];
    ThreadID=Args[9];
    Ox=Args[10];
    Oy=Args[11];
    Iin=Args[12];
    nlhs_d=Args[13];
    moded=Args[14]; mode=(int) moded[0];
    Nthreadsd=Args[15];  Nthreads=(int)Nthreadsd[0];
     
    nlhs=(int)nlhs_d[0];
    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 
    
    Osize[0] = (int)Osize_d[0]; 
    Osize[1] = (int)Osize_d[1]; 
    
    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
	if((mode==0)||(mode==2)){ black = false; } else { black = true; }
    if((mode==0)||(mode==1)){ cubic = false; } else { cubic = true; }
	
    ThreadOffset=(int) ThreadID[0];
	
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
							  Tlocalx+=val*Ox[indexO];
							  Tlocaly+=val*Oy[indexO];
						 }
					}
				}
            }            

            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            
            Iout[indexI]=interpolate_2d_double_gray(Tlocalx, Tlocaly, Isize, Iin,cubic,black); 
            
            /*  Store transformation field */
            if(nlhs>1) { Tx[indexI]=Tlocalx-(double)x; }
            if(nlhs>2) { Ty[indexI]=Tlocaly-(double)y; }
        }
    }
    
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
    double *Ox,*Oy, *Iin, *dxa, *dya, *Iout,*Tx,*Ty,*moded;
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
    mwSize  Isizex, Isizey;
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

    /* Loop variable  */
    int i;
    
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
  
  /* Create image matrix for the return arguments with the size of input image  */  
  if(Isize_d[2]>1) {
      plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  }
  else  {
      plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  }
  
  Isizex = (mwSize) Isize_d[0]; Isizey = (mwSize) Isize_d[1];
  if(nlhs>1) { plhs[1] = mxCreateDoubleMatrix(Isizex,Isizey, mxREAL); }
  if(nlhs>2) { plhs[2] = mxCreateDoubleMatrix(Isizex,Isizey, mxREAL); }
  
  
  /* Assign pointers to each input. */
  Ox=mxGetPr(prhs[0]);
  Oy=mxGetPr(prhs[1]);
  Iin=mxGetPr(prhs[2]);
  dxa=mxGetPr(prhs[3]);
  dya=mxGetPr(prhs[4]);
  moded=mxGetPr(prhs[5]);
  
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
    
  /* Assign pointer to output. */
  Iout = mxGetPr(plhs[0]);
  if(nlhs>1) { Tx = mxGetPr(plhs[1]); }
  if(nlhs>2) { Ty = mxGetPr(plhs[2]); }
  
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
  
  Isize_d[0]=Isizex;  Isize_d[1]=Isizey;
  Osize_d[0]=Osizex;  Osize_d[1]=Osizey;
  
  nlhs_d[0]=(double)nlhs;
  
  
  /* Reserve room for 16 function variables(arrays)   */
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (double *)malloc( 1* sizeof(double) );
    ThreadID1[0]=i;
    ThreadID[i]=ThreadID1;  

    /*  Make Thread Structure  */
    ThreadArgs1 = (double **)malloc( 16* sizeof( double * ) );  
    ThreadArgs1[0]=Bu;
    ThreadArgs1[1]=Bv;
    ThreadArgs1[2]=Isize_d;
    ThreadArgs1[3]=Osize_d;
    ThreadArgs1[4]=Iout;
    ThreadArgs1[5]=Tx;
    ThreadArgs1[6]=Ty;
    ThreadArgs1[7]=dxa;
    ThreadArgs1[8]=dya;
    ThreadArgs1[9]=ThreadID[i];
    ThreadArgs1[10]=Ox;
    ThreadArgs1[11]=Oy;
    ThreadArgs1[12]=Iin;
    ThreadArgs1[13]=nlhs_d;
    ThreadArgs1[14]=moded;
    ThreadArgs1[15]=Nthreadsd;
    ThreadArgs[i]=ThreadArgs1;
  
    if(Isize_d[2]>1) {
	 StartThread(ThreadList[i], &transformvolume_color, ThreadArgs[i])
    }
    else
    {
		 StartThread(ThreadList[i], &transformvolume_gray, ThreadArgs[i])

    }
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
}
        

