#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
/*   undef needed for LCC compiler  */
#include "multiple_os_thread.h"

/*  This function movepixels, will translate the pixels of an image
 *  according to x, y and z translation images (bilinear interpolated). 
 * 
 *  Iout = movepixels_3d_float(I,Tx,Ty,Tz);
 *
 *  Function is written by D.Kroon University of Twente (July 2009)
 */

voidthread transformvolume(float **Args) {
    /* I is the input image, Iout the transformed image  */
    /* Tx and Ty images of the translation of every pixel. */
    float *Iin, *Iout, *Tx, *Ty, *Tz;
    float *Nthreadsd;
    int Nthreads;
	/*  if one outside pixels are set to zero. */
	float  *moded;
	int mode=0;
   
    /* Cubic and outside black booleans */
    bool black, cubic;
    
    /* 3D index storage*/
    int indexI;
    
    /* Size of input image */
    float *Isize_d;
    int Isize[3]={0,0,0};
        
    /* Location of translated pixel */
    float Tlocalx;
    float Tlocaly;
    float Tlocalz;
    
    /* offset */
    int ThreadOffset=0;
    
    /* The thread ID number/name */
    float *ThreadID;
    
    /* X,Y coordinates of current pixel */
    int x,y,z;
    
    Iin=Args[0];
    Iout=Args[1];
    Tx=Args[2];
    Ty=Args[3];
    Tz=Args[4];
    Isize_d=Args[5];
    ThreadID=Args[6];
	moded=Args[7]; mode=(int) moded[0];
	Nthreadsd=Args[8];  Nthreads=(int)Nthreadsd[0];
    
    if(mode==0||mode==2){ black = false; } else { black = true; }
    if(mode==0||mode==1){ cubic = false; } else { cubic = true; }
    
    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Isize[2] = (int)Isize_d[2]; 
       
    ThreadOffset=(int) ThreadID[0];
	
    /*  Loop through all image pixel coordinates */
    for (z=ThreadOffset; z<Isize[2]; z=z+Nthreads)
	{
        for (y=0; y<Isize[1]; y++)
        {
            for (x=0; x<Isize[0]; x++)
            {
                Tlocalx=((float)x)+Tx[mindex3(x,y,z,Isize[0],Isize[1])];
                Tlocaly=((float)y)+Ty[mindex3(x,y,z,Isize[0],Isize[1])];
                Tlocalz=((float)z)+Tz[mindex3(x,y,z,Isize[0],Isize[1])];
                
                /* Set the current pixel value */
                indexI=mindex3(x,y,z,Isize[0],Isize[1]);
                Iout[indexI]=interpolate_3d_float_gray(Tlocalx, Tlocaly, Tlocalz, Isize, Iin,cubic,black); 
            }
        }
    }

    /*  explicit end thread, helps to ensure proper recovery of resources allocated for the thread */
   EndThread;
}


/* The matlab mex function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    /* I is the input image, Iout the transformed image  */
    /* Tx and Ty images of the translation of every pixel. */
    float *Iin, *Iout, *Tx, *Ty, *Tz;
	float *moded;
    mxArray *matlabCallOut[1]={0};
    mxArray *matlabCallIn[1]={0};
    double *Nthreadsd; float Nthreadsf[1]={0};
    int Nthreads;
	
    /* float pointer array to store all needed function variables  */
    float ***ThreadArgs;
    float **ThreadArgs1;
    
	/* Handles to the worker threads */
		ThreadHANDLE *ThreadList;
	
    
    /* ID of Threads */
    float **ThreadID;              
    float *ThreadID1;

    /* Size of input image */
    mwSize Isizex, Isizey,Isizez;
    float Isize_d[3]={0,0,0};
    const mwSize *dims;

	/* Loop variable  */
	int i;
	
    /* Check for proper number of arguments. */
    if(nrhs!=5) {
      mexErrMsgTxt("Five inputs are required.");
    } else if(nlhs!=1) {
      mexErrMsgTxt("One output required");
    }
 
    /* Get the sizes of the input image */
    dims = mxGetDimensions(prhs[0]);   
    Isizex = (mwSize)dims[0]; 
    Isizey = (mwSize)dims[1];
    Isizez = (mwSize)dims[2];

    Isize_d[0]=(float)Isizex;  Isize_d[1]=(float)Isizey; Isize_d[2]=(float)Isizez;
    
    /* Create image matrix for the return arguments with the size of input image  */  
    plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL); 
    
    /* Assign pointers to each input. */
    Iin=(float *)mxGetData(prhs[0]);
    Tx=(float *)mxGetData(prhs[1]);
    Ty=(float *)mxGetData(prhs[2]);
    Tz=(float *)mxGetData(prhs[3]);
	moded=(float *)mxGetData(prhs[4]);
	

	mexCallMATLAB(1, matlabCallOut, 0, matlabCallIn, "maxNumCompThreads");
	Nthreadsd=mxGetPr(matlabCallOut[0]);
	Nthreads=(int)Nthreadsd[0];  Nthreadsf[0]=(float)Nthreadsd[0];
    /* Reserve room for handles of threads in ThreadList  */
		ThreadList = (ThreadHANDLE*)malloc(Nthreads* sizeof( ThreadHANDLE ));
	
	ThreadID = (float **)malloc( Nthreads* sizeof(float *) );
	ThreadArgs = (float ***)malloc( Nthreads* sizeof(float **) );
	
    /* Assign pointer to output. */
    Iout = (float *)mxGetData(plhs[0]);
   
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (float *)malloc( 1* sizeof(float) );
    ThreadID1[0]=(float)i;
    ThreadID[i]=ThreadID1;  
    
	/*  Make Thread Structure  */
    ThreadArgs1 = (float **)malloc( 9* sizeof( float * ) );  
    ThreadArgs1[0]=Iin;
    ThreadArgs1[1]=Iout;
    ThreadArgs1[2]=Tx;
    ThreadArgs1[3]=Ty;
    ThreadArgs1[4]=Tz;
    ThreadArgs1[5]=Isize_d;
    ThreadArgs1[6]=ThreadID[i];
	ThreadArgs1[7]=moded;
	ThreadArgs1[8]=Nthreadsf;
	
    /* Start a Thread  */
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
}
        

