#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"

voidthread transformvolume_jacobiandet(double **Args) {
    double *Bu, *Bv, *Bdu, *Bdv, *Iout, *dxa, *dya, *ThreadID, *Ox, *Oy;
    double *Isize_d;
    double *Osize_d;
    int Isize[3]={0,0,0};
    int Osize[2]={0,0};

    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    /* Location of pixel which will be come the current pixel */
    double Tlocalx,Tlocaly;
    double Tlocaldxx, Tlocaldxy, Tlocaldyy, Tlocaldyx;
            
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
	double valx,valy;

	
	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
    /*  B-Spline loop variabels */
    int l,m;


    /* Current voxel/pixel */
	double Ipixel[3]={0,0,0};
    /* Split input into variables */
    Bu=Args[0];
    Bv=Args[1];
    Isize_d=Args[2];
    Osize_d=Args[3];
    Iout=Args[4];
    dxa=Args[5];
    dya=Args[6];
    ThreadID=Args[7];
    Ox=Args[8];
    Oy=Args[9];
    Nthreadsd=Args[10];  Nthreads=(int)Nthreadsd[0];
    Bdu=Args[11];
    Bdv=Args[12];
    Isize[0] = (int)Isize_d[0]; 
    Isize[1] = (int)Isize_d[1]; 
    Osize[0] = (int)Osize_d[0]; 
    Osize[1] = (int)Osize_d[1]; 

    /* Get the spacing of the uniform b-spline grid */
    dx=(int)dxa[0]; dy=(int)dya[0];
    
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
            Tlocaldxx=0; Tlocaldxy=0; 
            Tlocaldyy=0; Tlocaldyx=0;
            
            for(l=0; l<4; l++)
            {
				if(((i+l)>=0)&&((i+l)<Osize[0]))
				{
					for(m=0; m<4; m++)
					{    
						 if(((j+m)>=0)&&((j+m)<Osize[1]))
						 {
							  indexO=(i+l)+(j+m)*Osize[0];
                              valx=Bdu[l+u_index]*Bv[m+v_index];
                              valy=Bu[l+u_index]*Bdv[m+v_index];
                              
                              Tlocaldxx+=valx*Ox[indexO]; 
                              Tlocaldyy+=valy*Oy[indexO];
                              Tlocaldxy+=valy*Ox[indexO]; 
                              Tlocaldyx+=valx*Oy[indexO];
						 }
					}
				}
            }            

            /* Set the current pixel value */
            indexI=mindex2(x,y,Isize[0]);
            
            Iout[indexI]=Tlocaldxx*Tlocaldyy-Tlocaldyx*Tlocaldxy;




        }
    }
    
	/* Free memory index look up tables */
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
    double *Ox,*Oy, *dxa, *dya, *Iout;
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
    


    /* Size of input image */
    double *Isize_d;
    mwSize dims[3];
    
    
    /* Size of grid */
    mwSize  Osizex, Osizey;
    double Osize_d[2]={0,0};
    
    /* B-spline variablesl */
    double u,v;
    int u_index=0; 
    int v_index=0;
    double *Bu, *Bv, *Bdu, *Bdv;

    /* Loop variable  */
    int i;
    
    /* X,Y coordinates of current pixel */
    int x,y;
    /* Grid distance */
    int dx,dy; 
    
  /* Check for proper number of arguments. */
  if(nrhs!=5) {
    mexErrMsgTxt("Five inputs are required.");
  }
  
    /* Get the sizes of the grid */
  Osizex = (mwSize)mxGetM(prhs[0]);  
  Osizey = (mwSize)mxGetN(prhs[0]);
    
  /* Assign pointers to each input. */
  Ox=mxGetPr(prhs[0]);
  Oy=mxGetPr(prhs[1]);
  Isize_d=mxGetPr(prhs[2]);
  dxa=mxGetPr(prhs[3]);
  dya=mxGetPr(prhs[4]);
  
  /* Create image matrix for the return arguments with the size of input image  */  
  dims[0]=(mwSize)Isize_d[0]; dims[1]=(mwSize)Isize_d[1];
  plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
  
 




 
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


  
  /*  Make polynomial look up tables   */
  Bu=malloc(dx*4*sizeof(double));
  Bv=malloc(dy*4*sizeof(double));
  Bdu=malloc(dx*4*sizeof(double));
  Bdv=malloc(dy*4*sizeof(double));
  for (x=0; x<dx; x++)
  {
    u=(x/(double)dx)-floor(x/(double)dx);
    Bu[mindex2(0,x,4)] = BsplineCoefficient(u,0);
    Bu[mindex2(1,x,4)] = BsplineCoefficient(u,1);
    Bu[mindex2(2,x,4)] = BsplineCoefficient(u,2);
    Bu[mindex2(3,x,4)] = BsplineCoefficient(u,3);
    Bdu[mindex2(0,x,4)] = BsplineCoefficientDerivative(u,0)/dxa[0];
    Bdu[mindex2(1,x,4)] = BsplineCoefficientDerivative(u,1)/dxa[0];
    Bdu[mindex2(2,x,4)] = BsplineCoefficientDerivative(u,2)/dxa[0];
    Bdu[mindex2(3,x,4)] = BsplineCoefficientDerivative(u,3)/dxa[0];
  }
  
  for (y=0; y<dy; y++)
  {
    v=(y/(double)dy)-floor(y/(double)dy);
    Bv[mindex2(0,y,4)] = BsplineCoefficient(v,0);
    Bv[mindex2(1,y,4)] = BsplineCoefficient(v,1);
    Bv[mindex2(2,y,4)] = BsplineCoefficient(v,2);
    Bv[mindex2(3,y,4)] = BsplineCoefficient(v,3);
    Bdv[mindex2(0,y,4)] = BsplineCoefficientDerivative(v,0)/dya[0];
    Bdv[mindex2(1,y,4)] = BsplineCoefficientDerivative(v,1)/dya[0];
    Bdv[mindex2(2,y,4)] = BsplineCoefficientDerivative(v,2)/dya[0];
    Bdv[mindex2(3,y,4)] = BsplineCoefficientDerivative(v,3)/dya[0];
  }
  
  Osize_d[0]=Osizex;  Osize_d[1]=Osizey;
    
  /* Reserve room for 14 function variables(arrays)   */
  for (i=0; i<Nthreads; i++)
  {
    /*  Make Thread ID  */
    ThreadID1= (double *)malloc( 1* sizeof(double) );
    ThreadID1[0]=i;
    ThreadID[i]=ThreadID1;  

    /*  Make Thread Structure  */
    ThreadArgs1 = (double **)malloc( 13* sizeof( double * ) );  
    ThreadArgs1[0]=Bu;
    ThreadArgs1[1]=Bv;
    ThreadArgs1[2]=Isize_d;
    ThreadArgs1[3]=Osize_d;
    ThreadArgs1[4]=Iout;
    ThreadArgs1[5]=dxa;
    ThreadArgs1[6]=dya;
    ThreadArgs1[7]=ThreadID[i];
    ThreadArgs1[8]=Ox;
    ThreadArgs1[9]=Oy;
    ThreadArgs1[10]=Nthreadsd;
    ThreadArgs1[11]=Bdu;
    ThreadArgs1[12]=Bdv;

    ThreadArgs[i]=ThreadArgs1;
  
  StartThread(ThreadList[i], &transformvolume_jacobiandet, ThreadArgs[i])

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
  free(Bdu);
  free(Bv);
  free(Bdv);
  
}
        

