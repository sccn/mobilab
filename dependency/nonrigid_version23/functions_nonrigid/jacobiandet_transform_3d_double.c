#include "mex.h"
#include "math.h"
#include "image_interpolation.h"
#include "multiple_os_thread.h"
  
voidthread transformvolume_jacobiandet(double **Args) {
    double *Bu, *Bv, *Bw, *Bdu, *Bdv, *Bdw, *Iout;
    double *dxa, *dya, *dza, *ThreadID, *Ox, *Oy, *Oz;
    double *Isize_d;
    double *Osize_d;
    int mode=0;
    int Isize[3]={0,0,0};
    int Osize[3]={0,0,0};
    double *Nthreadsd;
    int Nthreads;
    /* Multiple threads, one does the odd the other even indexes */
    int ThreadOffset;
    /* Location of pixel which will be come the current pixel */
    double Tlocaldxx,Tlocaldxy,Tlocaldxz;
    double Tlocaldyx,Tlocaldyy,Tlocaldyz;
    double Tlocaldzx,Tlocaldzy,Tlocaldzz;
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
	double valx,valy,valz;
   	/* Look up tables index */
	int *u_index_array, *i_array;
	int *v_index_array, *j_array;
	int *w_index_array, *k_array;
	/*  B-Spline loop variabels */
    int l,m,n;
    /* Split input into variables */
    Bu=Args[0];
    Bv=Args[1];
    Bw=Args[2];
    Isize_d=Args[3];
    Osize_d=Args[4];
    Iout=Args[5];
    dxa=Args[6];
    dya=Args[7];
    dza=Args[8];
    ThreadID=Args[9];
    Ox=Args[10];
    Oy=Args[11];
    Oz=Args[12];
    Nthreadsd=Args[13];  Nthreads=(int)Nthreadsd[0];
	Bdu=Args[14];
    Bdv=Args[15];
    Bdw=Args[16];
       
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
                Tlocaldxx=0; Tlocaldxy=0; Tlocaldxz=0; 
                Tlocaldyx=0; Tlocaldyy=0; Tlocaldyz=0;
                Tlocaldzx=0; Tlocaldzy=0; Tlocaldzz=0;
                
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
                                         valx=Bdu[l+u_index]*Bv[m+v_index]*Bw[n+w_index];
										 valy=Bu[l+u_index]*Bdv[m+v_index]*Bw[n+w_index];
										 valz=Bu[l+u_index]*Bv[m+v_index]*Bdw[n+w_index];
                                  
									     Tlocaldxx+=valx*Ox[indexO];
                                         Tlocaldxy+=valy*Ox[indexO];
                                         Tlocaldxz+=valz*Ox[indexO];
										 Tlocaldyx+=valx*Oy[indexO]; 
                                         Tlocaldyy+=valy*Oy[indexO]; 
                                         Tlocaldyz+=valz*Oy[indexO]; 
										 Tlocaldzx+=valx*Oz[indexO];
										 Tlocaldzy+=valy*Oz[indexO];
										 Tlocaldzz+=valz*Oz[indexO];
										 
                                    }
                                }
                            }
                        }
                    }
                }          

                /* Set the current pixel value */
                indexI=x+y*Isize[0]+z*Isize[0]*Isize[1];

                Iout[indexI] =  Tlocaldxx*Tlocaldyy*Tlocaldzz + Tlocaldxy*Tlocaldyz*Tlocaldzx;
                Iout[indexI]+=  Tlocaldxz*Tlocaldyx*Tlocaldzy - Tlocaldxz*Tlocaldyy*Tlocaldzx;
                Iout[indexI]+= -Tlocaldxy*Tlocaldyx*Tlocaldzz - Tlocaldxx*Tlocaldyz*Tlocaldzy;
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
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    /* Ox and Oy are the grid points */
    /* Zo is the input image */
    /* Zi is the transformed image */

    /* nx and ny are the number of grid points (inside the image) */
    double *Ox,*Oy,*Oz,*dxa, *dya,*dza,*Iout;
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
	double *Isize_d;
    mwSize dims[3];


    /* Size of grid */
    mwSize  Osizex, Osizey, Osizez;
    double Osize_d[3]={0,0,0};
    const mwSize *dimso;
  
    /* B-spline variablesl */
    double u,v,w;
    int u_index=0; 
    int v_index=0;
    int w_index=0;
    
    double *Bu, *Bv, *Bw;
    double *Bdu, *Bdv, *Bdw;
	
	/* Loop variable  */
	int i;
	/* Grid distance */
    int dx,dy,dz; 
    /* X,Y,Z coordinates of current pixel */
    int x,y,z;
        
  /* Check for proper number of arguments. */
  if(nrhs!=7) {
    mexErrMsgTxt("Seven inputs are required.");
  }
 

  /* Get the sizes of the grid */
  dimso = mxGetDimensions(prhs[0]);   
  Osizex = dimso[0]; 
  Osizey = dimso[1];
  Osizez = dimso[2];
    
  /* Assign pointers to each input. */
  Ox=(double *)mxGetData(prhs[0]);
  Oy=(double *)mxGetData(prhs[1]);
  Oz=(double *)mxGetData(prhs[2]);
  Isize_d=(double *)mxGetData(prhs[3]);
  dxa=(double *)mxGetData(prhs[4]);
  dya=(double *)mxGetData(prhs[5]);
  dza=(double *)mxGetData(prhs[6]);
   
   
  /* Create image matrix for the return arguments with the size of input image  */  
  dims[0]=(mwSize)Isize_d[0];
  dims[1]=(mwSize)Isize_d[1];
  dims[2]=(mwSize)Isize_d[2];
  
  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 
  
  
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
 
   /*  Make polynomial look up tables   */
  Bu=malloc(dx*4*sizeof(double));
  Bv=malloc(dy*4*sizeof(double));
  Bw=malloc(dz*4*sizeof(double));
  Bdu=malloc(dx*4*sizeof(double));
  Bdv=malloc(dy*4*sizeof(double));
  Bdw=malloc(dz*4*sizeof(double));
  
  for (x=0; x<dx; x++)
  {
    u=((double)x/(double)dx)-floor((double)x/(double)dx);
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
    v=((double)y/(double)dy)-floor((double)y/(double)dy);
    Bv[mindex2(0,y,4)] = BsplineCoefficient(v,0);
    Bv[mindex2(1,y,4)] = BsplineCoefficient(v,1);
    Bv[mindex2(2,y,4)] = BsplineCoefficient(v,2);
    Bv[mindex2(3,y,4)] = BsplineCoefficient(v,3);
    Bdv[mindex2(0,y,4)] = BsplineCoefficientDerivative(v,0)/dya[0];
    Bdv[mindex2(1,y,4)] = BsplineCoefficientDerivative(v,1)/dya[0];
    Bdv[mindex2(2,y,4)] = BsplineCoefficientDerivative(v,2)/dya[0];
    Bdv[mindex2(3,y,4)] = BsplineCoefficientDerivative(v,3)/dya[0];
  }
  
  for (z=0; z<dz; z++)
  {
    w=((double)z/(double)dz)-floor((double)z/(double)dz);
    Bw[mindex2(0,z,4)] = BsplineCoefficient(w,0);
    Bw[mindex2(1,z,4)] = BsplineCoefficient(w,1);
    Bw[mindex2(2,z,4)] = BsplineCoefficient(w,2);
    Bw[mindex2(3,z,4)] = BsplineCoefficient(w,3);
    Bdw[mindex2(0,z,4)] = BsplineCoefficientDerivative(w,0)/dza[0];
    Bdw[mindex2(1,z,4)] = BsplineCoefficientDerivative(w,1)/dza[0];
    Bdw[mindex2(2,z,4)] = BsplineCoefficientDerivative(w,2)/dza[0];
    Bdw[mindex2(3,z,4)] = BsplineCoefficientDerivative(w,3)/dza[0];
	
  }
  

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
    ThreadArgs1 = (double **)malloc( 17* sizeof( double * ) );  
	ThreadArgs1[0]=Bu;
	ThreadArgs1[1]=Bv;
	ThreadArgs1[2]=Bw;
	ThreadArgs1[3]=Isize_d;
	ThreadArgs1[4]=Osize_d;
	ThreadArgs1[5]=Iout;
	ThreadArgs1[6]=dxa;
	ThreadArgs1[7]=dya;
	ThreadArgs1[8]=dza;
	ThreadArgs1[9]=ThreadID[i];
	ThreadArgs1[10]=Ox;
	ThreadArgs1[11]=Oy;
	ThreadArgs1[12]=Oz;
	ThreadArgs1[13]=Nthreadsd;
    ThreadArgs1[14]=Bdu;
    ThreadArgs1[15]=Bdv;
	ThreadArgs1[16]=Bdw;
	
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
  free(Bv);
  free(Bw);
  free(Bdu);
  free(Bdv);
  free(Bdw);
  
}
        

