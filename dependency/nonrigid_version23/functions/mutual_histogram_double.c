#include "mex.h"
#include "math.h"

/* This function makes a 2D joint histogram of 1D,2D...ND images
 * and also calculates the seperate histograms of both images.
 *
 * [hist12, hist1, hist2]=mutual_histogram_double(I1,I2,Imin,Imax,nbins);
 *
 * Function is written by D.Kroon University of Twente (July 2008)
 */

int mindex2(int x, int y, int sizx) { return y*sizx+x; }

/* The matlab mex function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    /*   I1 and I2 are the input images */
    /*  hist12 joint histogram */
    /*  hist1 and histogram of I1 and hist2 of I2 */
    double *I1, *I2, *Imin, *Imax, *nbins, *hist12, *hist1, *hist2;
    
    /* Size of input image */
    const mwSize *idims; 
    
    /*  Size of output */
    int odims2[2]={0,0};
    int odims1[1]={0};    
    /*  Dimensions */
    int nsubs;
    int npixels=1;
    double npixelsi=1;
   
    /* intensity location*/
    int sizex, sizexd;
    double xd, xm, xp, xmd, xpd;
    double yd, ym, yp, ymd, ypd;
    int xmi, xpi, ymi, ypi;
	
    /* loop vars*/
    int i;
    
    /*  vars*/
    double minv;
    double scav;
    
    /* Check for proper number of arguments. */
    if(nrhs!=5) {
       mexErrMsgTxt("five inputs are required.");
    } else if(nlhs!=3) {
       mexErrMsgTxt("Three outputs are required");
    }
  
    /*  Get the number of dimensions */
    nsubs = mxGetNumberOfDimensions(prhs[0]);
    /* Get the sizes of the grid */
    idims = mxGetDimensions(prhs[0]);   
    for (i=0; i<nsubs; i++) { npixels=npixels*idims[i]; }
    npixelsi=1/((double)npixels);

    /* Assign pointers to each input. */
    I1=(double *)mxGetData(prhs[0]);
    I2=(double *)mxGetData(prhs[1]);
    Imin=(double *)mxGetData(prhs[2]);
    Imax=(double *)mxGetData(prhs[3]);
    nbins=(double *)mxGetData(prhs[4]);
    
    /*  Create image matrix for the return arguments*/
    odims2[0]=(int) nbins[0]; odims2[1]=(int)nbins[0];  
    plhs[0] = mxCreateNumericArray(2, odims2, mxDOUBLE_CLASS, mxREAL);
    odims1[0]=(int) nbins[0]; 
    plhs[1] = mxCreateNumericArray(1, odims1, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(1, odims1, mxDOUBLE_CLASS, mxREAL);

    /* Assign pointers to each output. */
    hist12=(double *)mxGetData(plhs[0]);
    hist1=(double *)mxGetData(plhs[1]);
    hist2=(double *)mxGetData(plhs[2]);

    /* min value */
    minv=Imin[0];
    /* scale value */
    scav=nbins[0]/(Imax[0]-Imin[0]);
    sizex=(int) nbins[0];
    sizexd=sizex-1;

    for (i=0; i<npixels; i++)
    {
        xd=(double)scav*(I1[i]-minv);
        xm=(double)floor(xd); xp=xm+1;
        xmd=xp-xd; xpd=xd-xm;
                
        yd=(double)scav*(I2[i]-minv);
        ym=(double)floor(yd); yp=ym+1;
        ymd=yp-yd; ypd=yd-ym;

        xmi=(int)xm; xpi=(int)xp;
		ymi=(int)ym; ypi=(int)yp;
		
        /* Make sum of all values in histogram 1 and histrogram 2 equal to 1*/
         
        xmd*=npixelsi; ymd*=npixelsi; xpd*=npixelsi;  ypd*=npixelsi;
                        

        if(xmi<0){ xmi=0; } else if(xmi>sizexd) { xmi=sizexd; }
        if(xpi<0){ xpi=0; } else if(xpi>sizexd) { xpi=sizexd; }
        if(ymi<0){ ymi=0; } else if(ymi>sizexd) { ymi=sizexd; }
        if(ypi<0){ ypi=0; } else if(ypi>sizexd) { ypi=sizexd; }

        hist12[xmi+ymi*sizex]+=xmd*ymd;
        hist12[xpi+ymi*sizex]+=xpd*ymd;
        hist12[xmi+ypi*sizex]+=xmd*ypd;
        hist12[xpi+ypi*sizex]+=xpd*ypd;

        hist1[xmi]=hist1[xmi]+xmd; hist1[xpi]=hist1[xpi]+xpd;
        hist2[ymi]=hist2[ymi]+ymd; hist2[ypi]=hist2[ypi]+ypd;
    }
}
        

