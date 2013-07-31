/* Image and Volume interpolation 
 *
 * Function is written by D.Kroon University of Twente (June 2009)
 */

#define EPS  1e-20
#define EPSF 1e-12f
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


/* Convert 2D/3D matrix index to 1D index */
static __inline int mindex2(int x, int y, int sizx) { return y*sizx+x; }
static __inline int mindex3(int x, int y, int z, int sizx, int sizy) { return z*sizx*sizy+y*sizx+x;}
static __inline int mindex2c(int x, int y, int sizx, int sizy)  
{ 
    if(x<0) { x=0;}
    if(y<0) { y=0;}
    if(x>(sizx-1)) { x=sizx-1; };
    if(y>(sizy-1)) { y=sizy-1; };
    return y*sizx+x;  
}
static __inline int mindex3c(int x, int y, int z, int sizx, int sizy, int sizz)  
{ 
    if(x<0) { x=0;}
    if(y<0) { y=0;}
    if(z<0) { z=0;}
    if(x>(sizx-1)) { x=sizx-1; };
    if(y>(sizy-1)) { y=sizy-1; };
    if(z>(sizz-1)) { z=sizz-1; };
    return z*sizx*sizy+y*sizx+x;
}

static __inline double BsplineCoefficient(double u, int i) {
    double a=0;
    switch(i) {
        case 0:
            a= pow((1-u), 3.0)/6.0; break;
        case 1:
            a= ( 3.0*pow(u, 3) - 6.0*pow(u, 2.0) + 4.0)/6.0; break;
        case 2:
            a= (-3.0*pow(u, 3) + 3.0*pow(u, 2.0) + 3.0*u + 1)/6; break;
        case 3:
            a= pow(u, 3)/6; break;
    }
    return a;
}

static __inline double BsplineCoefficientDerivative(double u, int i) {
    double a=0;
    switch(i) {
        case 0:
            a=-pow(u - 1.0, 2.0)/2.0; break;
        case 1:
            a=(3.0*pow(u, 2.0))/2.0 - 2.0*u; break;
        case 2:
            a=-(3.0*pow(u, 2.0))/2.0 + u + 1.0/2.0; break;
        case 3:
            a=pow(u, 2.0)/2.0; break;
    }
    return a;
    
}



/* power of integers */
static __inline double pow2(double val) { return val*val; }
static __inline double pow3(double val) { return val*val*val; }
static __inline double pow4(double val) { return pow2(val)*pow2(val); }
static __inline float pow2_float(float val) { return val*val; }
static __inline float pow3_float(float val) { return val*val*val; }
static __inline float pow4_float(float val) { return pow2_float(val)*pow2_float(val); }


#ifdef __LCC__
static __inline float floorfloat(float val) { return (float)floor((double)val); }
#else
static __inline float floorfloat(float val) { return floorf(val); }
#endif


/* Get an pixel from an image, if outside image, black or nearest pixel */
double getintensity_mindex2(int x, int y, int sizx, int sizy, double *I);

/* Get an pixel from an image, if outside image, black or nearest pixel */
double getcolor_mindex2(int x, int y, int sizx, int sizy, double *I, int rgb);


/* Get an pixel from an image, if outside image, black or nearest pixel */
double getcolor_mindex3(int x, int y, int z, int sizx, int sizy, int sizz, double *I);
float  getcolor_mindex3_float(int x, int y, int z, int sizx, int sizy, int sizz, float *I);

/* Pixel Interpolation */
double interpolate_2d_double_gray(double Tlocalx, double Tlocaly, int *Isize, double *Iin,int cubic,int black);
void interpolate_2d_double_color(double *Ipixel, double Tlocalx, double Tlocaly, int *Isize, double *Iin, int cubic, int black);
double interpolate_3d_double_gray(double Tlocalx, double Tlocaly, double Tlocalz, int *Isize, double *Iin,int cubic,int black);
float interpolate_3d_float_gray(float Tlocalx, float Tlocaly, float Tlocalz, int *Isize, float *Iin,int cubic,int black);




