#include "math.h"
#include "MRG63k3a.h"

void sobol_ags(int n, int d, int randomize, int skip, int graycode, long seed, double *res, unsigned int *z, unsigned int d_max, unsigned int m_max){
    double scale = pow(2,(double) (-m_max));
	seed_MRG63k3a(seed); /* seed the IID RNG */
    if(randomize==1){
        /* linear matrix scramble */

    }
    if((randomize==1)||(randomize==2)){
        /* digital shift (also applid to linear matrix scramble) */

    }

}