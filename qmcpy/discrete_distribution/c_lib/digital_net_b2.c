#include "export_ctypes.h"
#include <math.h>

EXPORT int get_unsigned_long_size(){return sizeof(unsigned long);}
EXPORT int get_unsigned_long_long_size(){return sizeof(unsigned long long);}

EXPORT int gen_digitalnetb2(
    unsigned long n0, /* starting index in sequence. Must be a power of 2 if not using Graycode ordering */
    unsigned long n, /* n: sample indicies include n0:(n0+n). Must be a power of 2 if not using Graycode ordering */
    unsigned int d, /* dimension supported by generating vector */
    unsigned int graycode, /* Graycode flag */
    unsigned int m_max, /* 2^m_max is the maximum number of samples supported */
    unsigned long long *znew, /* generating vector with shape d_max x m_max from set_digitalnetb2_randomizations */
    unsigned int set_rshift, /* random shift flag */
    unsigned long long *rshift, /* length d vector of random digital shifts from set_digitalnetb2_randomizations */
    double *x, /* unrandomized points with shape n x d */
    double *xr){ /* randomized points with shape n x d */
    /*
    Digital Net Generator by alegresor 

    References:
            
        [1] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
        DOI:https://doi.org/10.1145/42288.214372
        
    Error Codes:
        1) using natural ordering (graycode=0) and n0 and/or (n0+n) is not 0 or a power of 2
        2) n0+n exceeds 2^m_max
    */
    if( (n==0) || (d==0) ){
        return(0);}
    if( (graycode==0) && ( ((n0!=0)&&fmod(log(n0)/log(2),1)!=0) || (fmod(log(n0+n)/log(2),1)!=0) ) ){
        /* for natural ordering, require n0 and (n0+n) be either 0 or powers of 2 */
        return(1);}
    if( (n0+n) > ldexp(1,m_max) ){
        /* too many samples */
        return(2);}
    /* variables */
    double scale = ldexp(1,-1*m_max);
    unsigned int j, m, k, s;
    unsigned long long i, im, xc, z1, b;
    /* generate points */
    for(j=0;j<d;j++){
        /* set an initial point */ 
        xc = 0; /* current point */
        z1 = 0; /* next directional vector */
        if(n0>0){
            im = n0-1;
            b = im; 
            im ^= im>>1;
            m = 0;
            while((im!=0) && (m<m_max)){
                if(im&1){
                    xc ^= znew[j*m_max+m];}
                im >>= 1;
                m += 1;}
            s = 0;
            while(b&1){
                b >>= 1;
                s += 1;}
            z1 = znew[j*m_max+m];}
        /* set the rest of the points */
        for(i=n0;i<(n0+n);i++){
            xc ^= z1;  
            /* set point */
            im = i;
            if(!graycode){
                im = i^(i>>1);}
            x[(im-n0)*d+j] = ((double) xc)*scale;
            if(set_rshift==1){
                xr[(im-n0)*d+j] = ((double) (xc^rshift[j]))*scale;}
            /* get the index of the rightmost 0 bit in i */
            b = i; 
            s = 0;
            while(b&1){
                b >>= 1;
                s += 1;}
            /* get the vector used for the next index */
            z1 = znew[j*m_max+s];}}
    return(0);}

/*int main(){return(0);}*/