#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MRG63k3a.h"
EXPORT int get_unsigned_long_size()
{
    return sizeof(unsigned long);
}

EXPORT int get_unsigned_long_long_size()
{
    return sizeof(unsigned long long);
}

EXPORT int sobol(unsigned long n, unsigned int d, unsigned long n0, unsigned int d0,
unsigned int randomize, unsigned int graycode, unsigned long long *seeds, double *x, unsigned int d_max,
unsigned int m_max, unsigned long long *z, unsigned int msb){
    /*
    Custom Sobol' Generator by alegresor

    n: sample indicies include n0:(n0+n). Must be a power of 2 if not using Graycode ordering. 
    d: dimension includes d0:(d0+d)
    n0: starting index in sequence. Must be a power of 2 if not using Graycode ordering.
    d0: starting dimension in the sequence
    randomize: 
        0 = None
        1 = linear matrix scramble (LMS) with digital shift (DS)
        2 = DS
    graycode: 
        0 = natural ordering
        1 = graycode ordering
    seed: length d array of seeds, one for each dimension
    x: n x d memory block to store samples
    z: d_max x m_max memory block storing directional numbers
    d_max: max supported dimension
    m_max: max supported samples = 2**m_max
    msb: 
        1 = most significant bit (MSB) in 0^th column of directional numbers in j^th row
        0 = least significant bit (LSB) in 0^th column of direction numbers in j^th row
        Ex 
            if j^th row of directional numbers is
                [1 0 0]
                [0 1 0]
                [0 0 1]
            then MSB order uses int representation [4 2 1] 
            while LSB order has int representation [1 2 4]
        Note that MSB order is faster as it does not require flipping bits
    
    Error Codes:
        1) requires 32 bit precision but system has unsigned int with < 32 bit precision
        2) using natural ordering (graycode=0) and n0 and/or (n0+n) is not 0 or a power of 2
        3) n0+n exceeds 2^m_max or d0+d exceeds d_max

    References:
        
        [1] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
        DOI:https://doi.org/10.1145/42288.214372
    */
    /* parameter checks */
    if( (n==0) || (d==0) ){
        return(0);}
    if(sizeof(unsigned int)<4){
        /* require 32 bit precision */
        return(1);}
    if( (graycode==0) && ( ((n0!=0)&&fmod(log(n0)/log(2),1)!=0) || (fmod(log(n0+n)/log(2),1)!=0) ) ){
        /* for natural ordering, require n0 and (n0+n) be either 0 or powers of 2 */
        return(2);}
    if( ((n0+n)>ldexp(1,m_max)) || ((d0+d)>d_max) ){
        /* too many samples or dimensions */
        return(3);}
    /* variables */
    unsigned int j, m, k, k1, k2, s;
    unsigned long long i, im, u, rshift, xc, xr, z1, b;
    double scale = ldexp(1,-1*m_max);
    unsigned long long *sm = (unsigned long long *) calloc(m_max, sizeof(unsigned long long)); /* scramble matrix */
    unsigned long long *zcp = (unsigned long long *) calloc(m_max, sizeof(unsigned long long));
    /* generate points */
    for(j=0;j<d;j++){
    	seed_MRG63k3a(seeds[j]); /* seed the IID RNG */
        /* LMS */
        if(randomize==1){
            memset(sm,0,m_max*sizeof(unsigned long long));
            /* initialize the scrambling matrix */
            for(k=1;k<m_max;k++){
                u = (unsigned long long) (MRG63k3a() * (((unsigned long long) 1) << k)); /* get random int between 0 and 2^k */
                sm[k] = u << (m_max-k);} /* shift bits to the left to make lower triangular matrix */
            for(k=0;k<m_max;k++){
                sm[k] |= ((unsigned long ) 1) << (m_max-1-k);}
            /* left multiply scrambling matrix to directional numbers */
            for(k=0;k<m_max;k++){
                z1 = 0;
                /* lef multiply scrambling matrix by direction number represeted as a column */
                for(k1=0;k1<m_max;k1++){
                    s = 0;
                    b = sm[k1] & z[(j+d0)*m_max+k];
                    for(k2=0;k2<m_max;k2++){
                        s += (b>>k2)&1;}
                    s %= 2;
                    if(s&&msb){
                        z1 |= ((unsigned long long) 1) << (m_max-1-k1);} /* restore (MSB) order */
                    if(s&&(!msb)){
                        z1 |= ((unsigned long long) 1) << k1;}} /* restore (LSB) order */
                zcp[k] = z1;}}
        /* initialize DS (will also be applied to LMS) */
        if((randomize==1) || (randomize==2)){
            rshift = (unsigned long long) (MRG63k3a()*ldexp(1,m_max));}
        /* copy generating matrix */
        if((randomize==0) || (randomize==2)){
            for(k=0;k<m_max;k++){
                zcp[k] = z[(j+d0)*m_max+k];}}
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
                    xc ^= zcp[m];}
                im >>= 1;
                m += 1;}
            s = 0;
            while(b&1){
                b >>= 1;
                s += 1;}
            z1 = zcp[s];}
        /* set the rest of the points */
        for(i=n0;i<(n0+n);i++){
            xc ^= z1;
            xr = xc; 
            /* flip bits if using LSB ordering*/
            if(!msb){    
                u = 0;
                for(k=0;k<m_max;k++){
                    u |= ((xr>>k)&1)<<(m_max-1-k);}
                xr = u;}
            /* DS (also applied to LMS) */
            if((randomize==1) || (randomize==2)){
                xr ^= rshift;}
            /* set point */
            im = i;
            if(!graycode){
                im = i^(i>>1);}
            x[(im-n0)*d+j] = ((double) xr)*scale;
            /* get the index of the rightmost 0 bit in i */
            b = i; 
            s = 0;
            while(b&1){
                b >>= 1;
                s += 1;}
            /* get the vector used for the next index */
            z1 = zcp[s];}}
    return(0);}

/*
int main(){
    unsigned long n = 2;
    unsigned int d = 2;
    unsigned long n0 = 2; 
    unsigned int d0 = 0;
    unsigned int randomize = 1;
    unsigned int graycode = 0;
    unsigned long seeds[2] = {7,17};
    double *x = (double*) calloc(n*d,sizeof(double));
    unsigned int d_max = 3;
    unsigned int m_max = 32;
    unsigned long z[3][32] = {
        {2147483648     ,1073741824     ,536870912      ,268435456      ,134217728      ,67108864       ,33554432       ,16777216       ,8388608        ,4194304        ,2097152        ,1048576        ,524288         ,262144         ,131072         ,65536          ,32768          ,16384          ,8192           ,4096           ,2048           ,1024           ,512            ,256            ,128            ,64             ,32             ,16             ,8              ,4              ,2              ,1              },
        {2147483648     ,3221225472     ,2684354560     ,4026531840     ,2281701376     ,3422552064     ,2852126720     ,4278190080     ,2155872256     ,3233808384     ,2694840320     ,4042260480     ,2290614272     ,3435921408     ,2863267840     ,4294901760     ,2147516416     ,3221274624     ,2684395520     ,4026593280     ,2281736192     ,3422604288     ,2852170240     ,4278255360     ,2155905152     ,3233857728     ,2694881440     ,4042322160     ,2290649224     ,3435973836     ,2863311530     ,4294967295     },
        {2147483648     ,3221225472     ,1610612736     ,2415919104     ,3892314112     ,1543503872     ,2382364672     ,3305111552     ,1753219072     ,2629828608     ,3999268864     ,1435500544     ,2154299392     ,3231449088     ,1626210304     ,2421489664     ,3900735488     ,1556135936     ,2388680704     ,3314585600     ,1751705600     ,2627492864     ,4008611328     ,1431684352     ,2147543168     ,3221249216     ,1610649184     ,2415969680     ,3892340840     ,1543543964     ,2382425838     ,3305133397     }};
    unsigned int msb = 1;
    unsigned long z[3][32] = {
            {1              ,2              ,4              ,8              ,16             ,32             ,64             ,128            ,256            ,512            ,1024           ,2048           ,4096           ,8192           ,16384          ,32768          ,65536          ,131072         ,262144         ,524288         ,1048576        ,2097152        ,4194304        ,8388608        ,16777216       ,33554432       ,67108864       ,134217728      ,268435456      ,536870912      ,1073741824     ,2147483648     },
	        {1              ,3              ,5              ,15             ,17             ,51             ,85             ,255            ,257            ,771            ,1285           ,3855           ,4369           ,13107          ,21845          ,65535          ,65537          ,196611         ,327685         ,983055         ,1114129        ,3342387        ,5570645        ,16711935       ,16843009       ,50529027       ,84215045       ,252645135      ,286331153      ,858993459      ,1431655765     ,4294967295     },
	        {1              ,3              ,6              ,9              ,23             ,58             ,113            ,163            ,278            ,825            ,1655           ,2474           ,5633           ,14595          ,30470          ,43529          ,65815          ,197434         ,394865         ,592291         ,1512982        ,3815737        ,7436151        ,10726058       ,18284545       ,54132739       ,108068870      ,161677321      ,370540567      ,960036922      ,2004287601     ,2863268003     }};
    unsigned int msb = 0;
    int rc; 
    rc = sobol_ags(n, d, n0, d0, randomize, graycode, seeds, x, d_max, m_max, *z, msb);
    printf("Return code: %d\n\n",rc);
    for(unsigned long i=0; i<n; i++){
        for(int j=0; j<d; j++){
            printf("%.3f\t",x[i*d+j]);}
        printf("\n");} 
    d = 3;
    unsigned long seeds2[3] = {7,17,18};
    double *x2 = (double*) calloc(n*d,sizeof(double));
    rc = sobol_ags(n, d, n0, d0, randomize, graycode, seeds2, x2, d_max, m_max, *z, msb);
    printf("Return code: %d\n\n",rc);
    for(unsigned long i=0; i<n; i++){
        for(int j=0; j<d; j++){
            printf("%.3f\t",x2[i*d+j]);}
        printf("\n");}    
    return(0);}
*/
    