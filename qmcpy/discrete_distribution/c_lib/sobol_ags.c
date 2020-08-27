#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MRG63k3a.h"

void sobol_ags(unsigned long n, unsigned int d, unsigned int randomize, unsigned long skip, unsigned int graycode, unsigned long seed, double *x, unsigned int d_max, unsigned int m_max, unsigned long *z, unsigned int msb){
    /*
    Custom Sobol' Generator by alegresor

    n: number of samples
    d: dimension
    randomize: 
        0 = None
        1 = linear matrix scramble (LMS) with digital shift (DS)
        2 = DS
    skip: starting index in sequence
    graycode: 
        0 = natural ordering
        1 = graycode ordering
    seed: must supply a seed for the random number generator
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
    */
    int j, m, k;
    unsigned long i, im, u, *rshift;
    unsigned long *xr = (unsigned long *) calloc(d, sizeof(unsigned long));
    double scale = 2. / (1<<m_max);
	seed_MRG63k3a(seed); /* seed the IID RNG */
    if(randomize==1){
        /* LMS */
        unsigned long *sm = (unsigned long *) calloc(m_max, sizeof(unsigned long)); /* scramble matrix */
        unsigned long z1, b; /* scrambled directional number */
        int k1, k2, s;
        for(j=0;j<d;j++){
            memset(sm,0,m_max*sizeof(unsigned long));
            /* initialize the scrambling matrix */
            for(k=1;k<m_max;k++){
                u = (int) (MRG63k3a()*(1<<k)); /* get random int between 0 and 2^k */
                sm[k] = u<<(m_max-k);} /* shift bits to the left to make lower triangular matrix */
            for(k=0;k<m_max;k++){
                sm[k] = sm[k] | (1<<(m_max-1-k));}
            /* left multiply scrambling matrix to directional numbers */
            for(k=0;k<m_max;k++){
                z1 = 0;
                /* lef multiply scrambling matrix by direction number represeted as a column */
                for(k1=0;k1<m_max;k1++){
                    s = 0;
                    b = sm[k1]&z[j*m_max+k];
                    for(k2=0;k2<m_max;k2++){
                        s = s + ((b>>k2)&1);}
                    s = s%2;
                    if(s&&msb){
                        z1 = z1|(1<<(m_max-1-k1));} /* restore (MSB) order */
                    if(s&&(!msb)){
                        z1 = z1|(1<<k1);}} /* restore (LSB) order */
                z[j*m_max+k] = z1;}}}
    if((randomize==1) || (randomize==2)){
        /* initialize DS (will also be applied to LMS) */
        rshift = (unsigned long *) calloc(m_max, sizeof(unsigned long)); /* shift vector */
        for(j=0;j<d;j++){
            rshift[j] = (int) (MRG63k3a()*(1<<(m_max-1)));}}
    /* generate points */
    for(i=skip;i<n+skip;i++){
        m = 0;
        memset(xr,0,d*sizeof(unsigned long));
        if(graycode){
            im = i^(i>>1);}
        else{
            im = i;}
        while((im!=0) && (m<m_max)){
            if(im&1){
                for(j=0;j<d;j++){
                    xr[j] = xr[j] ^ z[j*m_max+m];}}
            im = im>>1;
            m = m+1;}
        if(!msb){
            /* flip bits if using LSB ordering*/
            for(j=0;j<d;j++){
                u = 0;
                for(k=0;k<m_max;k++){
                    u = u|(((xr[j]>>k)&1)<<(m_max-1-k));}
                xr[j] = u;}}
        if((randomize==1) || (randomize==2)){
            /* DS (also applied to LMS) */
            for(j=0;j<d;j++){
                xr[j] = xr[j] ^ rshift[j];}}
        for(j=0;j<d;j++){
            x[(i-skip)*d+j] = ((double) xr[j])*scale;}}}

int main(){
    unsigned long n = 8;
    unsigned int d = 2;
    unsigned int randomize = 0;
    unsigned long skip = 0;
    unsigned int graycode = 0;
    unsigned long seed = 7;
    double *x = (double*) calloc(n*d,sizeof(double));
    unsigned int d_max = 3;
    unsigned int m_max = 32;
    unsigned int msb = 1;
    unsigned long z[3][32] = {
        {2147483648     ,1073741824     ,536870912      ,268435456      ,134217728      ,67108864       ,33554432       ,16777216       ,8388608        ,4194304        ,2097152        ,1048576        ,524288         ,262144         ,131072         ,65536          ,32768          ,16384          ,8192           ,4096           ,2048           ,1024           ,512            ,256            ,128            ,64             ,32             ,16             ,8              ,4              ,2              ,1              },
        {2147483648     ,3221225472     ,2684354560     ,4026531840     ,2281701376     ,3422552064     ,2852126720     ,4278190080     ,2155872256     ,3233808384     ,2694840320     ,4042260480     ,2290614272     ,3435921408     ,2863267840     ,4294901760     ,2147516416     ,3221274624     ,2684395520     ,4026593280     ,2281736192     ,3422604288     ,2852170240     ,4278255360     ,2155905152     ,3233857728     ,2694881440     ,4042322160     ,2290649224     ,3435973836     ,2863311530     ,4294967295     },
        {2147483648     ,3221225472     ,1610612736     ,2415919104     ,3892314112     ,1543503872     ,2382364672     ,3305111552     ,1753219072     ,2629828608     ,3999268864     ,1435500544     ,2154299392     ,3231449088     ,1626210304     ,2421489664     ,3900735488     ,1556135936     ,2388680704     ,3314585600     ,1751705600     ,2627492864     ,4008611328     ,1431684352     ,2147543168     ,3221249216     ,1610649184     ,2415969680     ,3892340840     ,1543543964     ,2382425838     ,3305133397     }};
    /*if(!msb){
        unsigned long z[d_max][m_max] = {
            {1              ,2              ,4              ,8              ,16             ,32             ,64             ,128            ,256            ,512            ,1024           ,2048           ,4096           ,8192           ,16384          ,32768          ,65536          ,131072         ,262144         ,524288         ,1048576        ,2097152        ,4194304        ,8388608        ,16777216       ,33554432       ,67108864       ,134217728      ,268435456      ,536870912      ,1073741824     ,-2147483648    },
	        {1              ,3              ,5              ,15             ,17             ,51             ,85             ,255            ,257            ,771            ,1285           ,3855           ,4369           ,13107          ,21845          ,65535          ,65537          ,196611         ,327685         ,983055         ,1114129        ,3342387        ,5570645        ,16711935       ,16843009       ,50529027       ,84215045       ,252645135      ,286331153      ,858993459      ,1431655765     ,-2147483648    },
	        {1              ,3              ,6              ,9              ,23             ,58             ,113            ,163            ,278            ,825            ,1655           ,2474           ,5633           ,14595          ,30470          ,43529          ,65815          ,197434         ,394865         ,592291         ,1512982        ,3815737        ,7436151        ,10726058       ,18284545       ,54132739       ,108068870      ,161677321      ,370540567      ,960036922      ,2004287601     ,-2147483648    }};}
    */
   sobol_ags(n, d, randomize, skip, graycode, seed, x, d_max, m_max, *z, msb);
    /* print result */
    for(unsigned long i=0; i<n; i++){
        for(int j=0; j<d; j++){
            printf("%.3f\t",x[j*d+i]);}
        printf("\n");}
    return(0);}
